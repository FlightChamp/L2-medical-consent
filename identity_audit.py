#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identity_audit.py — Identity 가 Gate 에서 탈락한 원인 규명
============================================================
문제:
    Identity 는 out == src 이므로 모든 Safety 지표가 만점이어야 한다.
    그런데 Phase 1 에서 13문서 중 3문서가 Gate 를 통과하지 못했다.
        Gate: 수치보존 ≥ 90 / 수치환각 ≤ 0건 / Grounding ≥ 70
        Identity Grounding 평균 = 74.00  (100 이 아님)

가설:
    동의서 내용의 상당 부분이 서식 항목(`과거병력 (질병, 상해전력) 무 유 미상`)
    이며 명제가 아니다. NLI 는 명제 간 함의를 판정하므로 이런 항목은
    자기 자신과 비교해도 entailment 가 낮게 나온다.
    → 이것은 모델 성능이 아니라 **평가자(evaluator)의 상한**이다.

이 스크립트가 하는 일:
    1. 문서별로 Identity 의 Grounding 을 다시 계산하고 Gate 판정을 재현
    2. 탈락 문서에서 자기 자신에게 함의되지 않은 문장을 전부 출력
    3. 그 문장이 서술문인지 서식 항목인지 분류
    4. 문서별 self-entailment 상한(= Identity Grounding)을 표로 제시

원칙:
    threshold 를 임의로 바꾸지 않는다.
    Gate 기준(Grounding ≥ 70)은 그대로 두고, Identity 값을 측정 상한으로
    별도 표기해 해석에 사용한다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python identity_audit.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, Splitter, build_chunks, topk
    from translate_verify2 import ko_is_pred
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder / translate_verify2 필요: {e}")

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
GROUND_MIN = 70.0          # Phase 1 Gate 와 동일. 수정하지 않는다


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


class Scorer:
    """문장별 최대 entailment 확률이 필요하므로 모델을 직접 올린다.
    translate_verify2.NLI 는 문서 단위 비율(rate)만 돌려주기 때문이다.
    모델·토크나이저·entailment 인덱스 결정 방식은 NLI 와 동일하게 맞췄다."""

    def __init__(self, gpu: int, batch: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.bs = batch
        self.dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[NLI] {NLI_MODEL} → {self.dev}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL).to(self.dev).eval()
        id2 = {int(k): str(v).lower()
               for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))

    def entail(self, prem: List[str], hyp: List[str]) -> List[float]:
        torch = self.torch
        out: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hyp[i:i + self.bs],
                           truncation=True, padding=True, max_length=256,
                           return_tensors="pt")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(p[:, self.i_ent].tolist())
        return out


def per_sentence_scores(scorer: "Scorer", hyps: List[str],
                        chunks: List[str], k: int) -> List[float]:
    """문장별 최대 entailment 확률."""
    prem, hy, owner = [], [], []
    for i, h in enumerate(hyps):
        cand = chunks if k <= 0 else topk(h, chunks, k)
        for c in cand:
            prem.append(c)
            hy.append(h)
            owner.append(i)
    sc = scorer.entail(prem, hy)
    best: Dict[int, float] = {}
    for o, s in zip(owner, sc):
        best[o] = max(best.get(o, -1.0), s)
    return [best.get(i, 0.0) for i in range(len(hyps))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--show", type=int, default=6, help="탈락 문서당 출력할 문장 수")
    ap.add_argument("--out", default="identity_audit.json")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))

    print("=" * 96)
    print("Identity Gate 탈락 audit — 원문을 원문과 비교했는데 왜 탈락하는가")
    print("=" * 96)
    print(f"  문서 {len(docs)}건 / Gate 기준 Grounding ≥ {GROUND_MIN} (수정하지 않음)")
    print(f"  τ = {a.tau}, k = {a.k}")

    nli = Scorer(a.gpu, a.batch)
    rows = []
    print(f"\n  {'문서':<8}{'문장':>6}{'서술문':>8}{'서식항목':>9}"
          f"{'Grounding':>11}{'Gate':>7}  자기함의 실패 유형")
    for doc in docs:
        K = load_K(doc, a.docdir)
        if not K:
            continue
        sents = Splitter.split(K)
        if not sents:
            continue
        chunks = build_chunks(sents)
        scores = per_sentence_scores(nli, sents, chunks, a.k)

        fails = [(i, s, sc) for i, (s, sc) in enumerate(zip(sents, scores))
                 if sc < a.tau]
        ground = 100 * (len(sents) - len(fails)) / len(sents)
        n_pred = sum(1 for s in sents if ko_is_pred(s))
        f_pred = sum(1 for _, s, _ in fails if ko_is_pred(s))
        f_form = len(fails) - f_pred
        passed = ground >= GROUND_MIN

        rows.append({
            "doc": doc, "n_sents": len(sents), "n_pred": n_pred,
            "n_form": len(sents) - n_pred,
            "ground": round(ground, 1), "gate_pass": passed,
            "n_fail": len(fails), "fail_pred": f_pred, "fail_form": f_form,
            "fail_examples": [
                {"sent_id": i, "score": round(sc, 3),
                 "is_predicate": ko_is_pred(s), "text": s[:110]}
                for i, s, sc in sorted(fails, key=lambda x: x[2])[:12]],
        })
        print(f"  {doc:<8}{len(sents):>6}{n_pred:>8}{len(sents)-n_pred:>9}"
              f"{ground:>11.1f}{'통과' if passed else '탈락':>7}"
              f"  서술문 {f_pred} / 서식 {f_form}", flush=True)

    # ── 탈락 문서 상세 ──────────────────────────────────────────────
    failed = [r for r in rows if not r["gate_pass"]]
    print("\n" + "=" * 96)
    print(f"탈락 문서 {len(failed)}건 — 자기 자신에게 함의되지 않은 문장")
    print("=" * 96)
    for r in failed:
        print(f"\n  ── {r['doc']}  Grounding {r['ground']}  "
              f"(문장 {r['n_sents']} 중 {r['n_fail']}건 실패: "
              f"서술문 {r['fail_pred']}, 서식 {r['fail_form']})")
        for e in r["fail_examples"][:a.show]:
            kind = "서술문" if e["is_predicate"] else "서식항목"
            print(f"     [{kind}] entail={e['score']:.3f}  {e['text']}")

    # ── 종합 ───────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("종합")
    print("=" * 96)
    tot_f = sum(r["n_fail"] for r in rows)
    tot_fp = sum(r["fail_pred"] for r in rows)
    tot_ff = sum(r["fail_form"] for r in rows)
    tot_s = sum(r["n_sents"] for r in rows)
    print(f"  전체 문장            {tot_s}")
    print(f"  자기함의 실패        {tot_f}  ({100*tot_f/max(tot_s,1):.1f}%)")
    print(f"    서식 항목          {tot_ff}  ({100*tot_ff/max(tot_f,1):.1f}% of 실패)")
    print(f"    서술문             {tot_fp}  ({100*tot_fp/max(tot_f,1):.1f}% of 실패)")
    g = [r["ground"] for r in rows]
    print(f"\n  Identity Grounding   평균 {st.fmean(g):.2f}  "
          f"최소 {min(g):.1f}  최대 {max(g):.1f}")
    print(f"  Gate 통과            {sum(1 for r in rows if r['gate_pass'])}/{len(rows)}")

    verdict = (tot_ff / max(tot_f, 1)) >= 0.5
    print(f"""
  판정
    {'실패의 대부분이 서식 항목입니다.' if verdict else '실패가 서술문에도 상당히 분포합니다.'}
    Identity 는 정의상 원문과 동일하므로 이 실패는 생성 품질이 아니라
    **평가자(NLI)가 해당 문장 유형을 판정할 수 없다**는 뜻입니다.

    따라서 문서별 Identity Grounding 을 그 문서의 **evaluator ceiling** 으로
    표기하고, 모델 값은 ceiling 대비 상대값으로 함께 해석해야 합니다.
    Gate 기준(Grounding ≥ {GROUND_MIN})은 수정하지 않습니다.""")

    print(f"\n  문서별 evaluator ceiling")
    print(f"  {'문서':<8}{'ceiling':>10}{'Gate':>8}{'서식 비율':>11}")
    for r in sorted(rows, key=lambda x: x["ground"]):
        print(f"  {r['doc']:<8}{r['ground']:>10.1f}"
              f"{'통과' if r['gate_pass'] else '탈락':>8}"
              f"{100*r['n_form']/max(r['n_sents'],1):>10.1f}%")
    print("=" * 96)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"ground_min": GROUND_MIN, "tau": a.tau, "k": a.k,
                   "ceiling_by_doc": {r["doc"]: r["ground"] for r in rows},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
