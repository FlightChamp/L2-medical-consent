#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_compare.py — 후보 청크 검색 방식 비교
================================================
배경:
    현재 ④ 한국어 검증은 이렇게 동작한다.
        생성 문장 → 후보 청크 검색 → 상위 k → mDeBERTa NLI → 근거율/커버리지
    후보 검색에 **문자 2-gram Jaccard** 를 쓰고 있다.
    k=5 로 좁히면 오탐 예산 동일 조건에서 탐지율이 평균 +27.7%p 올랐으나,
    용어 대치는 43.8% 에 머물렀다. 어휘가 겹치지 않는 의미적 오류를
    문자 겹침으로 찾는 데 한계가 있다는 가설이 가능하다.

질문:
    후보 검색을 의미 기반 임베딩으로 바꾸면 개선되는가?

비교하는 네 방식:
    A. char2gram   현재 방식. 공백 제거 후 문자 2-gram Jaccard
    B. scibert     allenai/scibert_scivocab_uncased 임베딩 코사인
                   (한국어 [UNK] 문제가 예상되나 대조군으로 포함한다)
    C. ko-sroberta jhgan/ko-sroberta-multitask 임베딩 코사인
                   한국어 문장 유사도로 contrastive 학습된 모델
    D. all         전체 청크 (검색 없음). 상한선 참고용

정답 정의 (Recall@k):
    nli_sensitivity2.py 가 만든 케이스는 원문 문장 하나를 오염시킨 것이고,
    cases.csv 에 그 문장의 sent_id 가 기록되어 있다.
    청크는 연속 1~3문장이며 어떤 문장들로 구성됐는지(sent_ids) 재구성 가능하다.
    → **오염된 문장을 포함하는 청크**가 정답이다.
       검색 결과 상위 k 안에 정답 청크가 하나라도 있으면 hit.

두 단계로 평가한다:
    1단계  Recall@1 / @3 / @5 / @10        검색 자체의 성능
    2단계  + mDeBERTa → 오류 유형별 탐지율  실제 검증 성능
           동일 Specificity(정상 통과율) 조건에서 비교한다.

주의:
    SciBERT vanilla checkpoint 는 retrieval 전용으로 contrastive 학습된
    모델이 아니다. 결과가 나쁘더라도 "SciBERT 는 semantic retrieval 에
    사용할 수 없다" 고 일반화하지 않는다. 정확한 표현은
    "off-the-shelf SciBERT representation 을 이용한 retrieval 은
     본 실험 설정에서 개선 효과를 확인하지 못했다" 이다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python experiment_scibert/retrieval_compare.py --stage recall     # 1단계만
    python experiment_scibert/retrieval_compare.py                    # 1+2단계
    python experiment_scibert/retrieval_compare.py --methods char2gram ko-sroberta
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
# 기존 스크립트는 상위 폴더(~/이윤우)에 있다. 같은 폴더에 두는 경우도 지원한다.
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, Splitter
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py 가 상위 폴더에 있어야 합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
CASES = os.path.join(HOME, "nli_sens2_k5_cases.csv")
OUTDIR = os.path.join(HERE, "outputs")

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMB_MODELS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
}
SPAN = 3                 # 청크 = 연속 1~3문장 (nli_sensitivity2 와 동일)
TARGET_SPEC = 0.966      # 기존 실험과 동일한 정상 통과율


# ===========================================================================
# 청크 — nli_sensitivity2.py 와 동일한 구성 규칙
# ===========================================================================

@dataclass
class Chunk:
    text: str
    sent_ids: Tuple[int, ...]


def build_chunks_with_ids(sents: List[str], span: int = SPAN) -> List[Chunk]:
    out: List[Chunk] = []
    seen: Set[str] = set()
    for k in range(1, span + 1):
        for i in range(0, len(sents) - k + 1):
            t = " ".join(sents[i:i + k])
            if t in seen:
                continue
            seen.add(t)
            out.append(Chunk(t, tuple(range(i, i + k))))
    return out or [Chunk("(빈 문서)", (0,))]


def bigrams(t: str) -> Set[str]:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


# ===========================================================================
# 검색기
# ===========================================================================

class Retriever:
    name = "base"

    def rank(self, query: str, chunks: List[Chunk]) -> List[int]:
        """점수 내림차순 청크 인덱스."""
        raise NotImplementedError


class Char2Gram(Retriever):
    name = "char2gram"

    def __init__(self):
        self._cache: Dict[int, List[Set[str]]] = {}

    def rank(self, query: str, chunks: List[Chunk]) -> List[int]:
        key = id(chunks)
        if key not in self._cache:
            self._cache[key] = [bigrams(c.text) for c in chunks]
        grams = self._cache[key]
        q = bigrams(query)
        sc = [(len(q & g) / max(len(q | g), 1), i) for i, g in enumerate(grams)]
        sc.sort(key=lambda x: -x[0])
        return [i for _, i in sc]


class AllChunks(Retriever):
    name = "all"

    def rank(self, query: str, chunks: List[Chunk]) -> List[int]:
        return list(range(len(chunks)))


class Embedding(Retriever):
    """평균 풀링 임베딩 코사인 유사도."""

    def __init__(self, name: str, model_id: str, gpu: int, batch: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = name
        self.bs = batch
        self.dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"  [EMB] {name}: {model_id} → {self.dev}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.dev).eval()
        self._cache: Dict[int, "torch.Tensor"] = {}
        self._qcache: Dict[str, "torch.Tensor"] = {}

    def encode(self, texts: List[str]):
        torch = self.torch
        vecs = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], truncation=True, padding=True,
                           max_length=256, return_tensors="pt")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(torch.nn.functional.normalize(v, dim=-1))
        return torch.cat(vecs, 0)

    def rank(self, query: str, chunks: List[Chunk]) -> List[int]:
        torch = self.torch
        key = id(chunks)
        if key not in self._cache:
            self._cache[key] = self.encode([c.text for c in chunks])
        C = self._cache[key]
        if query not in self._qcache:
            self._qcache[query] = self.encode([query])[0]
        q = self._qcache[query]
        sim = (C @ q).tolist()
        return [i for _, i in sorted(((s, i) for i, s in enumerate(sim)),
                                     key=lambda x: -x[0])]

    def free(self):
        del self.model
        self._cache.clear()
        self._qcache.clear()
        self.torch.cuda.empty_cache()


# ===========================================================================
# NLI
# ===========================================================================

class NLI:
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


def tau_for_spec(orig_scores: List[float], target: float) -> float:
    """정상 사례의 target 비율이 통과하는 임계값 (equalize_spec.py 와 동일)."""
    xs = sorted(orig_scores)
    k = int(round((1.0 - target) * len(xs)))
    return xs[max(0, min(k, len(xs) - 1))]


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = [-x for x in pos], [-x for x in neg]
    allv = sorted(p + n)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = sum(ranks[x] for x in p)
    return (r - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))


# ===========================================================================

def load_cases(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            sid = int(r["sent_id"])
        except (KeyError, ValueError):
            continue
        out.append({"doc": r["doc"], "sent_id": sid, "kind": r["kind"],
                    "detail": r.get("detail", ""), "text": r["text"]})
    return out


def load_doc_sents(docdir: str) -> Dict[str, List[str]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(docdir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("syn"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            secs = extract_sections(f.read())
        if secs:
            out[stem] = Splitter.split(" ".join(b for _, b in secs))
    return out


# ===========================================================================
# 1단계 — Recall@k
# ===========================================================================

def stage_recall(cases: List[dict], chunks_by_doc: Dict[str, List[Chunk]],
                 retrievers: List[Retriever], ks: List[int]) -> dict:
    print("\n" + "=" * 92)
    print("1단계 — Recall@k  (오염된 문장을 포함한 청크를 상위 k 에 넣는가)")
    print("=" * 92)
    res: Dict[str, Dict] = {}
    for R in retrievers:
        hits = {k: 0 for k in ks}
        bykind: Dict[str, Dict[int, List[int]]] = {}
        n = 0
        t0 = time.time()
        for c in cases:
            chunks = chunks_by_doc.get(c["doc"])
            if not chunks:
                continue
            gold = {i for i, ch in enumerate(chunks) if c["sent_id"] in ch.sent_ids}
            if not gold:
                continue
            n += 1
            order = R.rank(c["text"], chunks)
            bykind.setdefault(c["kind"], {k: [] for k in ks})
            for k in ks:
                hit = 1 if (set(order[:k]) & gold) else 0
                hits[k] += hit
                bykind[c["kind"]][k].append(hit)
        el = time.time() - t0
        res[R.name] = {
            "n": n, "elapsed": round(el, 1),
            "recall": {k: round(100 * hits[k] / max(n, 1), 1) for k in ks},
            "by_kind": {kd: {k: round(100 * st.fmean(v[k]), 1) if v[k] else None
                             for k in ks} for kd, v in bykind.items()},
        }
        print(f"  {R.name:<14} n={n:<5} "
              + "  ".join(f"R@{k}={res[R.name]['recall'][k]:5.1f}" for k in ks)
              + f"   [{el:.0f}s]")

    print(f"\n  {'방식':<14}" + "".join(f"{'R@'+str(k):>9}" for k in ks))
    for name, r in res.items():
        print(f"  {name:<14}" + "".join(f"{r['recall'][k]:>9.1f}" for k in ks))

    kinds = sorted({c["kind"] for c in cases})
    print(f"\n  오류 유형별 Recall@5")
    print(f"  {'유형':<18}" + "".join(f"{n:>14}" for n in res))
    for kd in kinds:
        line = f"  {kd:<18}"
        for name in res:
            v = res[name]["by_kind"].get(kd, {}).get(5)
            line += f"{v:>14.1f}" if v is not None else f"{'—':>14}"
        print(line)
    return res


# ===========================================================================
# 2단계 — end-to-end
# ===========================================================================

def stage_end2end(cases: List[dict], chunks_by_doc: Dict[str, List[Chunk]],
                  retrievers: List[Retriever], nli: NLI, k: int) -> dict:
    print("\n" + "=" * 92)
    print(f"2단계 — 검색 + mDeBERTa  (상위 {k}, 정상 통과율 {TARGET_SPEC:.1%} 고정)")
    print("=" * 92)
    res: Dict[str, Dict] = {}
    for R in retrievers:
        prem, hyp, owner = [], [], []
        valid = []
        for ci, c in enumerate(cases):
            chunks = chunks_by_doc.get(c["doc"])
            if not chunks:
                continue
            order = R.rank(c["text"], chunks)
            pick = order if k <= 0 else order[:k]
            for i in pick:
                prem.append(chunks[i].text)
                hyp.append(c["text"])
                owner.append(len(valid))
            valid.append(c)
        sc = nli.entail(prem, hyp)
        best: Dict[int, float] = {}
        for o, s in zip(owner, sc):
            best[o] = max(best.get(o, -1.0), s)
        scores = [best.get(i, 0.0) for i in range(len(valid))]

        orig = [s for c, s in zip(valid, scores) if c["kind"] == "ORIGINAL"]
        if not orig:
            print(f"  {R.name}: ORIGINAL 케이스가 없어 임계값을 정할 수 없습니다")
            continue
        tau = tau_for_spec(orig, TARGET_SPEC)

        bykind: Dict[str, List[float]] = {}
        for c, s in zip(valid, scores):
            bykind.setdefault(c["kind"], []).append(s)
        det = {}
        for kd, vals in bykind.items():
            if kd == "ORIGINAL":
                continue
            det[kd] = {
                "n": len(vals),
                "detect": round(100 * sum(1 for v in vals if v < tau) / len(vals), 1),
                "auroc": round(auroc(vals, orig), 3),
            }
        spec = 100 * sum(1 for v in orig if v >= tau) / len(orig)
        res[R.name] = {"tau": round(tau, 4), "spec": round(spec, 1),
                       "n_orig": len(orig), "by_kind": det,
                       "mean_detect": round(st.fmean(d["detect"] for d in det.values()), 1)}
        print(f"  {R.name:<14} tau={tau:.4f}  정상통과 {spec:.1f}%  "
              f"평균 탐지 {res[R.name]['mean_detect']:.1f}%")

    kinds = sorted({c["kind"] for c in cases if c["kind"] != "ORIGINAL"})
    print(f"\n  {'오류 유형':<18}{'n':>6}" + "".join(f"{n:>14}" for n in res))
    for kd in kinds:
        n_ = next((res[m]["by_kind"][kd]["n"] for m in res
                   if kd in res[m]["by_kind"]), 0)
        line = f"  {kd:<18}{n_:>6}"
        for m in res:
            d = res[m]["by_kind"].get(kd)
            line += f"{d['detect']:>14.1f}" if d else f"{'—':>14}"
        print(line)
    line = f"  {'평균':<18}{'':>6}"
    for m in res:
        line += f"{res[m]['mean_detect']:>14.1f}"
    print(line)
    return res


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--methods", nargs="+",
                    default=["char2gram", "scibert", "ko-sroberta", "all"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--e2e-k", type=int, default=5)
    ap.add_argument("--stage", choices=["recall", "both"], default="both")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "retrieval_compare.json"))
    a = ap.parse_args()

    if not os.path.exists(a.cases):
        sys.exit(f"[FATAL] {a.cases} 없음")
    cases = load_cases(a.cases)
    sents = load_doc_sents(a.docdir)
    chunks_by_doc = {d: build_chunks_with_ids(s) for d, s in sents.items()}

    print("=" * 92)
    print("후보 청크 검색 방식 비교")
    print("=" * 92)
    print(f"  케이스 {len(cases)}건 / 문서 {len(sents)}건")
    print(f"  청크 수: " + ", ".join(f"{d}={len(c)}"
                                    for d, c in list(chunks_by_doc.items())[:5]) + " ...")
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"  유형별: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"  정답 정의: 오염된 문장(sent_id)을 포함하는 청크")

    retrievers: List[Retriever] = []
    embs: List[Embedding] = []
    for m in a.methods:
        if m == "char2gram":
            retrievers.append(Char2Gram())
        elif m == "all":
            retrievers.append(AllChunks())
        elif m in EMB_MODELS:
            e = Embedding(m, EMB_MODELS[m], a.gpu)
            retrievers.append(e)
            embs.append(e)
        else:
            print(f"  [경고] 알 수 없는 방식: {m}")

    out = {"n_cases": len(cases), "methods": a.methods,
           "ks": a.ks, "target_spec": TARGET_SPEC}
    out["recall"] = stage_recall(cases, chunks_by_doc, retrievers, a.ks)

    if a.stage == "both":
        nli = NLI(a.gpu, a.batch)
        out["end2end"] = stage_end2end(cases, chunks_by_doc, retrievers,
                                       nli, a.e2e_k)

    # 판정
    print("\n" + "=" * 92)
    print("판정")
    print("=" * 92)
    base = out["recall"].get("char2gram")
    if base:
        for name, r in out["recall"].items():
            if name in ("char2gram", "all"):
                continue
            d5 = r["recall"][5] - base["recall"][5]
            print(f"  {name:<14} Recall@5 {r['recall'][5]:.1f} "
                  f"(기존 {base['recall'][5]:.1f}, {d5:+.1f})")
    if "end2end" in out:
        b = out["end2end"].get("char2gram")
        if b:
            for name, r in out["end2end"].items():
                if name == "char2gram":
                    continue
                d = r["mean_detect"] - b["mean_detect"]
                verdict = ("개선" if d >= 3 else
                           "차이 없음" if abs(d) < 3 else "악화")
                print(f"  {name:<14} 평균 탐지 {r['mean_detect']:.1f} "
                      f"(기존 {b['mean_detect']:.1f}, {d:+.1f}) → {verdict}")
    print("""
  해석 주의
    · SciBERT vanilla checkpoint 는 retrieval 전용 contrastive 학습이 되지
      않은 모델입니다. 결과가 나쁘더라도 "semantic retrieval 에 쓸 수 없다"
      고 일반화하지 마십시오. "off-the-shelf SciBERT representation 을
      이용한 retrieval 은 본 실험 설정에서 개선 효과를 확인하지 못했다"
      가 정확한 표현입니다.
    · Recall@k 는 표준 지표이고, 탐지율은 본 과제의 오염 주입 평가셋에
      대한 Recall 입니다.""")
    print("=" * 92)

    for e in embs:
        e.free()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
