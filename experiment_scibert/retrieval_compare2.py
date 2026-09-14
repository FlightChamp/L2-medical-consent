#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_compare2.py — 후보 청크 검색 비교 (v1 버그 수정판 + 감사)
===================================================================
[v1 버그 — 결과 폐기]
    v1 은 문서를 prompt_ladder.extract_sections() 로 읽고
    prompt_ladder.Splitter.split() 로 문장을 나눴다.
    그런데 정답(cases.csv 의 sent_id)을 만든 nli_sensitivity2.py 는
        docs[name] = TextNormalizer.clean(f.read())      # 원본 그대로
        sents = SentenceSplitter.split(text)             # 최소 길이 필터 없음
    을 쓴다. 두 문장 리스트가 달라 sent_id 가 가리키는 문장이 어긋났고,
    gold chunk 가 사실상 무작위로 정해졌다.
      · n=74 (777건 중 9.5%) — 대부분의 sent_id 가 범위를 벗어남
      · char2gram R@1 = 0.0% — 자기 문장을 포함한 청크가 1위여야 하는데 아님
      · NEGATE / NUM_CHANGE / UNIT_CHANGE R@5 = 0%
    청크 풀도 달라져 end-to-end 값까지 오염됐다
    (char2gram 평균 탐지 59.3 vs 원 실험 k=5 의 83.7).

[v2 수정]
    nli_sensitivity2.py 의 TextNormalizer / SentenceSplitter / ChunkStore 를
    **그대로 import** 한다. 정답을 만든 코드와 동일한 코드로 문장·청크를
    구성하므로 불일치가 원천 차단된다.
    또한 --audit 으로 gold mapping 이 실제로 맞는지 자가 점검한다.

[감사 항목]
    1. denominator 규명      전체/평가대상/제외 건수와 사유
    2. gold mapping sanity   대표 사례의 top-5 를 직접 출력
    3. macro / micro Recall  둘 다 계산해 구분
    4. Specificity 검증      방식별 τ 와 실제 정상 통과율
    5. robustness            문서별 승패, paired table, 문서 클러스터 부트스트랩
    6. k 민감도              k=1/3/5/10
    7. 유형별 사례           UNIT_CHANGE / TERM_SWAP 개선, NEGATE 악화

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python experiment_scibert/retrieval_compare2.py --audit          # 감사만(빠름)
    python experiment_scibert/retrieval_compare2.py                  # 전체
    python experiment_scibert/retrieval_compare2.py --ks 1 3 5 10
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import statistics as st
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    # 정답을 만든 코드와 **동일한** 전처리를 쓴다 (v1 버그의 원인 차단)
    from nli_sensitivity2 import TextNormalizer, SentenceSplitter, ChunkStore
except ImportError as e:
    sys.exit(f"[FATAL] nli_sensitivity2.py 가 상위 폴더에 있어야 합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
CASES = os.path.join(HOME, "nli_sens2_k5_cases.csv")
OUTDIR = os.path.join(HERE, "outputs")

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMB_MODELS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
}
SPAN = 3
TARGET_SPEC = 0.966


def bigrams(t: str) -> Set[str]:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


# ===========================================================================
# 검색기
# ===========================================================================

class Retriever:
    name = "base"

    def scores(self, query: str, doc: str, store) -> List[float]:
        raise NotImplementedError

    def rank(self, query: str, doc: str, store) -> List[int]:
        sc = self.scores(query, doc, store)
        return [i for _, i in sorted(((s, i) for i, s in enumerate(sc)),
                                     key=lambda x: -x[0])]


class Char2Gram(Retriever):
    name = "char2gram"

    def __init__(self):
        self._g: Dict[str, List[Set[str]]] = {}

    def scores(self, query: str, doc: str, store) -> List[float]:
        if doc not in self._g:
            self._g[doc] = [bigrams(c.text) for c in store.chunks]
        q = bigrams(query)
        return [len(q & g) / max(len(q | g), 1) for g in self._g[doc]]


class AllChunks(Retriever):
    name = "all"

    def scores(self, query: str, doc: str, store) -> List[float]:
        return [0.0] * len(store.chunks)

    def rank(self, query: str, doc: str, store) -> List[int]:
        return list(range(len(store.chunks)))


class Embedding(Retriever):
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
        self._c: Dict[str, "torch.Tensor"] = {}
        self._q: Dict[str, "torch.Tensor"] = {}

    def encode(self, texts: List[str]):
        torch = self.torch
        vs = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], truncation=True, padding=True,
                           max_length=256, return_tensors="pt")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                o = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (o * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vs.append(torch.nn.functional.normalize(v, dim=-1))
        return torch.cat(vs, 0)

    def scores(self, query: str, doc: str, store) -> List[float]:
        if doc not in self._c:
            self._c[doc] = self.encode([c.text for c in store.chunks])
        if query not in self._q:
            self._q[query] = self.encode([query])[0]
        return (self._c[doc] @ self._q[query]).tolist()

    def free(self):
        del self.model
        self._c.clear()
        self._q.clear()
        self.torch.cuda.empty_cache()


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


def tau_for_spec(orig: List[float], target: float) -> float:
    xs = sorted(orig)
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

def load_docs(docdir: str) -> Dict[str, str]:
    """nli_sensitivity2.py 와 동일하게 원본을 정규화만 한다."""
    out = {}
    for p in sorted(glob.glob(os.path.join(docdir, "*.txt"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith("syn"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            out[name] = TextNormalizer.clean(f.read())
    return out


def load_cases(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        try:
            sid = int(r["sent_id"])
        except (KeyError, ValueError):
            continue
        out.append({"case_id": i, "doc": r["doc"], "sent_id": sid,
                    "kind": r["kind"], "detail": r.get("detail", ""),
                    "text": r["text"], "best_chunk": r.get("best_chunk", "")})
    return out


# ===========================================================================
# 1. denominator 규명 + gold mapping 감사
# ===========================================================================

def audit_denominator(cases, stores, sents_by_doc) -> dict:
    print("\n" + "=" * 96)
    print("1. denominator 규명 — 몇 건이 평가 대상인가")
    print("=" * 96)
    reasons = {"ok": 0, "doc_missing": 0, "sent_id_out_of_range": 0,
               "no_gold_chunk": 0}
    by_kind_all: Dict[str, int] = {}
    by_kind_ok: Dict[str, int] = {}
    detail = []
    for c in cases:
        by_kind_all[c["kind"]] = by_kind_all.get(c["kind"], 0) + 1
        st_ = stores.get(c["doc"])
        if st_ is None:
            reasons["doc_missing"] += 1
            continue
        n_sents = len(sents_by_doc[c["doc"]])
        if not (0 <= c["sent_id"] < n_sents):
            reasons["sent_id_out_of_range"] += 1
            detail.append((c["doc"], c["sent_id"], n_sents))
            continue
        gold = [i for i, ch in enumerate(st_.chunks) if c["sent_id"] in ch.sent_ids]
        if not gold:
            reasons["no_gold_chunk"] += 1
            continue
        reasons["ok"] += 1
        by_kind_ok[c["kind"]] = by_kind_ok.get(c["kind"], 0) + 1

    print(f"  전체 cases            {len(cases)}")
    print(f"  retrieval 평가 대상    {reasons['ok']}")
    print(f"  제외                  {len(cases) - reasons['ok']}")
    print(f"\n  제외 사유별")
    for k, v in reasons.items():
        if k != "ok" and v:
            print(f"    {k:<26}{v}")
    if detail:
        print(f"    (예: {detail[0][0]} sent_id={detail[0][1]} "
              f"인데 문장 수는 {detail[0][2]})")

    print(f"\n  {'오류 유형':<18}{'전체 n':>9}{'평가 가능 n':>13}{'비율':>9}")
    for kd in sorted(by_kind_all):
        a_, b_ = by_kind_all[kd], by_kind_ok.get(kd, 0)
        print(f"  {kd:<18}{a_:>9}{b_:>13}{100*b_/max(a_,1):>8.1f}%")

    docs_ok: Dict[str, int] = {}
    for c in cases:
        st_ = stores.get(c["doc"])
        if st_ and 0 <= c["sent_id"] < len(sents_by_doc[c["doc"]]):
            docs_ok[c["doc"]] = docs_ok.get(c["doc"], 0) + 1
    print(f"\n  문서별 평가 가능 건수")
    print("    " + ", ".join(f"{d}={n}" for d, n in sorted(docs_ok.items())))
    return {"total": len(cases), "evaluable": reasons["ok"],
            "reasons": reasons, "by_kind_all": by_kind_all,
            "by_kind_ok": by_kind_ok, "by_doc_ok": docs_ok}


def audit_gold(cases, stores, sents_by_doc, retrievers, n_each=3):
    print("\n" + "=" * 96)
    print("2. gold mapping sanity check — 대표 사례의 검색 결과")
    print("=" * 96)
    want = ["ORIGINAL", "NEGATE", "NUM_CHANGE", "UNIT_CHANGE", "TERM_SWAP"]
    for kd in want:
        sel = [c for c in cases if c["kind"] == kd][:n_each]
        if not sel:
            continue
        print(f"\n  ── {kd} ──")
        for c in sel:
            st_ = stores.get(c["doc"])
            sents = sents_by_doc.get(c["doc"], [])
            if not st_ or not (0 <= c["sent_id"] < len(sents)):
                print(f"    [{c['doc']} #{c['case_id']}] sent_id={c['sent_id']} "
                      f"범위 밖 (문장 {len(sents)}개) — 평가 불가")
                continue
            gold = [i for i, ch in enumerate(st_.chunks)
                    if c["sent_id"] in ch.sent_ids]
            print(f"\n    [{c['doc']} #{c['case_id']}] sent_id={c['sent_id']} "
                  f"/ gold chunk {len(gold)}개")
            print(f"      원본  : {sents[c['sent_id']][:78]}")
            print(f"      오염  : {c['text'][:78]}")
            if gold:
                g0 = st_.chunks[gold[0]]
                print(f"      gold[0]: id={gold[0]} sent_ids={g0.sent_ids} "
                      f"| {g0.text[:60]}")
            for R in retrievers:
                sc = R.scores(c["text"], c["doc"], st_)
                order = sorted(range(len(sc)), key=lambda i: -sc[i])[:5]
                hit = "HIT" if set(order) & set(gold) else "miss"
                tops = ", ".join(f"{i}({sc[i]:.3f})" for i in order)
                print(f"      {R.name:<12}[{hit}] {tops}")


# ===========================================================================
# 2. Recall@k
# ===========================================================================

def stage_recall(cases, stores, sents_by_doc, retrievers, ks) -> dict:
    print("\n" + "=" * 96)
    print("3. Recall@k  (오염된 문장을 포함한 청크를 상위 k 에 넣는가)")
    print("=" * 96)
    res = {}
    for R in retrievers:
        hits = {k: 0 for k in ks}
        bykind: Dict[str, Dict[int, List[int]]] = {}
        n = 0
        for c in cases:
            st_ = stores.get(c["doc"])
            sents = sents_by_doc.get(c["doc"], [])
            if not st_ or not (0 <= c["sent_id"] < len(sents)):
                continue
            gold = {i for i, ch in enumerate(st_.chunks)
                    if c["sent_id"] in ch.sent_ids}
            if not gold:
                continue
            n += 1
            order = R.rank(c["text"], c["doc"], st_)
            bykind.setdefault(c["kind"], {k: [] for k in ks})
            for k in ks:
                h = 1 if (set(order[:k]) & gold) else 0
                hits[k] += h
                bykind[c["kind"]][k].append(h)
        res[R.name] = {
            "n": n,
            "recall": {k: round(100 * hits[k] / max(n, 1), 1) for k in ks},
            "by_kind": {kd: {k: (round(100 * st.fmean(v[k]), 1) if v[k] else None)
                             for k in ks} for kd, v in bykind.items()},
            "by_kind_n": {kd: len(v[ks[0]]) for kd, v in bykind.items()},
        }
    print(f"  {'방식':<14}{'n':>6}" + "".join(f"{'R@'+str(k):>9}" for k in ks))
    for name, r in res.items():
        print(f"  {name:<14}{r['n']:>6}"
              + "".join(f"{r['recall'][k]:>9.1f}" for k in ks))

    kinds = sorted({c["kind"] for c in cases})
    print(f"\n  오류 유형별 Recall@5")
    print(f"  {'유형':<18}{'n':>6}" + "".join(f"{n:>14}" for n in res))
    for kd in kinds:
        n_ = next((res[m]["by_kind_n"].get(kd, 0) for m in res), 0)
        line = f"  {kd:<18}{n_:>6}"
        for m in res:
            v = res[m]["by_kind"].get(kd, {}).get(5)
            line += f"{v:>14.1f}" if v is not None else f"{'—':>14}"
        print(line)
    return res


# ===========================================================================
# 3. end-to-end
# ===========================================================================

def stage_end2end(cases, stores, retrievers, nli, k) -> dict:
    print("\n" + "=" * 96)
    print(f"4. 검색 + mDeBERTa  (상위 {k}, 정상 통과율 {TARGET_SPEC:.1%} 고정)")
    print("=" * 96)
    res = {}
    per_case: Dict[str, Dict[int, bool]] = {}
    for R in retrievers:
        prem, hyp, owner, valid = [], [], [], []
        for c in cases:
            st_ = stores.get(c["doc"])
            if not st_:
                continue
            order = R.rank(c["text"], c["doc"], st_)
            for i in (order if k <= 0 else order[:k]):
                prem.append(st_.chunks[i].text)
                hyp.append(c["text"])
                owner.append(len(valid))
            valid.append(c)
        sc = nli.entail(prem, hyp)
        best: Dict[int, float] = {}
        for o, s in zip(owner, sc):
            best[o] = max(best.get(o, -1.0), s)
        scores = [best.get(i, 0.0) for i in range(len(valid))]

        orig = [s for c, s in zip(valid, scores) if c["kind"] == "ORIGINAL"]
        tau = tau_for_spec(orig, TARGET_SPEC)
        spec = 100 * sum(1 for v in orig if v >= tau) / len(orig)

        bykind: Dict[str, List[float]] = {}
        flags: Dict[int, bool] = {}
        for c, s in zip(valid, scores):
            bykind.setdefault(c["kind"], []).append(s)
            if c["kind"] != "ORIGINAL":
                flags[c["case_id"]] = s < tau
        per_case[R.name] = flags

        det = {kd: {"n": len(v),
                    "detect": round(100 * sum(1 for x in v if x < tau) / len(v), 1),
                    "auroc": round(auroc(v, orig), 3)}
               for kd, v in bykind.items() if kd != "ORIGINAL"}
        n_err = sum(d["n"] for d in det.values())
        micro = 100 * sum(d["n"] * d["detect"] / 100 for d in det.values()) / max(n_err, 1)
        res[R.name] = {"tau": round(tau, 4), "spec": round(spec, 1),
                       "n_total": len(valid), "n_orig": len(orig),
                       "n_err": n_err, "by_kind": det,
                       "macro": round(st.fmean(d["detect"] for d in det.values()), 1),
                       "micro": round(micro, 1)}

    print(f"  {'Retriever':<14}{'전체 n':>8}{'정상 n':>8}{'오류 n':>8}"
          f"{'tau':>9}{'Specificity':>13}{'Macro':>9}{'Micro':>9}")
    for m, r in res.items():
        print(f"  {m:<14}{r['n_total']:>8}{r['n_orig']:>8}{r['n_err']:>8}"
              f"{r['tau']:>9.4f}{r['spec']:>12.1f}%{r['macro']:>9.1f}{r['micro']:>9.1f}")

    kinds = sorted({c["kind"] for c in cases if c["kind"] != "ORIGINAL"})
    print(f"\n  {'오류 유형':<18}{'n':>6}" + "".join(f"{m:>14}" for m in res))
    for kd in kinds:
        n_ = next((res[m]["by_kind"][kd]["n"] for m in res
                   if kd in res[m]["by_kind"]), 0)
        line = f"  {kd:<18}{n_:>6}"
        for m in res:
            d = res[m]["by_kind"].get(kd)
            line += f"{d['detect']:>14.1f}" if d else f"{'—':>14}"
        print(line)
    return res, per_case


# ===========================================================================
# 4. robustness
# ===========================================================================

def audit_robust(cases, per_case, base="char2gram", cand="ko-sroberta",
                 n_boot=2000, seed=20260913):
    if base not in per_case or cand not in per_case:
        return {}
    print("\n" + "=" * 96)
    print(f"5. robustness — {cand} vs {base}")
    print("=" * 96)
    bykind = {}
    both = only_b = only_c = neither = 0
    by_doc: Dict[str, List[Tuple[bool, bool]]] = {}
    cid2case = {c["case_id"]: c for c in cases}
    for cid in per_case[base]:
        if cid not in per_case[cand]:
            continue
        b, c_ = per_case[base][cid], per_case[cand][cid]
        kd = cid2case[cid]["kind"]
        bykind.setdefault(kd, [0, 0, 0, 0])
        if b and c_:
            both += 1
            bykind[kd][0] += 1
        elif b:
            only_b += 1
            bykind[kd][1] += 1
        elif c_:
            only_c += 1
            bykind[kd][2] += 1
        else:
            neither += 1
            bykind[kd][3] += 1
        by_doc.setdefault(cid2case[cid]["doc"], []).append((b, c_))

    print(f"  paired 비교 (오류 케이스만)")
    print(f"    둘 다 탐지          {both}")
    print(f"    {base}만 탐지       {only_b}")
    print(f"    {cand}만 탐지  {only_c}")
    print(f"    둘 다 실패          {neither}")
    print(f"\n  {'유형':<18}{'둘다':>8}{'기존만':>8}{'신규만':>8}{'둘다실패':>10}")
    for kd in sorted(bykind):
        v = bykind[kd]
        print(f"  {kd:<18}{v[0]:>8}{v[1]:>8}{v[2]:>8}{v[3]:>10}")

    print(f"\n  문서별 탐지율")
    print(f"  {'문서':<8}{'n':>6}{base:>12}{cand:>14}{'차이':>9}")
    wins = {"cand": 0, "base": 0, "tie": 0}
    for d in sorted(by_doc):
        v = by_doc[d]
        rb = 100 * sum(1 for b, _ in v if b) / len(v)
        rc = 100 * sum(1 for _, c_ in v if c_) / len(v)
        dd = rc - rb
        wins["cand" if dd > 0 else "base" if dd < 0 else "tie"] += 1
        print(f"  {d:<8}{len(v):>6}{rb:>12.1f}{rc:>14.1f}{dd:>+9.1f}")
    print(f"\n    {cand} 우세 {wins['cand']}문서 / "
          f"{base} 우세 {wins['base']}문서 / 동일 {wins['tie']}문서")

    rng = random.Random(seed)
    docs = sorted(by_doc)
    diffs = []
    for _ in range(n_boot):
        pick = [rng.choice(docs) for _ in docs]
        nb = nc = tot = 0
        for d in pick:
            for b, c_ in by_doc[d]:
                nb += b
                nc += c_
                tot += 1
        if tot:
            diffs.append(100 * (nc - nb) / tot)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    obs = (100 * sum(1 for v in by_doc.values() for _, c_ in v if c_)
           - 100 * sum(1 for v in by_doc.values() for b, _ in v if b)) \
          / sum(len(v) for v in by_doc.values())
    print(f"\n  문서 클러스터 부트스트랩 ({n_boot}회, 문서 {len(docs)}개 복원추출)")
    print(f"    관측 차이 (micro)   {obs:+.2f}%p")
    print(f"    95% CI              [{lo:+.2f}, {hi:+.2f}]")
    if lo > 0:
        print(f"    → CI 가 0 을 포함하지 않습니다. 개선이 비교적 안정적입니다.")
    else:
        print(f"    → CI 가 0 을 포함합니다. 이 표본에서 개선을 단정할 수 없습니다.")
    return {"paired": {"both": both, "only_base": only_b,
                       "only_cand": only_c, "neither": neither},
            "by_kind": bykind, "by_doc_wins": wins,
            "bootstrap": {"observed": round(obs, 2),
                          "ci95": [round(lo, 2), round(hi, 2)],
                          "n_boot": n_boot, "unit": "document cluster"}}


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--methods", nargs="+",
                    default=["char2gram", "scibert", "ko-sroberta", "all"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--e2e-ks", nargs="+", type=int, default=[5])
    ap.add_argument("--audit", action="store_true", help="감사만 (NLI 생략)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "retrieval_compare2.json"))
    a = ap.parse_args()

    docs = load_docs(a.docdir)
    cases = load_cases(a.cases)
    sents_by_doc = {d: SentenceSplitter.split(t) for d, t in docs.items()}
    stores = {d: ChunkStore(s, SPAN) for d, s in sents_by_doc.items()}

    print("=" * 96)
    print("후보 청크 검색 방식 비교 (v2 — gold mapping 수정판)")
    print("=" * 96)
    print(f"  전처리: nli_sensitivity2.TextNormalizer + SentenceSplitter + ChunkStore")
    print(f"          (정답을 만든 코드와 동일 — v1 은 extract_sections 를 써서 어긋났음)")
    print(f"  문서 {len(docs)}건 / 케이스 {len(cases)}건")
    print("  문장 수: " + ", ".join(f"{d}={len(s)}"
                                  for d, s in list(sents_by_doc.items())[:6]) + " ...")
    print("  청크 수: " + ", ".join(f"{d}={len(s.chunks)}"
                                  for d, s in list(stores.items())[:6]) + " ...")

    retrievers, embs = [], []
    for m in a.methods:
        if m == "char2gram":
            retrievers.append(Char2Gram())
        elif m == "all":
            retrievers.append(AllChunks())
        elif m in EMB_MODELS:
            e = Embedding(m, EMB_MODELS[m], a.gpu)
            retrievers.append(e)
            embs.append(e)

    out = {"version": "v2", "n_cases": len(cases), "methods": a.methods,
           "target_spec": TARGET_SPEC,
           "preprocessing": "nli_sensitivity2.TextNormalizer/SentenceSplitter/ChunkStore"}

    out["denominator"] = audit_denominator(cases, stores, sents_by_doc)
    audit_gold(cases, stores, sents_by_doc, retrievers)
    out["recall"] = stage_recall(cases, stores, sents_by_doc, retrievers, a.ks)

    if not a.audit:
        nli = NLI(a.gpu, a.batch)
        out["end2end"] = {}
        for k in a.e2e_ks:
            e2e, per_case = stage_end2end(cases, stores, retrievers, nli, k)
            out["end2end"][f"k={k}"] = e2e
            if k == 5:
                out["robustness"] = audit_robust(cases, per_case)

        print("\n" + "=" * 96)
        print("6. 판정")
        print("=" * 96)
        b = out["end2end"]["k=5"].get("char2gram")
        if b:
            for m, r in out["end2end"]["k=5"].items():
                if m == "char2gram":
                    continue
                dma = r["macro"] - b["macro"]
                dmi = r["micro"] - b["micro"]
                v = "개선" if dma >= 3 else "차이 없음" if abs(dma) < 3 else "악화"
                print(f"  {m:<14} Macro {r['macro']:.1f} ({dma:+.1f})  "
                      f"Micro {r['micro']:.1f} ({dmi:+.1f})  → {v}")
        print("""
  해석 주의
    · 임계값을 맞춘 데이터와 성능을 평가한 데이터가 같습니다.
      따라서 이 결과는 **동일 false-positive budget 에서의 in-sample 비교**이며,
      외부 일반화 성능이 아닙니다.
    · SciBERT vanilla checkpoint 는 retrieval 전용 contrastive 학습이 되지
      않았습니다. "off-the-shelf SciBERT representation 을 이용한 retrieval 은
      본 실험 설정에서 개선 효과를 확인하지 못했다" 가 정확한 표현입니다.
    · Recall@k 는 표준 지표, 탐지율은 본 과제 오염 주입 평가셋의 Recall 입니다.""")
        print("=" * 96)

    for e in embs:
        e.free()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
