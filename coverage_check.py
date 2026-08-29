#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_check.py — ④ 한국어 검증은 누락을 잡을 수 있는가
=========================================================
문제:
    현재 ④ 검증은 premise = 원문 청크, hypothesis = 변환문 문장 이다.
    이것은 "변환문이 원문에 근거가 있는가"(환각 탐지)만 묻는다.
    "원문의 내용이 변환문에 다 남아 있는가"(누락 탐지)는 구조상 물어볼 수 없다.

    언어 교차 실험에서 같은 맹점이 확인됐다.
    번역 누락 탐지 AUROC 가 원문→번역 방향에서 0.49~0.55(무작위)였고,
    방향을 뒤집자 0.89~0.92 로 올랐다. 세 언어 모두에서 재현됐다.
    한국어 단일 언어에서도 같은지 확인한다.

설계 — 커버리지 검사:
    premise   = 변환문(모사), hypothesis = 원문 문장
    정상 조건 : 그 문장이 들어있는 문서   → 함의되어야 한다
    누락 조건 : 그 문장을 훼손한 문서     → 함의되지 않아야 한다

누락 3종:
    FULL_OMIT     문장 전체 삭제           항목이 통째로 빠짐
    PARTIAL_OMIT  절 하나 삭제(수치 우선)  세부 조건만 빠짐 — 실제로 가장 흔함
    TERM_OMIT     의료 용어만 제거         용어를 뭉뚱그림
    OFF_TOPIC     무관 문장(코드 정상성 확인용)

    k=5 후보 축소는 앞선 실험에서 확정된 운영값이므로,
    어휘 유사도 상위 5개만 뽑아 채점한다. 전체 청크를 돌지 않아 빠르다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python coverage_check.py
    python coverage_check.py --per-doc 60 --topk 5
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
KINDS = ("FULL_OMIT", "PARTIAL_OMIT", "TERM_OMIT", "OFF_TOPIC")


# ===========================================================================
# 전처리 (기존 실험과 동일 기준)
# ===========================================================================

class Norm:
    CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, t: str) -> str:
        t = unicodedata.normalize("NFKC", t)
        t = cls.CTRL.sub(" ", t)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = cls.WS.sub(" ", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls.BOUND.split(t):
            if not p:
                continue
            s = p.strip()
            if s and not cls.NUM_PREFIX.match(s):
                out.append(s)
        return out


def build_chunks(sents: Sequence[str], span: int = 3) -> List[str]:
    seen, out = set(), []
    for k in range(1, span + 1):
        for i in range(0, len(sents) - k + 1):
            c = " ".join(sents[i:i + k])
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out or ["(빈 문서)"]


def bigrams(t: str) -> Set[str]:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def topk(hypothesis: str, chunks: Sequence[str], k: int) -> List[str]:
    hg = bigrams(hypothesis)
    scored = []
    for c in chunks:
        g = bigrams(c)
        scored.append((len(hg & g) / max(len(hg | g), 1), c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


# ===========================================================================
# 누락 주입
# ===========================================================================

TERMS = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양", "감염",
    "출혈", "혈전", "색전증", "마취", "수혈", "봉합", "절제", "이식", "배액관",
    "합병증", "후유증", "통증", "부작용", "재발", "불유합", "성대마비",
]

OFF_TOPIC = (
    "오늘 서울의 최고 기온은 28도이며 오후 늦게 비가 내리겠습니다.",
    "이 제품의 보증 기간은 구매일로부터 2년이며 소모품은 제외됩니다.",
    "다음 정기 주주총회는 본사 대회의실에서 개최될 예정입니다.",
)

CLAUSE_SPLIT = re.compile(r"(?<=[,،;])\s+|(?<=며)\s+|(?<=고)\s+|(?<=거나)\s+")
NUM_UNIT = re.compile(r"\d{1,4}\s*(주|일|개월|시간|분|년|%|cc|ml|mg|회|명|번)")


class Omitter:
    """원문 문장을 훼손해 '누락된 변환문'을 만든다."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def full(self, sents: List[str], idx: int) -> Optional[List[str]]:
        if len(sents) < 4:
            return None
        return sents[:idx] + sents[idx + 1:]

    def partial(self, sents: List[str], idx: int) -> Optional[List[str]]:
        """절 하나를 삭제한다. 수치가 든 절을 우선 지운다."""
        s = sents[idx]
        parts = [p for p in CLAUSE_SPLIT.split(s) if p.strip()]
        if len(parts) < 2:
            return None
        with_num = [i for i, p in enumerate(parts) if NUM_UNIT.search(p)]
        drop = with_num[0] if with_num else self.rng.randrange(len(parts))
        kept = " ".join(p for i, p in enumerate(parts) if i != drop).strip()
        if len(kept) < 10:
            return None
        return sents[:idx] + [kept] + sents[idx + 1:]

    def term(self, sents: List[str], idx: int) -> Optional[List[str]]:
        s = sents[idx]
        present = [t for t in TERMS if t in s]
        if not present:
            return None
        t = max(present, key=len)
        kept = s.replace(t, "").replace("  ", " ").strip()
        if len(kept) < 10:
            return None
        return sents[:idx] + [kept] + sents[idx + 1:]


# ===========================================================================
# 채점
# ===========================================================================

class Scorer:
    def __init__(self, batch_size: int = 128, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.bs, self.ml = batch_size, max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] device={self.device} model={MODEL_ID}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()
        id2 = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))
        print(f"[INIT] id2label={id2} -> entail={self.i_ent}")

    def entail(self, prem: List[str], hyp: List[str]) -> List[float]:
        torch = self.torch
        out: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hyp[i:i + self.bs],
                           truncation=True, padding=True, max_length=self.ml,
                           return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(p[:, self.i_ent].tolist())
            if i and (i // self.bs) % 20 == 0:
                print(f"    {i}/{len(prem)}", flush=True)
        return out


# ===========================================================================
# 통계
# ===========================================================================

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


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def tau_for_spec(xs: Sequence[float], target: float) -> float:
    s = sorted(xs)
    k = max(0, min(int(round((1.0 - target) * len(s))), len(s) - 1))
    return s[k]


# ===========================================================================

@dataclass
class Case:
    doc: str
    sent_id: int
    kind: str
    hypothesis: str
    premise: str = ""
    score: float = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--per-doc", type=int, default=40, help="문서당 표본 문장 수")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--target-spec", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default="coverage")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    paths = [p for p in sorted(glob.glob(os.path.join(a.docs, "*.txt")))
             if not os.path.basename(p).startswith("syn")]
    if not paths:
        sys.exit(f"[FATAL] {a.docs}/*.txt 없음")
    print(f"[LOAD] 실제 동의서 {len(paths)}건")

    om = Omitter(rng)
    cases: List[Case] = []
    skipped = Counter()

    for path in paths:
        doc = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8", errors="replace") as f:
            sents = Splitter.split(Norm.clean(f.read()))
        pool = [i for i, s in enumerate(sents)
                if a.min_chars <= len(s) <= a.max_chars]
        rng.shuffle(pool)
        picked = sorted(pool[: a.per_doc])
        base_chunks = build_chunks(sents, a.span)

        for idx in picked:
            hyp = sents[idx]
            # 정상 조건 — 문장이 그대로 들어있는 문서
            cases.append(Case(doc, idx, "COMPLETE", hyp,
                              " || ".join(topk(hyp, base_chunks, a.topk))))
            for kind, fn in (("FULL_OMIT", om.full),
                             ("PARTIAL_OMIT", om.partial),
                             ("TERM_OMIT", om.term)):
                mod = fn(sents, idx)
                if mod is None:
                    skipped[kind] += 1
                    continue
                ch = build_chunks(mod, a.span)
                cases.append(Case(doc, idx, kind, hyp,
                                  " || ".join(topk(hyp, ch, a.topk))))
        # 코드 정상성 확인용
        for _ in range(max(3, a.per_doc // 10)):
            o = rng.choice(OFF_TOPIC)
            cases.append(Case(doc, -1, "OFF_TOPIC", o,
                              " || ".join(topk(o, base_chunks, a.topk))))
        print(f"  {doc}: 문장 {len(sents)} / 표본 {len(picked)} "
              f"/ 누적 케이스 {len(cases)}", flush=True)

    if skipped:
        print(f"[SKIP] 적용 불가로 건너뜀: {dict(skipped)}")

    # -- 채점: 후보 5개 각각을 채점해 최댓값을 취한다 ------------------------
    scorer = Scorer(a.batch_size)
    prem, hyp, owner = [], [], []
    for ci, c in enumerate(cases):
        for chunk in c.premise.split(" || "):
            prem.append(chunk)
            hyp.append(c.hypothesis)
            owner.append(ci)
    print(f"[SCORE] 케이스 {len(cases)} → 쌍 {len(prem)}")
    scores = scorer.entail(prem, hyp)
    best: Dict[int, float] = {}
    for o, s in zip(owner, scores):
        if s > best.get(o, -1.0):
            best[o] = s
    for i, c in enumerate(cases):
        c.score = best.get(i, 0.0)

    # -- 집계 ---------------------------------------------------------------
    pos = [c for c in cases if c.kind == "COMPLETE"]
    tau = tau_for_spec([c.score for c in pos], a.target_spec)
    print("\n" + "=" * 76)
    print(f"커버리지 검사 결과   (정상 통과율 {a.target_spec:.0%} 통일)")
    print("=" * 76)
    print(f"  정상 조건 {len(pos)}건  평균 entailment "
          f"{sum(c.score for c in pos)/len(pos):.3f}  보정 임계값 {tau:.3f}\n")
    print(f"  {'누락 유형':<16}{'n':>6}{'평균':>9}{'탐지율':>10}{'AUROC':>9}{'95% CI':>16}")
    summary = {"tau": round(tau, 4), "n_complete": len(pos), "by_kind": {}}
    for kind in KINDS:
        sel = [c for c in cases if c.kind == kind]
        if not sel:
            print(f"  {kind:<16}{0:>6}   (표본 없음)")
            continue
        det = sum(1 for c in sel if c.score < tau)
        au = auroc([c.score for c in sel], [c.score for c in pos])
        lo, hi = wilson(det, len(sel))
        mean = sum(c.score for c in sel) / len(sel)
        summary["by_kind"][kind] = {"n": len(sel), "mean": round(mean, 4),
                                    "detect": round(det / len(sel), 4),
                                    "auroc": round(au, 4)}
        print(f"  {kind:<16}{len(sel):>6}{mean:>9.3f}{det/len(sel):>9.1%}"
              f"{au:>9.3f}   [{lo:.0%},{hi:.0%}]")

    # -- 판정 ---------------------------------------------------------------
    print("\n" + "=" * 76)
    print("판정")
    print("=" * 76)
    bk = summary["by_kind"]
    off = bk.get("OFF_TOPIC", {}).get("auroc", 1.0)
    if off < 0.95:
        print(f"  [!] OFF_TOPIC AUROC {off:.3f} — 0.95 미만입니다.")
        print("      지표 해석 전에 문장 분리·청크 구성을 먼저 점검하세요.")
    else:
        print(f"  코드 정상성 확인 OFF_TOPIC AUROC {off:.3f} — 정상")

    print()
    for kind, label in (("FULL_OMIT", "문장 전체 누락"),
                        ("PARTIAL_OMIT", "절 단위 누락"),
                        ("TERM_OMIT", "용어 누락")):
        v = bk.get(kind)
        if not v:
            continue
        au, det = v["auroc"], v["detect"]
        if au >= 0.90:
            verdict = "커버리지 검사로 잡힙니다"
        elif au >= 0.75:
            verdict = "부분적으로만 잡힙니다 — 보완 필요"
        else:
            verdict = "못 잡습니다 — 별도 모듈 필요"
        print(f"  {label:<14} AUROC {au:.3f} / 탐지율 {det:.1%}  → {verdict}")

    fo = bk.get("FULL_OMIT", {}).get("auroc")
    po = bk.get("PARTIAL_OMIT", {}).get("auroc")
    if fo is not None and po is not None:
        print(f"\n  전체 누락({fo:.3f}) 대비 절 단위 누락({po:.3f}) 차이 {po-fo:+.3f}")
        if po < fo - 0.10:
            print("  → 실제로 가장 흔한 '세부 정보만 빠지는' 누락이 가장 안 잡힙니다.")
            print("     ④ 검증에 커버리지 방향을 추가하더라도 이 유형은 남습니다.")
        else:
            print("  → 두 유형의 난이도가 비슷합니다.")
    print("=" * 76)

    with open(f"{a.out}_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(f"{a.out}_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "sent_id", "kind", "score", "hypothesis", "premise"])
        for c in sorted(cases, key=lambda x: (x.doc, x.sent_id, x.kind)):
            w.writerow([c.doc, c.sent_id, c.kind, round(c.score, 4),
                        c.hypothesis[:300], c.premise[:400]])
    print(f"\n[SAVE] {a.out}_report.json / {a.out}_cases.csv")


if __name__ == "__main__":
    main()
