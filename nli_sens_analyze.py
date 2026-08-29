#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nli_sens_analyze.py — NLI 민감도 실험 결과 사후 분석 (GPU 불필요)
=================================================================
nli_sensitivity.py 가 남긴 nli_sensitivity_cases.csv 만 가지고
재실행 없이 다음을 계산한다.

[0] 데이터 개요            표본 수 / 문서 규모 분포
[1] 쌍대 순위 정확도       임계값 무관. 같은 문장의 (원문, 오염문) 쌍에서
                          오염문이 더 낮은 점수를 받은 비율
[2] AUROC                  임계값 무관. 원문 분포와 오염 분포의 분리 정도
                          0.5 = 무작위, 1.0 = 완전 분리
[3] 임계값 표              지정 tau에서 유형별 탐지율
[4] 실제 vs 합성 분리      syn* 문서가 결과를 끌어올렸는지 확인
[5] 청크 수 교란 점검      문서가 길수록(청크가 많을수록) 점수가 뜨는지
                          → CometKiwi의 '입력 분할이 점수를 좌우' 문제와 동일 검사
[6] 놓친 케이스 덤프       탐지 실패 사례를 유형별로 출력 (사람 눈 검증용)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python nli_sens_analyze.py
    python nli_sens_analyze.py --tau 0.5 --dump-kind TERM_SWAP --dump-n 20
    python nli_sens_analyze.py --exclude-syn        # 합성문서 빼고 재집계
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# nli_sensitivity.py 와 동일한 전처리 (청크 수를 같은 기준으로 세기 위함)
# ---------------------------------------------------------------------------


class TextNormalizer:
    _CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = cls._CTRL.sub(" ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = cls._WS.sub(" ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


class SentenceSplitter:
    _NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    _BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, text: str) -> List[str]:
        out = []
        for piece in cls._BOUND.split(text):
            if not piece:
                continue
            s = piece.strip()
            if s and not cls._NUM_PREFIX.match(s):
                out.append(s)
        return out


def count_chunks(sents: List[str], span: int = 3) -> int:
    seen = set()
    for k in range(1, span + 1):
        for i in range(0, len(sents) - k + 1):
            seen.add(" ".join(sents[i:i + k]))
    return len(seen)


# ---------------------------------------------------------------------------
# 통계 유틸 (외부 의존성 없음)
# ---------------------------------------------------------------------------


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """pos(오염)의 점수가 neg(원문)보다 '낮을수록' 좋은 상황.

    탐지 관점으로 뒤집어, 부호를 반전한 뒤 표준 순위 기반 AUC를 계산한다.
    반환값 1.0 = 오염이 항상 더 낮은 점수, 0.5 = 무작위.
    """
    if not pos or not neg:
        return float("nan")
    p = [-x for x in pos]
    n = [-x for x in neg]
    allv = sorted(p + n)
    ranks: Dict[float, float] = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        ranks[allv[i]] = r
        i = j + 1
    rsum = sum(ranks[x] for x in p)
    return (rsum - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """이항 비율의 95% 신뢰구간. 표본이 작을 때 정직하게 보고하기 위함."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------


class Row(dict):
    @property
    def key(self):
        return (self["doc"], self["sent_id"])


def load_cases(path: str, exclude_syn: bool) -> List[Row]:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {path} 없음. nli_sensitivity.py 를 먼저 실행하세요.")
    rows: List[Row] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if exclude_syn and r["doc"].startswith("syn"):
                continue
            rows.append(Row({
                "doc": r["doc"],
                "sent_id": int(r["sent_id"]),
                "kind": r["kind"],
                "detail": r["detail"],
                "collides": r["collides"] == "1",
                "max_ent": float(r["max_ent"]),
                "max_con": float(r["max_con"]),
                "text": r["text"],
                "best_chunk": r["best_chunk"],
            }))
    return rows


def doc_sizes(docs_dir: str) -> Dict[str, Tuple[int, int]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8", errors="replace") as f:
            sents = SentenceSplitter.split(TextNormalizer.clean(f.read()))
        out[name] = (len(sents), count_chunks(sents))
    return out


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


class Analyzer:
    def __init__(self, rows: List[Row], sizes: Dict[str, Tuple[int, int]], tau: float):
        self.rows = rows
        self.sizes = sizes
        self.tau = tau
        self.orig = [r for r in rows if r["kind"] == "ORIGINAL"]
        self.kinds = sorted({r["kind"] for r in rows if r["kind"] != "ORIGINAL"})
        self.by_kind = {k: [r for r in rows if r["kind"] == k] for k in self.kinds}
        self.orig_index = {r.key: r for r in self.orig}

    # -- [0] --------------------------------------------------------------
    def overview(self):
        docs = sorted({r["doc"] for r in self.rows})
        print("\n[0] 데이터 개요")
        print(f"  케이스 {len(self.rows)}건 / 원문 {len(self.orig)}건 / 문서 {len(docs)}건")
        known = [(d, self.sizes[d][1]) for d in docs if d in self.sizes]
        if known:
            known.sort(key=lambda x: x[1])
            print(f"  청크 수 범위: {known[0][0]} {known[0][1]}개 "
                  f"~ {known[-1][0]} {known[-1][1]}개 "
                  f"({known[-1][1] / max(known[0][1], 1):.1f}배 차이)")

    # -- [1] --------------------------------------------------------------
    def paired_rank(self):
        print(f"\n[1] 쌍대 순위 정확도 — 임계값 무관")
        print("    같은 문장에서 나온 (원문, 오염문) 쌍 중 오염문 점수가 더 낮은 비율")
        print(f"    {'유형':<16}{'n':>5}{'순위정확도':>12}{'95% CI':>18}")
        for k in sorted(self.kinds,
                        key=lambda k: -self._paired(k)[0] if self._paired(k)[1] else 0):
            win, n = self._paired(k)
            if n == 0:
                continue
            lo, hi = wilson(round(win * n), n)
            print(f"    {k:<16}{n:>5}{win:>11.1%}   [{lo:.1%}, {hi:.1%}]")

    def _paired(self, kind: str) -> Tuple[float, int]:
        wins = tot = 0
        for r in self.by_kind[kind]:
            o = self.orig_index.get(r.key)
            if o is None:
                continue
            tot += 1
            if r["max_ent"] < o["max_ent"]:
                wins += 1
        return (wins / tot if tot else 0.0, tot)

    # -- [2] --------------------------------------------------------------
    def auc_table(self):
        print("\n[2] AUROC — 임계값 무관 (0.5=무작위, 1.0=완전 분리)")
        neg = [r["max_ent"] for r in self.orig]
        res = []
        for k in self.kinds:
            pos = [r["max_ent"] for r in self.by_kind[k]]
            res.append((k, auroc(pos, neg), len(pos)))
        print(f"    {'유형':<16}{'n':>5}{'AUROC':>10}   판정")
        for k, a, n in sorted(res, key=lambda x: -x[1]):
            verdict = ("양호" if a >= 0.90 else
                       "보통" if a >= 0.75 else
                       "취약" if a >= 0.60 else "사실상 무반응")
            flag = "  <== " + verdict if a < 0.75 else "  " + verdict
            print(f"    {k:<16}{n:>5}{a:>10.3f}{flag}")

    # -- [3] --------------------------------------------------------------
    def tau_table(self):
        print(f"\n[3] 임계값 tau={self.tau} 에서의 탐지율")
        spec = mean([1.0 if r["max_ent"] >= self.tau else 0.0 for r in self.orig])
        real = [r for r in self.rows if r["kind"] not in ("ORIGINAL", "OFF_TOPIC")]
        rec = mean([1.0 if r["max_ent"] < self.tau else 0.0 for r in real])
        print(f"    정상 통과율 {spec:.1%}   전체 탐지율 {rec:.1%} (OFF_TOPIC 제외)")
        print(f"    {'유형':<16}{'n':>5}{'탐지율':>10}{'95% CI':>18}")
        rows = []
        for k in self.kinds:
            sel = self.by_kind[k]
            d = sum(1 for r in sel if r["max_ent"] < self.tau)
            rows.append((k, d, len(sel)))
        for k, d, n in sorted(rows, key=lambda x: -(x[1] / x[2]) if x[2] else 0):
            lo, hi = wilson(d, n)
            warn = "   [표본 부족]" if n < 30 else ""
            print(f"    {k:<16}{n:>5}{d / n:>9.1%}   [{lo:.1%}, {hi:.1%}]{warn}")

    # -- [4] --------------------------------------------------------------
    def real_vs_syn(self):
        groups = {
            "실제 동의서": [r for r in self.rows if not r["doc"].startswith("syn")],
            "합성 문서": [r for r in self.rows if r["doc"].startswith("syn")],
        }
        if not groups["합성 문서"]:
            return
        print("\n[4] 실제 vs 합성 문서 분리")
        print(f"    {'집단':<14}{'원문평균':>10}{'오염평균':>10}{'탐지율':>10}{'n':>7}")
        for name, rs in groups.items():
            o = [r["max_ent"] for r in rs if r["kind"] == "ORIGINAL"]
            p = [r for r in rs if r["kind"] not in ("ORIGINAL", "OFF_TOPIC")]
            det = mean([1.0 if r["max_ent"] < self.tau else 0.0 for r in p])
            print(f"    {name:<14}{mean(o):>10.3f}{mean([r['max_ent'] for r in p]):>10.3f}"
                  f"{det:>10.1%}{len(rs):>7}")
        print("    → 두 집단 탐지율 차이가 크면 합성문서가 전체 수치를 왜곡한 것")

    # -- [5] --------------------------------------------------------------
    def chunk_confound(self):
        print("\n[5] 청크 수 교란 점검 — 문서가 길수록 점수가 뜨는가")
        stats = []
        for doc in sorted({r["doc"] for r in self.rows}):
            if doc not in self.sizes:
                continue
            nch = self.sizes[doc][1]
            o = [r["max_ent"] for r in self.rows
                 if r["doc"] == doc and r["kind"] == "ORIGINAL"]
            off = [r["max_ent"] for r in self.rows
                   if r["doc"] == doc and r["kind"] == "OFF_TOPIC"]
            if o and off:
                stats.append((doc, nch, mean(o), mean(off)))
        if len(stats) < 3:
            print("    비교 가능한 문서가 부족합니다.")
            return
        stats.sort(key=lambda x: x[1])
        print(f"    {'문서':<8}{'청크':>7}{'원문평균':>10}{'무관문장평균':>14}")
        for d, n, o, f in stats:
            print(f"    {d:<8}{n:>7}{o:>10.3f}{f:>14.3f}")
        r_off = pearson([s[1] for s in stats], [s[3] for s in stats])
        r_org = pearson([s[1] for s in stats], [s[2] for s in stats])
        print(f"\n    상관계수 r(청크 수, 무관문장 점수) = {r_off:+.3f}")
        print(f"    상관계수 r(청크 수, 원문 점수)     = {r_org:+.3f}")
        if r_off > 0.4:
            print("    → 무관한 문장인데도 문서가 길수록 점수가 오릅니다.")
            print("       entailment 최댓값이 청크 수에 비례해 부풀려지는 극값 문제입니다.")
            print("       문서 간·길이 간 비교에 그대로 쓰면 안 되고, 길이 보정이 필요합니다.")
        else:
            print("    → 청크 수에 따른 체계적 편향은 관찰되지 않습니다.")

    # -- [6] --------------------------------------------------------------
    def dump_missed(self, kind: str, n: int):
        sel = [r for r in self.by_kind.get(kind, []) if r["max_ent"] >= self.tau]
        print(f"\n[6] 탐지 실패 케이스 — {kind} ({len(sel)}건 중 상위 {min(n, len(sel))}건)")
        if not sel:
            print("    없음")
            return
        for r in sorted(sel, key=lambda x: -x["max_ent"])[:n]:
            o = self.orig_index.get(r.key)
            print(f"\n  · {r['doc']} #{r['sent_id']}  ent={r['max_ent']:.3f}"
                  f"  (원문 {o['max_ent']:.3f})" if o else "")
            print(f"    변경: {r['detail']}"
                  + ("   [충돌: 대체어가 원문에 이미 존재]" if r["collides"] else ""))
            print(f"    오염문: {r['text'][:110]}")
            print(f"    매칭청크: {r['best_chunk'][:110]}")

    # -- 종합 판정 ---------------------------------------------------------
    def verdict(self):
        print("\n" + "=" * 74)
        print("종합")
        print("=" * 74)
        neg = [r["max_ent"] for r in self.orig]
        weak = []
        for k in self.kinds:
            if k == "OFF_TOPIC":
                continue
            a = auroc([r["max_ent"] for r in self.by_kind[k]], neg)
            if a < 0.75:
                weak.append((k, a, len(self.by_kind[k])))
        if weak:
            print("  NLI가 신뢰할 수 없는 오류 유형 (AUROC < 0.75):")
            for k, a, n in sorted(weak, key=lambda x: x[1]):
                small = " — 표본 부족, 확대 재측정 필요" if n < 30 else ""
                print(f"    · {k}  AUROC {a:.3f} (n={n}){small}")
            print("  → 이 유형들은 NLI 단독으로 못 잡습니다. 규칙 기반 모듈이 필요합니다.")
        else:
            print("  전 유형 AUROC 0.75 이상 — NLI 단독으로 방어 가능한 범위입니다.")
        off_a = auroc([r["max_ent"] for r in self.by_kind.get("OFF_TOPIC", [])], neg)
        print(f"\n  코드 정상 동작 확인용 OFF_TOPIC AUROC = {off_a:.3f}")
        print("  (0.95 미만이면 지표 해석 전에 청크 구성·문장 분리를 먼저 점검)")
        print("=" * 74)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="nli_sensitivity_cases.csv")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--exclude-syn", action="store_true",
                    help="syn* 합성 문서를 제외하고 집계")
    ap.add_argument("--dump-kind", default="TERM_SWAP")
    ap.add_argument("--dump-n", type=int, default=10)
    a = ap.parse_args()

    rows = load_cases(a.cases, a.exclude_syn)
    sizes = doc_sizes(a.docs)

    print("=" * 74)
    print(f"NLI 민감도 사후 분석   tau={a.tau}"
          + ("   [합성문서 제외]" if a.exclude_syn else ""))
    print("=" * 74)

    an = Analyzer(rows, sizes, a.tau)
    an.overview()
    an.paired_rank()
    an.auc_table()
    an.tau_table()
    an.real_vs_syn()
    an.chunk_confound()
    an.dump_missed(a.dump_kind, a.dump_n)
    an.verdict()


if __name__ == "__main__":
    main()
