#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
equalize_spec.py — 오탐률을 맞춘 상태에서 집계 방식을 재비교 (GPU 불필요)
=========================================================================
문제:
    nli_sensitivity2.py 는 네 가지 집계를 같은 tau=0.5 로 비교했다.
    그런데 집계마다 후보 청크 수가 달라 점수 분포가 통째로 이동한다.
    (정상 통과율 GLOBAL 96.6% vs TOPK 92.2%)
    즉 TOPK 의 탐지율 상승분 일부는 '오탐을 더 낸 대가'다.

해결:
    각 집계마다 '정상 통과율이 목표치와 같아지는' 임계값을 따로 찾은 뒤
    그 지점에서 탐지율을 비교한다. 같은 오탐 예산에서의 공정 비교.

사용법:
    python equalize_spec.py
    python equalize_spec.py --target-spec 0.95
    python equalize_spec.py --cases nli_sens2_cases.csv --baseline GLOBAL
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Sequence, Tuple

MODES = ("GLOBAL", "FILTERED", "TOPK", "LOCAL")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def load(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {path} 없음")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            for m in MODES:
                r[m] = float(r[m])
            r["sent_id"] = int(r["sent_id"])
            rows.append(r)
    return rows


def tau_for_spec(orig_scores: Sequence[float], target: float) -> float:
    """정상 문장의 target 비율이 통과(>= tau)하도록 하는 임계값.

    분위수로 직접 구한다. target=0.966 이면 하위 3.4% 지점이 임계값이 된다.
    """
    xs = sorted(orig_scores)
    k = int(round((1.0 - target) * len(xs)))
    k = max(0, min(k, len(xs) - 1))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="nli_sens2_cases.csv")
    ap.add_argument("--baseline", default="GLOBAL",
                    help="이 모드의 실제 정상 통과율을 목표치로 삼는다")
    ap.add_argument("--target-spec", type=float, default=None,
                    help="직접 지정 시 baseline 대신 이 값을 사용 (예: 0.95)")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="baseline 통과율을 계산할 때 쓰는 원래 임계값")
    a = ap.parse_args()

    rows = load(a.cases)
    orig = [r for r in rows if r["kind"] == "ORIGINAL"]
    kinds = sorted({r["kind"] for r in rows if r["kind"] != "ORIGINAL"})
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in kinds}

    if a.target_spec is not None:
        target = a.target_spec
        src = "직접 지정"
    else:
        target = sum(1 for r in orig if r[a.baseline] >= a.tau) / len(orig)
        src = f"{a.baseline} @ tau={a.tau} 의 실측값"

    taus = {m: tau_for_spec([r[m] for r in orig], target) for m in MODES}

    print("=" * 78)
    print(f"오탐률 통일 재비교   목표 정상 통과율 = {target:.1%}  ({src})")
    print("=" * 78)
    print(f"  원문 표본 {len(orig)}건 / 전체 {len(rows)}건\n")

    print("  집계별 보정 임계값과 실제 통과율")
    print(f"    {'모드':<12}{'tau':>9}{'실제 통과율':>14}")
    for m in MODES:
        spec = sum(1 for r in orig if r[m] >= taus[m]) / len(orig)
        print(f"    {m:<12}{taus[m]:>9.3f}{spec:>13.1%}")

    print("\n" + "=" * 78)
    print("같은 오탐 예산에서의 탐지율")
    print("=" * 78)
    print("    {:<16}{:>5}".format("유형", "n") + "".join(f"{m:>11}" for m in MODES)
          + "   TOPK 95%CI")
    deltas = []
    for k in kinds:
        sel = by_kind[k]
        row = f"    {k:<16}{len(sel):>5}"
        vals = {}
        for m in MODES:
            d = sum(1 for r in sel if r[m] < taus[m])
            vals[m] = d / len(sel)
            row += f"{vals[m]:>10.1%} "
        lo, hi = wilson(sum(1 for r in sel if r["TOPK"] < taus["TOPK"]), len(sel))
        deltas.append((k, vals["TOPK"] - vals["GLOBAL"], len(sel)))
        print(row + f"  [{lo:.0%},{hi:.0%}]")

    print("\n" + "=" * 78)
    print("판정 — 오탐률을 맞춘 뒤에도 TOPK 가 이득인가")
    print("=" * 78)
    for k, d, n in sorted(deltas, key=lambda x: -x[1]):
        mark = "" if abs(d) > 0.05 else "   (차이 미미)"
        print(f"    {k:<16}{d:+8.1%}   n={n}{mark}")
    avg = sum(d for _, d, _ in deltas) / len(deltas)
    print(f"\n    평균 {avg:+.1%}")
    if avg > 0.05:
        print("    → 오탐을 더 내서 얻은 이득이 아닙니다. 집계 방식 개선이 실제 효과입니다.")
    elif avg > 0.0:
        print("    → 이득이 남지만 폭이 줄었습니다. 상승분 상당 부분은 임계값 이동 효과였습니다.")
    else:
        print("    → 오탐률을 맞추면 이득이 사라집니다. TOPK 채택 근거가 약합니다.")

    print("\n" + "=" * 78)
    print("취약 유형 진단 — TOPK 대비 LOCAL(오라클 상한) 여유")
    print("=" * 78)
    shown = False
    for k in kinds:
        sel = by_kind[k]
        t = sum(1 for r in sel if r["TOPK"] < taus["TOPK"]) / len(sel)
        l = sum(1 for r in sel if r["LOCAL"] < taus["LOCAL"]) / len(sel)
        if t >= 0.85:
            continue
        shown = True
        gap = l - t
        if gap > 0.10:
            dx = "검색으로 더 개선 가능 (후보 k 확대·정렬 개선)"
        else:
            dx = "검색을 완벽히 해도 한계 — NLI 자체가 못 잡는 유형"
        print(f"    {k:<16} TOPK {t:>6.1%} / LOCAL {l:>6.1%}  여유 {gap:+.1%}  → {dx}")
    if not shown:
        print("    탐지율 85% 미만인 취약 유형이 없습니다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
