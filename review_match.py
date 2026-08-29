#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_match.py — 맹검 판정과 팽창률 기준 대조
================================================
질문:
    팽창률로 '내용창작'을 가려낼 수 있는가.

방법:
    사람 판정(정상/의심, 유형)과 answer 파일(group, ratio, 근거율, 수치환각)을
    번호로 맞춰 정밀도·재현율을 낸다.

주의:
    표본이 24건이고 이상치 쪽으로 치우쳐 뽑았으므로,
    여기서 나오는 정밀도·재현율은 전체 모집단의 값이 아니다.
    '이상치로 뽑은 것들이 실제로 창작인가'에 대한 답으로만 쓴다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python review_match.py
    python review_match.py --sheet expansion_review_final.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

CREATE = "내용창작"       # 오류로 볼 유형


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p and r and not math.isnan(p) and not math.isnan(r) else float("nan")
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="expansion_review_final.tsv")
    ap.add_argument("--answer", default="expansion_review_answer.json")
    a = ap.parse_args()

    if not os.path.exists(a.sheet):
        sys.exit(f"[FATAL] {a.sheet} 없음")
    if not os.path.exists(a.answer):
        sys.exit(f"[FATAL] {a.answer} 없음")

    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f, delimiter="\t"))
    with open(a.answer, encoding="utf-8") as f:
        ans = {int(x["번호"]): x for x in json.load(f)}

    Jc = next(c for c in sheet[0] if c.startswith("판정"))
    Tc = next(c for c in sheet[0] if c.startswith("유형"))

    rows = []
    for r in sheet:
        try:
            no = int(r["번호"])
        except (KeyError, ValueError):
            continue
        if no not in ans:
            continue
        v = ans[no]
        rows.append({
            "no": no, "doc": v["doc"], "title": v["title"],
            "group": v["group"], "ratio": float(v["ratio"]),
            "src_hangul": v.get("src_hangul"),
            "ground": v.get("ground"),
            "halluc": v.get("halluc") or [],
            "judge": r[Jc].strip(), "type": r[Tc].strip(),
        })
    if not rows:
        sys.exit("[FATAL] 번호가 맞는 행이 없습니다")

    print("=" * 96)
    print(f"맹검 대조   판정 {len(rows)}건")
    print("=" * 96)
    print(f"  {'번호':<4}{'문서':<7}{'그룹':<6}{'길이비':>9}{'근거율':>8}"
          f"{'수치환각':>9}{'판정':<6}{'유형':<10}섹션")
    for r in sorted(rows, key=lambda x: -x["ratio"]):
        g = f"{r['ground']:.1f}" if r["ground"] is not None else "—"
        print(f"  {r['no']:<4}{r['doc']:<7}{r['group']:<6}{r['ratio']:>9.2f}"
              f"{g:>8}{len(r['halluc']):>9}  {r['judge']:<6}{r['type']:<10}"
              f"{r['title'][:26]}")

    # --- 그룹별 ---
    print("\n" + "=" * 96)
    print("1. 이상치 그룹과 대조 그룹의 판정 차이")
    print("=" * 96)
    print(f"  {'그룹':<8}{'n':>4}{'의심':>7}{'내용창작':>10}{'서식재구성':>11}"
          f"{'부연설명':>10}{'정상':>7}")
    for grp in ("이상치", "대조"):
        sel = [r for r in rows if r["group"] == grp]
        if not sel:
            continue
        c = Counter(r["type"] for r in sel)
        n_sus = sum(1 for r in sel if r["judge"] == "의심")
        print(f"  {grp:<8}{len(sel):>4}{n_sus:>7}{c.get(CREATE, 0):>10}"
              f"{c.get('서식재구성', 0):>11}{c.get('부연설명', 0):>10}"
              f"{sum(1 for r in sel if r['judge'] == '정상'):>7}")

    # --- 팽창률 기준 평가 ---
    print("\n" + "=" * 96)
    print("2. 팽창률 기준으로 '내용창작'을 가려낼 수 있는가")
    print("=" * 96)
    print(f"  {'기준':<14}{'TP':>4}{'FP':>4}{'FN':>4}{'정밀도':>9}{'재현율':>9}{'F1':>8}")
    for thr in (1.5, 2.0, 3.0, 5.0, 10.0):
        tp = sum(1 for r in rows if r["ratio"] >= thr and r["type"] == CREATE)
        fp = sum(1 for r in rows if r["ratio"] >= thr and r["type"] != CREATE)
        fn = sum(1 for r in rows if r["ratio"] < thr and r["type"] == CREATE)
        p, rc, f1 = prf(tp, fp, fn)
        print(f"  길이비 ≥ {thr:<6.1f}{tp:>4}{fp:>4}{fn:>4}"
              f"{p:>9.1%}{rc:>9.1%}{f1:>8.2f}")

    # 근거율 기준
    gs = [r for r in rows if r["ground"] is not None]
    if len(gs) >= 8:
        print()
        for thr in (40, 55, 70, 80):
            tp = sum(1 for r in gs if r["ground"] < thr and r["type"] == CREATE)
            fp = sum(1 for r in gs if r["ground"] < thr and r["type"] != CREATE)
            fn = sum(1 for r in gs if r["ground"] >= thr and r["type"] == CREATE)
            p, rc, f1 = prf(tp, fp, fn)
            print(f"  근거율 < {thr:<6}{tp:>4}{fp:>4}{fn:>4}"
                  f"{p:>9.1%}{rc:>9.1%}{f1:>8.2f}")

        print()
        for rt in (2.0, 3.0):
            for gt in (55, 70):
                tp = sum(1 for r in gs if r["ratio"] >= rt and r["ground"] < gt
                         and r["type"] == CREATE)
                fp = sum(1 for r in gs if r["ratio"] >= rt and r["ground"] < gt
                         and r["type"] != CREATE)
                fn = sum(1 for r in gs if not (r["ratio"] >= rt and r["ground"] < gt)
                         and r["type"] == CREATE)
                p, rc, f1 = prf(tp, fp, fn)
                print(f"  ≥{rt:.0f}배 & 근거<{gt:<3}{tp:>4}{fp:>4}{fn:>4}"
                      f"{p:>9.1%}{rc:>9.1%}{f1:>8.2f}")

    # --- 지표별 판별력 ---
    print("\n" + "=" * 96)
    print("3. 어떤 지표가 '내용창작'을 잘 가리는가")
    print("=" * 96)
    lab = [1.0 if r["type"] == CREATE else 0.0 for r in rows]
    print(f"  Spearman(길이비, 창작)   = {spearman([r['ratio'] for r in rows], lab):+.3f}")
    if len(gs) >= 5:
        print(f"  Spearman(근거율, 창작)   = "
              f"{spearman([r['ground'] for r in gs], [1.0 if r['type'] == CREATE else 0.0 for r in gs]):+.3f}")
    print(f"  Spearman(수치환각, 창작) = "
          f"{spearman([len(r['halluc']) for r in rows], lab):+.3f}")

    print("\n  유형별 지표 평균")
    print(f"  {'유형':<10}{'n':>4}{'길이비':>10}{'근거율':>10}{'수치환각':>10}")
    for t in sorted({r["type"] for r in rows}):
        sel = [r for r in rows if r["type"] == t]
        g = [r["ground"] for r in sel if r["ground"] is not None]
        print(f"  {t:<10}{len(sel):>4}"
              f"{statistics.fmean(r['ratio'] for r in sel):>10.2f}"
              f"{(statistics.fmean(g) if g else float('nan')):>10.1f}"
              f"{statistics.fmean(len(r['halluc']) for r in sel):>10.2f}")

    # --- 판정 ---
    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    ctl = [r for r in rows if r["group"] == "대조"]
    ctl_bad = [r for r in ctl if r["type"] == CREATE]
    if ctl and len(ctl_bad) / len(ctl) >= 0.25:
        print(f"  대조군 {len(ctl)}건 중 내용창작 {len(ctl_bad)}건.")
        print("  → 정상 길이비 구간에서도 창작이 일어납니다.")
        print("     팽창률은 창작 탐지의 충분조건이 못 됩니다.")
    elif ctl:
        print(f"  대조군 {len(ctl)}건 중 내용창작 {len(ctl_bad)}건 — "
              "정상 구간은 비교적 안전합니다.")

    cands = []
    for thr in (1.5, 2.0, 3.0, 5.0, 10.0):
        tp = sum(1 for r in rows if r["ratio"] >= thr and r["type"] == CREATE)
        fp = sum(1 for r in rows if r["ratio"] >= thr and r["type"] != CREATE)
        fn = sum(1 for r in rows if r["ratio"] < thr and r["type"] == CREATE)
        p, rc, f1 = prf(tp, fp, fn)
        cands.append((thr, f1 if not math.isnan(f1) else -1.0))
    best_thr, best_f1 = max(cands, key=lambda x: x[1])
    print(f"\n  팽창률 단독 최고 F1 = {best_f1:.2f} (길이비 ≥ {best_thr})")
    if best_f1 < 0.6:
        print("  → 팽창률 단독으로는 부족합니다. 조합 기준이나 다른 지표가 필요합니다.")
    else:
        print("  → 팽창률 단독으로도 쓸 만합니다.")

    with open("review_match_report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] review_match_report.json")


if __name__ == "__main__":
    main()
