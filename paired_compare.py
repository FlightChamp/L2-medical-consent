#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paired_compare.py — fail-closed 를 제외한 짝지은 비교
======================================================
문제:
    phase2_eval.py 의 평균은 fail-closed 문서를 포함한다.
    fail-closed 문서는 원문을 그대로 쓰므로 Copy=1.0, 어절=22.59 이고,
    평균을 Identity 쪽으로 끌어당긴다.

        B 조건 13문서 중 4건이 fail-closed
        D 조건 13문서 중 2건이 fail-closed
        → 표면 평균(B 어절 14.61)은 실제 변환 성능이 아니다.

이 스크립트가 하는 일:
    1. 두 조건이 **모두 정상 변환한 문서**만 골라 짝지어 비교한다
    2. 문서별 차이를 그대로 보여준다 (13문서이므로 개별 값이 중요하다)
    3. 문서 단위 부트스트랩으로 차이의 95% CI 를 낸다
    4. fail-closed 문서를 따로 표기한다 — 숨기지 않는다

비교 쌍:
    A vs B    P0 에서 Protected 효과
    C vs D    P4 에서 Protected 효과
    A vs C    Protected OFF 에서 P4 효과
    B vs D    Protected ON 에서 P4 효과

원칙:
    · 표본이 13문서이므로 평균 하나로 판단하지 않는다.
    · 문서 단위 부트스트랩을 쓴다 (문장을 독립 표본으로 취급하지 않는다).
    · CI 가 0 을 포함하면 "이 표본에서 차이를 단정할 수 없다" 고 쓴다.
    · Safety 와 Simplicity 를 분리한다. 단일 총점을 만들지 않는다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python paired_compare.py                      # phase2_metrics.json 사용
    python paired_compare.py --pairs A:B C:D
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

HOME = os.path.expanduser("~/이윤우")

# 비교할 지표 — (키, 그룹, 방향, 표시명)
METRICS = [
    ("ground", "safety", "higher", "Grounding"),
    ("cover", "safety", "higher", "Coverage"),
    ("term_keep", "safety", "higher", "용어 보존"),
    ("num_keep", "safety", "higher", "수치 보존"),
    ("n_halluc", "safety", "lower", "수치 환각"),
    ("ko2025_A", "simplicity", "higher", "L2 이독성 A"),
    ("ko2025_B", "simplicity", "higher", "L2 이독성 B"),
    ("mid_vocab_A", "simplicity", "lower", "중급이상어휘 A"),
    ("mid_vocab_B", "simplicity", "lower", "중급이상어휘 B"),
    ("words_per_sent", "simplicity", "lower", "어절/문장"),
    ("long_sent_ratio", "simplicity", "lower", "긴문장 비율"),
    ("term_explain_rate", "simplicity", "higher", "용어 설명율"),
    ("copy_similarity", "simplicity", "info", "Copy 유사도"),
]

ROUTE_OF = {"A": "A P0/OFF", "B": "B P0/ON",
            "C": "C P4/OFF", "D": "D P4/ON", "I": "Identity"}


def load(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {path} 없음. phase2_eval.py 를 먼저 실행하세요.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_rows(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["route"], r["doc"]): r for r in rows}


def get(rec: Optional[dict], grp: str, key: str) -> Optional[float]:
    if rec is None:
        return None
    v = rec.get(grp, {}).get(key)
    return None if v is None else float(v)


def boot_ci(diffs: List[float], n: int = 4000,
            seed: int = 20260917) -> Tuple[float, float]:
    """문서 단위 복원추출. 문서가 표본 단위이므로 문장을 섞지 않는다."""
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [rng.choice(diffs) for _ in diffs]
        means.append(st.fmean(s))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def compare(idx, docs, a_key: str, b_key: str, args) -> dict:
    ra, rb = ROUTE_OF[a_key], ROUTE_OF[b_key]

    # fail-closed 문서 파악
    fc_a = {d for d in docs
            if idx.get((ra, d), {}).get("fail_closed")}
    fc_b = {d for d in docs
            if idx.get((rb, d), {}).get("fail_closed")}
    excluded = sorted(fc_a | fc_b)
    common = [d for d in docs
              if (ra, d) in idx and (rb, d) in idx and d not in excluded]

    print("\n" + "=" * 100)
    print(f"{ra}  vs  {rb}")
    print("=" * 100)
    print(f"  전체 {len(docs)}문서 / 비교 대상 {len(common)}문서")
    if excluded:
        print(f"  제외 {len(excluded)}문서 (fail-closed): {', '.join(excluded)}")
        if fc_a:
            print(f"    {ra}: {', '.join(sorted(fc_a))}")
        if fc_b:
            print(f"    {rb}: {', '.join(sorted(fc_b))}")
    if not common:
        print("  공통 문서가 없어 비교할 수 없습니다.")
        return {"pair": f"{a_key}:{b_key}", "n": 0}

    out = {"pair": f"{a_key}:{b_key}", "n": len(common),
           "excluded": excluded, "metrics": {}}

    print(f"\n  {'지표':<16}{'방향':>5}{ra:>13}{rb:>13}{'차이':>10}"
          f"{'95% CI':>20}{'판정':>10}")
    for key, grp, direc, label in METRICS:
        pa = [get(idx.get((ra, d)), grp, key) for d in common]
        pb = [get(idx.get((rb, d)), grp, key) for d in common]
        pairs = [(x, y) for x, y in zip(pa, pb)
                 if x is not None and y is not None]
        if len(pairs) < 2:
            continue
        va = st.fmean(x for x, _ in pairs)
        vb = st.fmean(y for _, y in pairs)
        diffs = [y - x for x, y in pairs]
        d = st.fmean(diffs)
        lo, hi = boot_ci(diffs, args.boot)

        if direc == "info":
            verdict = "참고"
        elif lo > 0:
            verdict = "개선" if direc == "higher" else "악화"
        elif hi < 0:
            verdict = "악화" if direc == "higher" else "개선"
        else:
            verdict = "판단보류"

        arrow = {"higher": "↑", "lower": "↓", "info": "–"}[direc]
        print(f"  {label:<16}{arrow:>5}{va:>13.2f}{vb:>13.2f}{d:>+10.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{verdict:>10}")
        out["metrics"][key] = {
            "label": label, "direction": direc, "n": len(pairs),
            "mean_a": round(va, 3), "mean_b": round(vb, 3),
            "diff": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "verdict": verdict,
        }

    # 문서별 상세 — 13문서이므로 개별 값을 보여준다
    if args.detail:
        show = [("ko2025_A", "simplicity"), ("words_per_sent", "simplicity"),
                ("ground", "safety"), ("num_keep", "safety"),
                ("copy_similarity", "simplicity")]
        print(f"\n  문서별 ({rb} − {ra})")
        print(f"  {'문서':<8}" + "".join(
            f"{k:>17}" for k, _ in show))
        for d in common:
            line = f"  {d:<8}"
            for k, g in show:
                x, y = get(idx.get((ra, d)), g, k), get(idx.get((rb, d)), g, k)
                line += (f"{'—':>17}" if x is None or y is None
                         else f"{y-x:>+17.2f}")
            print(line)

    print(f"""
  읽는 법
    · 차이는 {rb} − {ra} 입니다. 방향(↑/↓)에 맞게 판정했습니다.
    · CI 는 문서 {len(common)}개를 복원추출한 것입니다({args.boot}회).
      13문서 규모이므로 CI 는 넓습니다. 0 을 포함하면 '판단보류' 입니다.
    · fail-closed 문서는 제외했습니다. 제외 사실 자체가 결과의 일부입니다.""")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(HOME, "phase2_metrics.json"))
    ap.add_argument("--pairs", nargs="+",
                    default=["A:B", "C:D", "A:C", "B:D"])
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--detail", action="store_true", default=True)
    ap.add_argument("--no-detail", dest="detail", action="store_false")
    ap.add_argument("--out", default="paired_compare.json")
    a = ap.parse_args()

    data = load(a.metrics)
    rows = data["rows"]
    idx = index_rows(rows)
    docs = sorted({r["doc"] for r in rows if r["doc"]})
    routes = set(r["route"] for r in rows)

    print("=" * 100)
    print("짝지은 비교 — fail-closed 제외")
    print("=" * 100)
    print(f"  입력: {a.metrics}")
    print(f"  문서 {len(docs)}건 / Route {len(routes)}종")
    fc = {}
    for r in rows:
        if r.get("fail_closed"):
            fc.setdefault(r["route"], []).append(r["doc"])
    if fc:
        print("\n  fail-closed 현황")
        for k, v in sorted(fc.items()):
            print(f"    {k:<12}{len(v)}건: {', '.join(sorted(v))}")
    else:
        print("\n  fail-closed 문서 없음")

    results = []
    for pr in a.pairs:
        try:
            x, y = pr.split(":")
        except ValueError:
            print(f"  [건너뜀] 형식 오류: {pr}")
            continue
        if ROUTE_OF.get(x) not in routes or ROUTE_OF.get(y) not in routes:
            print(f"  [건너뜀] {pr} — 결과에 없는 조건")
            continue
        results.append(compare(idx, docs, x, y, a))

    # 요약
    print("\n" + "=" * 100)
    print("요약 — 각 쌍에서 유의한 변화만")
    print("=" * 100)
    for r in results:
        if not r.get("metrics"):
            continue
        ra, rb = r["pair"].split(":")
        sig = [(m["label"], m["diff"], m["verdict"])
               for m in r["metrics"].values()
               if m["verdict"] in ("개선", "악화")]
        print(f"\n  {ROUTE_OF[ra]} → {ROUTE_OF[rb]}  (n={r['n']})")
        if not sig:
            print("    유의한 변화 없음 (모든 지표 CI 가 0 을 포함)")
        for label, d, v in sig:
            print(f"    {v}  {label:<18}{d:+.2f}")
    print("=" * 100)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"source": a.metrics, "boot": a.boot,
                   "fail_closed": fc, "pairs": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
