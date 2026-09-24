#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v11.py — Glossary v1.1 실행 및 v1.05 대비 비교
====================================================
독립 변수는 glossary coverage 하나뿐이다.
프롬프트·모델·생성 설정은 바꾸지 않는다.

    A        HARI P0 baseline           (outputs_prompt/{doc}__P0.json)
    E_v105   A + glossary 7개           (outputs_glossary_v105/E/)
    E_v11    A + glossary 11개          (outputs_glossary_v11/E/)

기존 결과 파일은 건드리지 않는다. v1.1 은 새 폴더·새 파일에 쓴다.

실행:
    cd ~/이윤우 && source .venv/bin/activate
    python run_v11.py

    # 비교표만 다시 보기 (실행 없이)
    python run_v11.py --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HOME = os.path.expanduser("~/이윤우")

GLOSSARY_V11 = "glossary_v1_1.json"
OUTDIR_V11 = "outputs_glossary_v11"
EVAL_V11 = "eval_A_vs_E_v11.json"
CSV_V11 = "eval_A_vs_E_v11_per_doc.csv"

EVAL_V105 = "eval_A_vs_E_v105.json"
VARIANT_REVIEW = "variant_review.csv"


def sh(cmd: list) -> int:
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    return subprocess.call(cmd)


def fmt(v, d=2, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:.{d}f}{suffix}"


def delta(a, b, d=2, suffix=""):
    """b - a. 둘 중 하나라도 없으면 —."""
    if a is None or b is None:
        return "—"
    x = b - a
    return f"{x:+.{d}f}{suffix}"


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report(a_path: str, b_path: str) -> None:
    A, B = load(a_path), load(b_path)
    if A is None:
        sys.exit(f"[FATAL] {a_path} 없음. v1.05 평가를 먼저 실행하세요.")
    if B is None:
        sys.exit(f"[FATAL] {b_path} 없음. v1.1 평가가 실패했습니다.")

    sa, sb = A["summary"], B["summary"]
    ta = A.get("terminology_integrity") or {}
    tb = B.get("terminology_integrity") or {}

    W = 40
    def row(label, va, vb, dv=""):
        print("  " + str(label).ljust(W)
              + str(va).rjust(16) + str(vb).rjust(16) + str(dv).rjust(14))

    print("\n" + "=" * 100)
    print("v1.05 vs v1.1 — glossary coverage 확대만을 독립 변수로")
    print("=" * 100)
    print("  " + "지표".ljust(W) + "v1.05 (7개)".rjust(16)
          + "v1.1 (11개)".rjust(16) + "변화".rjust(14))
    print("  " + "-" * (W + 46))

    print("\n  [Glossary 적용]")
    row("verified glossary entries",
        len(A.get("glossary_terms", [])), len(B.get("glossary_terms", [])),
        f"{len(B.get('glossary_terms', [])) - len(A.get('glossary_terms', [])):+d}")
    row("target occurrence", sa.get("target_occurrences"), sb.get("target_occurrences"),
        f"{(sb.get('target_occurrences') or 0) - (sa.get('target_occurrences') or 0):+d}")
    row("eligible retained occurrence",
        sa.get("eligible_occurrences"), sb.get("eligible_occurrences"),
        f"{(sb.get('eligible_occurrences') or 0) - (sa.get('eligible_occurrences') or 0):+d}")
    row("실제 insertion", sa.get("insertions"), sb.get("insertions"),
        f"{(sb.get('insertions') or 0) - (sa.get('insertions') or 0):+d}")
    row("target-level coverage",
        fmt(sa.get("coverage_pct_target_level"), 1, "%"),
        fmt(sb.get("coverage_pct_target_level"), 1, "%"),
        delta(sa.get("coverage_pct_target_level"),
              sb.get("coverage_pct_target_level"), 1, "%p"))
    row("occurrence-level (참고)",
        fmt(sa.get("coverage_pct_occurrence_level"), 1, "%"),
        fmt(sb.get("coverage_pct_occurrence_level"), 1, "%"),
        delta(sa.get("coverage_pct_occurrence_level"),
              sb.get("coverage_pct_occurrence_level"), 1, "%p"))
    row("incorrect insertion",
        sa.get("incorrect_insertions"), sb.get("incorrect_insertions"),
        f"{(sb.get('incorrect_insertions') or 0) - (sa.get('incorrect_insertions') or 0):+d}")

    print("\n  [skip 사유별]")
    row("첫 등장 아님", sa.get("skip_not_first"), sb.get("skip_not_first"),
        f"{(sb.get('skip_not_first') or 0) - (sa.get('skip_not_first') or 0):+d}")
    row("원문 괄호 (source-parenthetical)",
        sa.get("skip_source_parenthetical"), sb.get("skip_source_parenthetical"),
        f"{(sb.get('skip_source_parenthetical') or 0) - (sa.get('skip_source_parenthetical') or 0):+d}")
    row("출력에만 있는 괄호",
        sa.get("skip_existing_parenthetical"), sb.get("skip_existing_parenthetical"),
        f"{(sb.get('skip_existing_parenthetical') or 0) - (sa.get('skip_existing_parenthetical') or 0):+d}")

    print("\n  [Safety / Fidelity]")
    row("Grounding A", fmt(sa.get("grounding_A")), fmt(sb.get("grounding_A")),
        delta(sa.get("grounding_A"), sb.get("grounding_A")))
    row("Grounding E (all)",
        fmt(sa.get("grounding_E_all")), fmt(sb.get("grounding_E_all")),
        delta(sa.get("grounding_E_all"), sb.get("grounding_E_all")))
    row("  └ Grounding cost (E − A)",
        fmt(sa.get("delta_grounding"), 2, "%p"),
        fmt(sb.get("delta_grounding"), 2, "%p"),
        delta(sa.get("delta_grounding"), sb.get("delta_grounding"), 2, "%p"))
    row("Terminology integrity [conservative]",
        fmt(ta.get("conservative"), 1, "%"), fmt(tb.get("conservative"), 1, "%"),
        delta(ta.get("conservative"), tb.get("conservative"), 1, "%p"))
    row("  [human_adjudicated] ← 주지표",
        fmt(ta.get("human_adjudicated"), 1, "%"),
        fmt(tb.get("human_adjudicated"), 1, "%"),
        delta(ta.get("human_adjudicated"), tb.get("human_adjudicated"), 1, "%p"))
    row("  [upper_bound]",
        fmt(ta.get("upper_bound_all_variants"), 1, "%"),
        fmt(tb.get("upper_bound_all_variants"), 1, "%"),
        delta(ta.get("upper_bound_all_variants"),
              tb.get("upper_bound_all_variants"), 1, "%p"))

    print("\n  [Simplicity / Cost]")
    row("length change", fmt(sa.get("length_increase_pct"), 2, "%"),
        fmt(sb.get("length_increase_pct"), 2, "%"),
        delta(sa.get("length_increase_pct"), sb.get("length_increase_pct"), 2, "%p"))
    row("평균 어절/문장 (A)", fmt(sa.get("wps_A")), fmt(sb.get("wps_A")),
        delta(sa.get("wps_A"), sb.get("wps_A")))
    row("평균 어절/문장 (E)", fmt(sa.get("wps_E")), fmt(sb.get("wps_E")),
        delta(sa.get("wps_E"), sb.get("wps_E")))
    row("  └ 순수 삽입 효과 (짝지은 문장)",
        fmt(sa.get("delta_wps_paired"), 3), fmt(sb.get("delta_wps_paired"), 3),
        delta(sa.get("delta_wps_paired"), sb.get("delta_wps_paired"), 3))

    # Ko2025 는 rows 에서 평균
    import statistics as st
    def mean(d, k):
        v = [r[k] for r in d.get("rows", []) if r.get(k) is not None]
        return st.fmean(v) if v else None
    for label, key in [("Ko2025 (A)", "ko2025_A"), ("Ko2025 (E)", "ko2025_E"),
                       ("중급이상 어휘비율 (A)", "mid_vocab_A"),
                       ("중급이상 어휘비율 (E)", "mid_vocab_E")]:
        row(label, fmt(mean(A, key)), fmt(mean(B, key)),
            delta(mean(A, key), mean(B, key)))

    # ── 판정 ──────────────────────────────────────────────────────────
    cov_ok = (sb.get("coverage_pct_target_level") or 0) >= 99.99
    bad_ok = (sb.get("incorrect_insertions") or 0) == 0
    integ_same = (tb.get("human_adjudicated") == ta.get("human_adjudicated"))
    n_ins_a = sa.get("insertions") or 0
    n_ins_b = sb.get("insertions") or 0

    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    print(f"  1. coverage 100% 유지          {'예' if cov_ok else '아니오'}"
          f"   ({sb.get('insertions')}/{sb.get('target_occurrences')})")
    print(f"  2. incorrect insertion 0 유지  {'예' if bad_ok else '아니오'}"
          f"   ({sb.get('incorrect_insertions')}건)")
    print(f"  3. terminology integrity 유지   {'예' if integ_same else '아니오'}"
          f"   ({fmt(tb.get('human_adjudicated'), 1, '%')})")
    print(f"  4. Grounding cost              {fmt(sb.get('delta_grounding'), 2, '%p')}"
          f"   (v1.05 는 {fmt(sa.get('delta_grounding'), 2, '%p')})")
    print(f"  5. 적용 범위 증가               {n_ins_a} → {n_ins_b}"
          f"   ({100*(n_ins_b-n_ins_a)/max(n_ins_a,1):+.1f}%)")

    print(f"""
  결론 문장 (그대로 인용 가능)

    검증 완료 용어 {len(B.get('glossary_terms', []))}개,
    target {sb.get('target_occurrences')}건에서
    coverage {fmt(sb.get('coverage_pct_target_level'), 1, '%')},
    incorrect insertion {sb.get('incorrect_insertions')}건이며,
    Grounding cost 는 {fmt(sb.get('delta_grounding'), 2, '%p')}였다.

  해석 시 주의
    · coverage 100% 는 '검증된 glossary target 내에서' 만을 뜻합니다.
    · 용어 수 증가를 성능 향상으로 해석하지 않습니다. 목적은 coverage 확대입니다.
    · Ko2025 가 소폭 떨어지는 것은 설명이 문장을 길게 만들기 때문이며 실패가 아닙니다.
    · 실제 L2 환자 이해도 향상을 검증한 것이 아닙니다.
    · 13문서 외부로 일반화하지 않습니다.""")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="실행 없이 기존 결과만 비교")
    ap.add_argument("--glossary", default=GLOSSARY_V11)
    ap.add_argument("--outdir", default=OUTDIR_V11)
    a = ap.parse_args()

    if not a.report_only:
        if not os.path.exists(a.glossary):
            sys.exit(f"[FATAL] {a.glossary} 가 현재 폴더에 없습니다.")
        g = load(a.glossary)
        ok = [e for e in g["entries"]
              if e.get("source_verified") and e.get("explanation_verified")]
        print("=" * 100)
        print(f"glossary {g['glossary_version']} — 삽입 가능 {len(ok)} / 전체 {len(g['entries'])}")
        print("=" * 100)
        for e in ok:
            tag = "신규" if e.get("version_added") == "v1.1" else "    "
            print(f"  {tag} {e['term']:<12}{e['easy_explanation']}")
        print(f"\n  제외 {len(g.get('rejected_v11', []))}건: "
              + ", ".join(x["term"] for x in g.get("rejected_v11", [])))

        if sh([sys.executable, "apply_glossary.py",
               "--glossary", a.glossary, "--outdir", a.outdir]) != 0:
            sys.exit("[FATAL] apply_glossary.py 실패")

        if sh([sys.executable, "eval_A_vs_E.py",
               "--edir", os.path.join(a.outdir, "E"),
               "--variant-review", VARIANT_REVIEW,
               "--out", EVAL_V11, "--csv", CSV_V11]) != 0:
            sys.exit("[FATAL] eval_A_vs_E.py 실패")

    report(EVAL_V105, EVAL_V11)


if __name__ == "__main__":
    main()
