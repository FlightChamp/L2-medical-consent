#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
variant_review.py — variant_form occurrence 수동 검토
=======================================================
목적:
    terminology integrity 기준을 확정한다.
        variant_form 포함  (199+10)/250 = 83.6%
        variant_form 제외   199   /250 = 79.6%
    차이는 이 10건의 의미 동등성 판정 하나뿐이다.

variant_form 의 정의:
    원문 용어가 HARI 출력에 그대로는 없지만, 포함관계에 있는 다른 형태가 있다.
        원문 심근경색증  →  출력 심근경색     (접미 탈락)
        원문 절개        →  출력 피부절개     (접두 부착)

    이것이 같은 의미인지는 규칙으로 판정할 수 없다. 사람이 확정한다.

이 스크립트가 하는 일:
    1. term_status_review.csv 에서 auto_status == variant_form 인 행을 모은다
    2. 각 occurrence 에 대해 다음을 나란히 보여준다
         · 원문 용어
         · HARI 출력에서 발견된 변이형
         · 원문 문맥 (해당 용어 주변)
         · 출력 문맥 (변이형 주변)
    3. 자동 사전 판정을 제안한다 (접미 탈락 / 접두 부착 / 판단 필요)
    4. variant_review.csv 에 equivalent 열을 만들어 확정하게 한다
    5. --apply 로 확정 결과를 읽어 integrity 를 재계산한다

사용법:
    # 1단계 — 검토 자료 생성
    cd ~/이윤우 && source .venv/bin/activate
    python variant_review.py

    # 2단계 — variant_review.csv 의 equivalent 열을 채운 뒤
    python variant_review.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections
except ImportError as e:
    sys.exit("[FATAL] prompt_ladder.py 필요: " + str(e))

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
HARI_P0 = os.path.join(HOME, "outputs_prompt", "{doc}__P0.json")
STATUS_CSV = os.path.join(HOME, "term_status_review.csv")
AUDIT = os.path.join(HOME, "term_audit.json")
OUT_CSV = os.path.join(HOME, "variant_review.csv")


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_A(doc: str) -> Optional[str]:
    p = HARI_P0.format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("out")


def ctx(text: str, term: str, span: int = 55) -> str:
    i = text.find(term)
    if i < 0:
        return "(찾지 못함)"
    s = max(0, i - span)
    e = min(len(text), i + len(term) + span)
    out = text[s:e].replace("\n", " ")
    return ("..." if s > 0 else "") + re.sub(r"\s+", " ", out) + ("..." if e < len(text) else "")


def classify(src_term: str, variant: str) -> Tuple[str, str]:
    """(유형, 사전 제안). 제안은 참고용이며 사람이 확정한다."""
    if not variant:
        return "변이형 없음", ""
    if src_term in variant:
        extra = variant.replace(src_term, "", 1)
        return ("접두 부착" if variant.startswith(extra) else "접미 부착",
                "검토 필요 — 추가된 '" + extra + "' 가 의미를 바꾸는지 확인")
    if variant in src_term:
        dropped = src_term.replace(variant, "", 1)
        hint = ("동등 가능성 높음 — 탈락한 '" + dropped + "' 가 접미사"
                if dropped in ("증", "술", "염", "성", "적")
                else "검토 필요 — 탈락한 '" + dropped + "' 가 의미를 바꾸는지 확인")
        return "접미 탈락", hint
    return "기타", "검토 필요"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-csv", default=STATUS_CSV)
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--apply", action="store_true",
                    help="variant_review.csv 의 equivalent 열을 읽어 integrity 재계산")
    a = ap.parse_args()

    if not os.path.exists(a.status_csv):
        sys.exit("[FATAL] " + a.status_csv + " 없음. term_audit.py 를 먼저 실행하세요.")

    with open(a.status_csv, encoding="utf-8-sig", newline="") as f:
        allrows = list(csv.DictReader(f))
    vrows = [r for r in allrows if r.get("auto_status") == "variant_form"]

    # ── --apply: 확정 결과 반영 ────────────────────────────────────────
    if a.apply:
        if not os.path.exists(a.out):
            sys.exit("[FATAL] " + a.out + " 없음. --apply 없이 먼저 실행하세요.")
        with open(a.out, encoding="utf-8-sig", newline="") as f:
            done = list(csv.DictReader(f))
        yes = [r for r in done if str(r.get("equivalent", "")).strip().lower()
               in ("1", "true", "y", "yes", "예", "o")]
        no = [r for r in done if str(r.get("equivalent", "")).strip().lower()
              in ("0", "false", "n", "no", "아니오", "x")]
        blank = [r for r in done if r not in yes and r not in no]

        tal = {}
        if os.path.exists(a.audit):
            with open(a.audit, encoding="utf-8") as f:
                tal = json.load(f).get("status_tally") or {}
        tot = sum(tal.values()) or 250
        keep = (tal.get("exact_retained", 167)
                + tal.get("explained_in_source", 27)
                + tal.get("explained_by_model", 5))
        nvar = tal.get("variant_form", 10)

        print("=" * 92)
        print("variant_form 검토 결과 반영")
        print("=" * 92)
        print("  검토 대상            " + str(len(done)))
        print("  equivalent=true      " + str(len(yes)))
        print("  equivalent=false     " + str(len(no)))
        print("  미기입               " + str(len(blank)))
        if blank:
            print("\n  [경고] 미기입 " + str(len(blank)) + "건이 있습니다. "
                  "미기입은 보수적으로 false 로 처리합니다.")
            for r in blank[:5]:
                print("     " + r.get("doc", "") + " / " + r.get("term", ""))

        integ = 100 * (keep + len(yes)) / tot
        print("\n  " + "기준".ljust(34) + "값".rjust(8) + "   산식")
        print("  " + "variant 전부 제외 (보수)".ljust(34)
              + ("%7.1f%%" % (100 * keep / tot))
              + "   " + str(keep) + "/" + str(tot))
        print("  " + "검토 확정 (equivalent=true 만)".ljust(34)
              + ("%7.1f%%" % integ)
              + "   (" + str(keep) + "+" + str(len(yes)) + ")/" + str(tot)
              + "   ← 주 기준")
        print("  " + "variant 전부 포함 (상한)".ljust(34)
              + ("%7.1f%%" % (100 * (keep + nvar) / tot))
              + "   (" + str(keep) + "+" + str(nvar) + ")/" + str(tot))

        res = {
            "reviewed": len(done), "equivalent_true": len(yes),
            "equivalent_false": len(no), "blank_treated_as_false": len(blank),
            "denominator": tot, "keep_base": keep, "n_variant_total": nvar,
            "integrity_conservative": round(100 * keep / tot, 1),
            "integrity_confirmed": round(integ, 1),
            "integrity_upper": round(100 * (keep + nvar) / tot, 1),
            "decision": ("보고서 주 기준은 integrity_confirmed 를 쓴다. "
                         "미기입은 false 로 처리했다."),
            "rows": done,
        }
        with open("variant_review_result.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("\n[SAVE] variant_review_result.json")
        print("\n  최종 보고서에는 " + ("%.1f%%" % integ) + " 를 주 기준으로 쓰십시오.")
        print("=" * 92)
        return

    # ── 검토 자료 생성 ────────────────────────────────────────────────
    print("=" * 92)
    print("variant_form occurrence 수동 검토 자료")
    print("=" * 92)
    print("  대상 " + str(len(vrows)) + "건")
    print("  각 건의 의미 동등성을 확인해 " + a.out + " 의 equivalent 열에")
    print("  1(동등) 또는 0(비동등)을 기입하십시오.")
    print("  판정된 것만 terminology integrity 성공에 포함합니다.")

    Ks: Dict[str, str] = {}
    As: Dict[str, str] = {}
    out_rows = []
    for i, r in enumerate(vrows, 1):
        doc, term = r.get("doc", ""), r.get("term", "")
        var = r.get("variant_in_output", "")
        if doc not in Ks:
            Ks[doc] = load_K(doc, a.docdir) or ""
            As[doc] = load_A(doc) or ""
        kind, hint = classify(term, var)
        k_ctx = ctx(Ks[doc], term)
        a_ctx = ctx(As[doc], var) if var else "(변이형 없음)"

        print("\n" + "-" * 92)
        print("  [" + str(i) + "/" + str(len(vrows)) + "]  "
              + doc + "   원문 용어: " + term + "   →   출력 형태: " + var)
        print("        유형: " + kind + "   |   " + hint)
        print("        원문: " + k_ctx)
        print("        출력: " + a_ctx)

        out_rows.append({
            "no": i, "doc": doc, "src_term": term, "variant_in_output": var,
            "variant_kind": kind, "auto_hint": hint,
            "source_context": k_ctx, "output_context": a_ctx,
            "equivalent": "", "reviewer_note": "",
        })

    if not out_rows:
        print("\n  variant_form 행이 없습니다. term_status_review.csv 를 확인하세요.")
        return

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for x in out_rows:
            w.writerow(x)

    print("\n" + "=" * 92)
    print("유형별 집계")
    print("=" * 92)
    kinds: Dict[str, int] = {}
    for x in out_rows:
        kinds[x["variant_kind"]] = kinds.get(x["variant_kind"], 0) + 1
    for k, v in sorted(kinds.items(), key=lambda z: -z[1]):
        print("  " + k.ljust(16) + str(v) + "건")

    print("\n[SAVE] " + a.out)
    print("""
결과 보는 법
   · '접미 탈락' 중 탈락한 글자가 증/술/염 같은 접미사면 동등 가능성이 높습니다.
     예: 심근경색증 → 심근경색 (같은 질환을 가리킴)
   · '접두 부착' 은 의미가 좁아질 수 있어 주의가 필요합니다.
     예: 절개 → 피부절개 (일반 절개가 피부 절개로 한정됨)
   · 자동 제안(auto_hint)은 참고용입니다. 문맥을 보고 확정하십시오.

다음
   variant_review.csv 의 equivalent 열을 채운 뒤
   python variant_review.py --apply
""")
    print("=" * 92)


if __name__ == "__main__":
    main()
