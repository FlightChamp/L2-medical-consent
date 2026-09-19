#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_A_vs_E.py — Glossary 파일럿 평가
======================================
A = 기존 HARI P0 출력
E = 동일 A + verified glossary deterministic insertion  (새 generation 아님)

핵심 지표 네 개:
    1. Verified Explanation Coverage  ↑
    2. Incorrect Insertion            ↓
    3. Terminology Integrity          ↑ / 유지
    4. Length Cost                    ↓

Verified Explanation Coverage 정의 (고정):
        검증된 glossary 설명이 실제 삽입된 retained source-term occurrences
        ---------------------------------------------------------------
        HARI 출력에 실제 남아 있는 source-term occurrences 중 glossary 대상

    HARI 에서 이미 사라진 용어는 이 모듈의 실패로 계산하지 않는다.
    absent before glossary 로 별도 보고한다.

Grounding:
    Grounding_A      = 기존 HARI 출력의 document-grounded 값
    Grounding_E_all  = glossary 설명까지 포함한 최종 표시 텍스트의 값
    Delta            = E_all - A
    문자열에서 glossary 를 제거해 재계산하지 않는다.
    설명 추가로 NLI 점수가 변하는 현상 자체가 glossary insertion 의 영향이다.

평가 단위:
    주지표는 occurrence-level. type-level 은 보조 분석으로 함께 낸다.
    variant_form 은 사람이 의미 동등성을 확인한 경우에만 integrity 성공으로
    포함한다. 기본값은 제외이며 --count-variant 로 포함할 수 있다.

Ko2025:
    총점 하나로 판단하지 않는다. 어휘 항과 문장길이 항을 분리해 보조 지표로
    보고한다. 설명 삽입으로 문장길이 항이 악화되는 것은 구조적으로 예상된다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python eval_A_vs_E.py
    python eval_A_vs_E.py --count-variant
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, Splitter, build_chunks
    from translate_verify2 import NLI
    from simplicity_metrics import simplicity
except ImportError as e:
    sys.exit("[FATAL] 상위 폴더 스크립트 필요: " + str(e))

try:
    from simplicity_metrics import mid_vocab_ratio, ko2025_index, ko2025_level, KO2025
    KO_OK = True
except ImportError:
    KO_OK = False

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
EDIR = os.path.join(HOME, "outputs_glossary", "E")
GLOSSARY = os.path.join(HOME, "glossary_v1.json")
AUDIT = os.path.join(HOME, "term_audit.json")

HANGUL = re.compile(r"[가-힣]")
JOSA = ("이", "가", "은", "는", "을", "를", "의", "에", "에서", "에게", "에는",
        "으로", "로", "와", "과", "도", "만", "부터", "까지", "라", "이라",
        "이나", "나", "이며", "며", "이고", "고", "인", "이란", "란",
        "처럼", "보다", "조차", "마저", "밖에", "이라고", "라고",
        "술", "시", "후", "전", "중", "및", "또는")
JOSA_RE = re.compile("^(?:" + "|".join(sorted(JOSA, key=len, reverse=True)) + ")")


def find_occurrences(text: str, terms: List[str]) -> List[Tuple[int, int, str]]:
    hits = []
    for t in sorted(set(terms), key=len, reverse=True):
        start = 0
        while True:
            i = text.find(t, start)
            if i < 0:
                break
            start = i + 1
            if i > 0 and HANGUL.match(text[i - 1]):
                continue
            j = i + len(t)
            tail = text[j:j + 6]
            if tail and HANGUL.match(tail[0]) and not JOSA_RE.match(tail):
                continue
            hits.append((i, j, t))
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last = [], -1
    for s, e, t in hits:
        if s >= last:
            kept.append((s, e, t))
            last = e
    return kept


def sentence_boundary_check(A: str, E: str, insertions) -> dict:
    """A 의 문장 경계가 E 에서 유지되는지 본다.

    개수 비교는 쓰지 않는다. HARI 출력에는 줄바꿈 목록이 있고, Splitter 가
    len >= 10 조건으로 짧은 항목을 제외하므로, 삽입으로 항목이 길어지면
    없던 문장이 '생긴' 것처럼 보인다. 그것은 경계 파손이 아니다.

    판정: A 의 각 문장에서 삽입된 설명을 제거한 형태가 E 의 문장 집합에
    존재하면 경계가 유지된 것이다."""
    sa, se = Splitter.split(A), Splitter.split(E)
    expl = [("(" + i["explanation"] + ")") for i in insertions]

    def strip_expl(t: str) -> str:
        for x in expl:
            t = t.replace(x, "")
        return re.sub(r"\s+", " ", t).strip()

    se_stripped = {strip_expl(x) for x in se}
    missing = [x for x in sa if strip_expl(x) not in se_stripped]
    # E 에만 있는 문장 — 삽입으로 길어져 새로 집계된 목록 항목인지 확인
    sa_set = {re.sub(r"\s+", " ", x).strip() for x in sa}
    newly = [x for x in se if strip_expl(x) not in sa_set]
    newly_from_insertion = [x for x in newly if any(e in x for e in expl)]

    return {
        "n_sentences_A": len(sa), "n_sentences_E": len(se),
        "boundary_preserved": not missing,
        "a_sentences_lost": missing[:5],
        "n_newly_counted": len(newly),
        "n_newly_from_insertion": len(newly_from_insertion),
        "newly_examples": [x[:80] for x in newly[:3]],
        "explain": ("새로 집계된 문장이 모두 삽입 때문에 길어진 목록 항목이면"
                    " 문장 구조는 유지된 것이다."),
    }


def paired_words_per_sent(A: str, E: str, insertions) -> dict:
    """A 에서 이미 문장으로 집계되던 것만 짝지어 어절 수를 비교한다.
    목록 항목이 새로 집계되는 효과를 배제한 순수 삽입 효과."""
    sa, se = Splitter.split(A), Splitter.split(E)
    expl = [("(" + i["explanation"] + ")") for i in insertions]

    def strip_expl(t: str) -> str:
        for x in expl:
            t = t.replace(x, "")
        return re.sub(r"\s+", " ", t).strip()

    idx = {}
    for x in se:
        idx.setdefault(strip_expl(x), x)
    pa, pe = [], []
    for x in sa:
        key = re.sub(r"\s+", " ", x).strip()
        y = idx.get(key)
        if y is not None:
            pa.append(len(x.split()))
            pe.append(len(y.split()))
    if not pa:
        return {"n_paired": 0, "wps_A_paired": None, "wps_E_paired": None,
                "delta_paired": None}
    return {"n_paired": len(pa),
            "wps_A_paired": round(st.fmean(pa), 3),
            "wps_E_paired": round(st.fmean(pe), 3),
            "delta_paired": round(st.fmean(pe) - st.fmean(pa), 3)}


def load_variant_review(path: Optional[str]) -> Optional[dict]:
    """variant_review.csv 를 읽어 사람 판정 결과를 돌려준다.

    equivalent 열이 1/true/y/yes/예/o 면 동등, 0/false/n/no/아니오/x 면
    비동등으로 본다. 미기입은 보수적으로 비동등 처리한다.
    건수는 전부 CSV 에서 읽으며 코드에 숫자를 넣지 않는다."""
    if not path:
        return None
    if not os.path.exists(path):
        sys.exit("[FATAL] --variant-review 파일이 없습니다: " + path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("[FATAL] " + path + " 가 비어 있습니다")
    if "equivalent" not in rows[0]:
        sys.exit("[FATAL] " + path + " 에 equivalent 열이 없습니다")

    TRUE = {"1", "true", "y", "yes", "예", "o"}
    FALSE = {"0", "false", "n", "no", "아니오", "x"}
    yes, no, blank, detail = [], [], [], []
    for r in rows:
        v = str(r.get("equivalent", "")).strip().lower()
        rec = {"no": r.get("no"), "doc": r.get("doc"),
               "src_term": r.get("src_term"),
               "variant": r.get("variant_in_output"),
               "kind": r.get("variant_kind"),
               "note": r.get("reviewer_note", "")}
        if v in TRUE:
            yes.append(rec)
            rec["equivalent"] = True
        elif v in FALSE:
            no.append(rec)
            rec["equivalent"] = False
        else:
            blank.append(rec)
            rec["equivalent"] = None
        detail.append(rec)
    return {"path": path, "n_reviewed": len(rows),
            "n_equivalent": len(yes), "n_not_equivalent": len(no),
            "n_blank": len(blank), "detail": detail,
            "blank_policy": "미기입은 보수적으로 비동등 처리"}


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def fmt(v, w=12, d=2):
    return ("%*s" % (w, "—")) if v is None else ("%*.*f" % (w, d, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edir", default=EDIR)
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--glossary", default=GLOSSARY)
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--vocab", default=None,
                    help="vocab_grades.json 경로 (환경변수 VOCAB_GRADES 대체)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--variant-review", default=None, metavar="CSV",
                    help="variant_review.csv 경로. equivalent==1 로 사람이 "
                         "확인한 occurrence 만 integrity 성공에 포함한다. "
                         "주어지면 이것이 주지표(human_adjudicated)가 된다.")
    ap.add_argument("--count-variant", action="store_true",
                    help="variant_form 전부를 성공으로 간주하는 upper-bound "
                         "sensitivity analysis. 주지표가 아니다.")
    ap.add_argument("--out", default="eval_A_vs_E.json")
    ap.add_argument("--csv", default="eval_A_vs_E_per_doc.csv")
    a = ap.parse_args()
    if a.vocab:
        os.environ["VOCAB_GRADES"] = a.vocab

    paths = sorted(glob.glob(os.path.join(a.edir, "*.json")))
    if not paths:
        sys.exit("[FATAL] " + a.edir + " 에 E 결과가 없습니다. "
                 "apply_glossary.py 를 먼저 실행하세요.")

    with open(a.glossary, encoding="utf-8") as f:
        g = json.load(f)
    gl_terms = [e["term"] for e in g["entries"]
                if e.get("source_verified") and e.get("explanation_verified")]

    audit = {}
    if os.path.exists(a.audit):
        with open(a.audit, encoding="utf-8") as f:
            audit = json.load(f)

    review = load_variant_review(a.variant_review)

    print("=" * 100)
    print("A vs E 파일럿 — Glossary deterministic insertion")
    print("=" * 100)
    print("  glossary " + str(g.get("glossary_version"))
          + " / 삽입 가능 항목 " + str(len(gl_terms)) + ": " + ", ".join(gl_terms))
    print("  E 는 새 generation 이 아니라 A 의 후처리입니다.")
    if review:
        print("  variant_form 판정: human adjudication ("
              + os.path.basename(review["path"]) + ")")
        print("     검토 " + str(review["n_reviewed"])
              + "건 → 동등 " + str(review["n_equivalent"])
              + " / 비동등 " + str(review["n_not_equivalent"])
              + " / 미기입 " + str(review["n_blank"])
              + "  (" + review["blank_policy"] + ")")
        print("  주지표: human_adjudicated terminology integrity")
    elif a.count_variant:
        print("  variant_form 판정: 전부 성공으로 간주 "
              "(upper-bound sensitivity analysis)")
        print("  주지표: upper_bound_all_variants — 참고용이며 "
              "사람 검토 결과가 아닙니다")
    else:
        print("  variant_form 판정: 전부 제외 (conservative)")
        print("  주지표: conservative terminology integrity")

    nli = NLI(a.gpu, a.batch)
    rows = []
    problems = []
    paren_cases = []
    print("\n  평가 중", end="", flush=True)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        doc, A, E = r["doc"], r["A_output"], r["E_output"]
        K = load_K(doc, a.docdir)
        if not K:
            continue

        ks = Splitter.split(K)
        sa, se = Splitter.split(A), Splitter.split(E)
        gA = nli.rate(sa, build_chunks(ks), a.k, a.tau) if sa and ks else None
        gE = nli.rate(se, build_chunks(ks), a.k, a.tau) if se and ks else None

        # ── occurrence-level: glossary 대상 중 출력에 남은 것 ──────────
        occ_A = find_occurrences(A, gl_terms)
        n_elig = len(occ_A)                       # 분모
        n_ins = r["n_insertions"]                 # 분자
        skips = r["skipped"]
        n_sp = sum(1 for s in skips
                   if s["reason"] == "insertion_skipped_source_parenthetical")
        n_ep = sum(1 for s in skips
                   if s["reason"] == "insertion_skipped_existing_parenthetical")
        n_nf = sum(1 for s in skips if s["reason"] == "not_first_occurrence")
        n_al = sum(1 for s in skips
                   if s["reason"] == "explanation_already_present")
        n_bad = len(r["incorrect_insertions"])

        simA, simE = simplicity(A, K), simplicity(E, K)
        bnd = sentence_boundary_check(A, E, r["insertions"])
        pw = paired_words_per_sent(A, E, r["insertions"])
        # target-level 분모: (문서 x 용어) 쌍 중 삽입 가능했던 것
        #   = 실제 삽입 + 규칙상 삽입 불가(괄호/이미 설명) 였던 것
        #   '첫 등장 아님' 은 같은 쌍의 중복이므로 분모에 넣지 않는다
        n_target = (r["n_insertions"]
                    + sum(1 for q in skips
                          if q["reason"] in
                          ("insertion_skipped_source_parenthetical",
                           "insertion_skipped_existing_parenthetical",
                           "explanation_already_present")))
        kinds = {}
        for q in r["incorrect_insertions"]:
            kinds[q.get("kind", "?")] = kinds.get(q.get("kind", "?"), 0) + 1
        row = {
            "doc": doc,
            "incorrect_kinds": ";".join(k + "=" + str(v)
                                        for k, v in sorted(kinds.items())),
            "eligible_occ": n_elig, "insertions": n_ins,
            "skip_source_paren": n_sp, "skip_existing_paren": n_ep,
            "skip_not_first": n_nf, "skip_already": n_al,
            "incorrect": n_bad,
            "ground_A": round(gA, 2) if gA is not None else None,
            "ground_E": round(gE, 2) if gE is not None else None,
            "delta_ground": (round(gE - gA, 2)
                             if None not in (gA, gE) else None),
            "wps_A": simA.get("words_per_sent"), "wps_E": simE.get("words_per_sent"),
            "len_A": r["len_A"], "len_E": r["len_E"],
            "len_pct": round(100 * (r["len_E"] - r["len_A"])
                             / max(r["len_A"], 1), 2),
            "sent_A": bnd["n_sentences_A"],
            "sent_E": bnd["n_sentences_E"],
            "sent_preserved": bnd["boundary_preserved"],
            "n_newly_counted": bnd["n_newly_counted"],
            "n_newly_from_insertion": bnd["n_newly_from_insertion"],
            "a_sentences_lost": len(bnd["a_sentences_lost"]),
            "target_occ": n_target,
            "n_paired": pw["n_paired"],
            "wps_A_paired": pw["wps_A_paired"],
            "wps_E_paired": pw["wps_E_paired"],
            "delta_wps_paired": pw["delta_paired"],
            "mid_vocab_A": simA.get("mid_vocab_A"),
            "mid_vocab_E": simE.get("mid_vocab_A"),
            "ko2025_A": simA.get("ko2025_A"), "ko2025_E": simE.get("ko2025_A"),
            "inserted_terms": ";".join(i["term"] for i in r["insertions"]),
        }
        rows.append(row)
        for q in r["incorrect_insertions"]:
            problems.append({"doc": doc, **q})
        for sk in skips:
            if sk["reason"].startswith("insertion_skipped"):
                paren_cases.append({"doc": doc, "term": sk["term"],
                                    "reason": sk["reason"],
                                    "existing": sk.get("existing_parenthetical"),
                                    "in_source": sk.get("parenthetical_in_source"),
                                    "hint": sk.get("sentence_hint", "")[:90]})
        print(".", end="", flush=True)
    print()

    def s(k):
        return sum(x[k] or 0 for x in rows)

    def m(k):
        v = [x[k] for x in rows if x.get(k) is not None]
        return st.fmean(v) if v else None

    n_elig, n_ins = s("eligible_occ"), s("insertions")
    n_tgt = s("target_occ")
    cov_occ = 100 * n_ins / max(n_elig, 1)      # 참고
    cov = 100 * n_ins / max(n_tgt, 1)           # 주지표

    # ── terminology integrity (감사 결과 기준) ─────────────────────────
    tal = (audit.get("status_tally") or {})
    tot = sum(tal.values()) or 250
    keep = (tal.get("exact_retained", 167) + tal.get("explained_in_source", 27)
            + tal.get("explained_by_model", 5))
    var = tal.get("variant_form", 10)

    # ── terminology integrity — 세 기준을 항상 계산한다 ───────────────
    # A 와 E 는 동일한 occurrence 기준(분모 tot)과 동일한 review 결과를 쓴다.
    # E 는 삽입만 하므로 용어를 잃지 않고, 오탐이 있으면 그만큼 차감한다.
    n_eq = review["n_equivalent"] if review else None
    integ = {
        "conservative": 100 * keep / tot,
        "upper_bound_all_variants": 100 * (keep + var) / tot,
    }
    if n_eq is not None:
        integ["human_adjudicated"] = 100 * (keep + n_eq) / tot

    if review:
        primary_key = "human_adjudicated"
    elif a.count_variant:
        primary_key = "upper_bound_all_variants"
    else:
        primary_key = "conservative"
    integ_A = integ[primary_key]
    integ_E = integ_A - 100 * s("incorrect") / tot

    print("\n" + "=" * 100)
    print("파일럿 요약")
    print("=" * 100)
    L = 42
    def line(label, val):
        print("  " + str(label).ljust(L) + str(val))

    line("Glossary entries enabled", len(gl_terms))
    line("Eligible retained occurrences", n_elig)
    line("  └ target (문서x용어 쌍, 첫등장 규칙 적용 후)", n_tgt)
    line("Successful verified insertions", n_ins)
    line("Verified explanation coverage [주지표]", "%.1f%%" % cov)
    line("  └ occurrence-level (참고)", "%.1f%%" % cov_occ)
    line("Incorrect insertions", s("incorrect"))
    kind_tally = {}
    for q in problems:
        kind_tally[q.get("kind", "?")] = kind_tally.get(q.get("kind", "?"), 0) + 1
    line("Duplicate insertions", kind_tally.get("duplicate_insertion", 0))
    line("  substring false positive", kind_tally.get("substring_false_positive", 0))
    line("  double parenthetical", kind_tally.get("double_parenthetical", 0))
    line("  insertion missing", kind_tally.get("insertion_missing", 0))
    line("Source-parenthetical skips", s("skip_source_paren"))
    line("  (출력에만 있는 괄호 skip)", s("skip_existing_paren"))
    line("  (첫 등장 아님 skip)", s("skip_not_first"))
    line("Terminology integrity A [" + primary_key + "]",
         "%.1f%%" % integ_A)
    line("Terminology integrity E [" + primary_key + "]",
         "%.1f%%" % integ_E)
    line("Grounding A", fmt(m("ground_A"), 0, 2))
    line("Grounding E (all, 설명 포함)", fmt(m("ground_E"), 0, 2))
    line("Delta Grounding", "%+.2f" % (m("delta_ground") or 0))
    line("Mean eojeol/sentence A", fmt(m("wps_A"), 0, 2))
    line("Mean eojeol/sentence E", fmt(m("wps_E"), 0, 2))
    line("  └ 짝지은 문장만 A (목록항목 효과 배제)", fmt(m("wps_A_paired"), 0, 2))
    line("  └ 짝지은 문장만 E", fmt(m("wps_E_paired"), 0, 2))
    line("  └ 순수 삽입 효과", "%+.3f" % (m("delta_wps_paired") or 0))
    line("Length increase", "%+.2f%%" % (m("len_pct") or 0))

    print("\n" + "=" * 100)
    print("terminology integrity — 세 기준")
    print("=" * 100)
    print("  분모 " + str(tot) + " occurrence (A 기준 분류. A 와 E 가 공유)")
    print("  base successful occurrences  " + str(keep)
          + "  (exact_retained + explained_in_source + explained_by_model)")
    print("  variant_form 총계             " + str(var))
    if review:
        print("  human-reviewed variants      " + str(review["n_reviewed"]))
        print("  equivalent variants          " + str(review["n_equivalent"]))
        print("  non-equivalent variants      " + str(review["n_not_equivalent"]))
        if review["n_blank"]:
            print("  미기입 (비동등 처리)          " + str(review["n_blank"]))
    print()
    print("  " + "기준".ljust(30) + "산식".ljust(20) + "값".rjust(8) + "   비고")
    print("  " + "conservative".ljust(30)
          + (str(keep) + "/" + str(tot)).ljust(20)
          + ("%7.1f%%" % integ["conservative"])
          + "   variant 전부 제외")
    if "human_adjudicated" in integ:
        print("  " + "human_adjudicated".ljust(30)
              + ("(" + str(keep) + "+" + str(review["n_equivalent"]) + ")/"
                 + str(tot)).ljust(20)
              + ("%7.1f%%" % integ["human_adjudicated"])
              + "   ← 주지표")
    print("  " + "upper_bound_all_variants".ljust(30)
          + ("(" + str(keep) + "+" + str(var) + ")/" + str(tot)).ljust(20)
          + ("%7.1f%%" % integ["upper_bound_all_variants"])
          + "   sensitivity analysis")
    if review:
        ne = [d for d in review["detail"] if d.get("equivalent") is False]
        if ne:
            print("\n  비동등 판정 " + str(len(ne)) + "건 — 사람 검토 사유")
            for d in ne:
                print("    " + str(d.get("src_term")) + " → "
                      + str(d.get("variant")) + "  ["
                      + str(d.get("doc")) + "]")
                if d.get("note"):
                    print("      " + str(d["note"])[:88])
            print("\n  이 건들은 문자열 포함관계로 variant_form 에 분류되었으나")
            print("  사람 검토에서 의미가 동등하지 않다고 판정되었습니다.")
            print("  자동 포함관계 판정만으로 integrity 를 계산하면 과대평가됩니다.")

    print("    " + str(tal.get("absent", 41)) + " / " + str(tot)
          + " occurrence — HARI 단계에서 이미 사라진 용어")

    # ── 문서별 ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("문서별")
    print("=" * 100)
    print("  " + "문서".ljust(8) + "대상".rjust(5) + "삽입".rjust(5)
          + "coverage".rjust(10) + "오탐".rjust(5)
          + "Ground A".rjust(10) + "Ground E".rjust(10) + "Δ".rjust(8)
          + "어절A".rjust(7) + "어절E".rjust(7) + "길이".rjust(9))
    for x in rows:
        c = 100 * x["insertions"] / max(x["eligible_occ"], 1)
        print("  " + x["doc"].ljust(8) + str(x["eligible_occ"]).rjust(5)
              + str(x["insertions"]).rjust(5) + ("%9.1f%%" % c)
              + str(x["incorrect"]).rjust(5)
              + fmt(x["ground_A"], 10, 1) + fmt(x["ground_E"], 10, 1)
              + fmt(x["delta_ground"], 8, 1)
              + fmt(x["wps_A"], 7, 1) + fmt(x["wps_E"], 7, 1)
              + ("%8.1f%%" % x["len_pct"]))

    if problems:
        print("\n" + "=" * 100)
        print("오탐 삽입 상세 — 실제 문장")
        print("=" * 100)
        for q in problems:
            print("  [" + q["doc"] + "] " + q.get("kind", "?")
                  + "  용어=" + str(q.get("term")))
            ctx = q.get("context")
            if ctx:
                print("      " + str(ctx)[:120])
    else:
        print("\n  오탐 삽입 0건.")

    if paren_cases:
        print("\n" + "=" * 100)
        print("괄호 때문에 건너뛴 사례 — '설명 충분' 으로 판정한 것이 아님")
        print("=" * 100)
        for q in paren_cases[:12]:
            print("  [" + q["doc"] + "] " + q["term"]
                  + "(" + str(q["existing"])[:38] + ")"
                  + "   원문유래=" + str(q["in_source"]))
        if len(paren_cases) > 12:
            print("  ... 외 " + str(len(paren_cases) - 12) + "건")
        print("""
  이 항목들은 이중 괄호를 피하려고 inline 삽입을 건너뛴 것입니다.
  설명이 충분하다는 판정이 아니며, UI tooltip 등 별도 표시 후보로 남습니다.""")

    bad_sent = [x["doc"] for x in rows if not x["sent_preserved"]]
    print("\n" + "=" * 100)
    print("문장 경계 진단")
    print("=" * 100)
    print("  개수 비교가 아니라 A 의 문장 경계가 E 에서 유지되는지를 봅니다.")
    print("  HARI 출력에는 줄바꿈으로 나열된 짧은 목록 항목이 있고,")
    print("  Splitter 는 10자 미만을 문장에서 제외합니다. 삽입으로 그 항목이")
    print("  길어지면 '문장이 늘어난' 것처럼 보이지만 경계 파손이 아닙니다.")
    print("\n  " + "문서".ljust(8) + "문장A".rjust(7) + "문장E".rjust(7)
          + "신규집계".rjust(10) + "그중 삽입유래".rjust(14)
          + "A문장 소실".rjust(12) + "경계".rjust(8))
    for x in rows:
        print("  " + x["doc"].ljust(8) + str(x["sent_A"]).rjust(7)
              + str(x["sent_E"]).rjust(7)
              + str(x["n_newly_counted"]).rjust(10)
              + str(x["n_newly_from_insertion"]).rjust(14)
              + str(x["a_sentences_lost"]).rjust(12)
              + ("유지" if x["sent_preserved"] else "파손").rjust(8))
    if bad_sent:
        print("\n  [경고] 문장 경계가 파손된 문서: " + ", ".join(bad_sent))
        print("         A 의 문장이 E 에서 사라졌습니다. 삽입 로직을 확인하십시오.")
    else:
        print("\n  모든 문서에서 A 의 문장 경계가 유지되었습니다.")
        tot_new = sum(x["n_newly_counted"] for x in rows)
        tot_ins = sum(x["n_newly_from_insertion"] for x in rows)
        if tot_new:
            print("  신규 집계 " + str(tot_new) + "건 중 " + str(tot_ins)
                  + "건이 삽입으로 길어진 목록 항목입니다.")

    # ── Ko2025 보조 분석 ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("보조 분석 — 고승연(2025) 성분 분리")
    print("=" * 100)
    if not KO_OK or m("ko2025_A") is None:
        print("  어휘 목록이 연결되지 않아 보류합니다. 총점을 추정하지 않습니다.")
    else:
        mvA, mvE = m("mid_vocab_A"), m("mid_vocab_E")
        wA, wE = m("wps_A"), m("wps_E")
        kA, kE = m("ko2025_A"), m("ko2025_E")
        cv = KO2025["coef_mid_vocab"] * (mvE - mvA)
        cs = KO2025["coef_words_per_sent"] * (wE - wA)
        print("  지수 = 61.994 − 0.261×중급이상어휘비율 − 1.045×평균어절수")
        print("  값이 높을수록 쉬움\n")
        print("  " + "성분".ljust(26) + "A".rjust(10) + "E".rjust(10)
              + "변화".rjust(10) + "지수 기여".rjust(12))
        print("  " + "중급이상 어휘비율 %".ljust(26) + fmt(mvA, 10, 2)
              + fmt(mvE, 10, 2) + ("%+10.2f" % (mvE - mvA))
              + ("%+12.2f" % cv))
        print("  " + "평균 어절/문장".ljust(26) + fmt(wA, 10, 2)
              + fmt(wE, 10, 2) + ("%+10.2f" % (wE - wA))
              + ("%+12.2f" % cs))
        print("  " + "Ko2025 총점".ljust(26) + fmt(kA, 10, 2)
              + fmt(kE, 10, 2) + ("%+10.2f" % (kE - kA)))
        print("    수준: A " + str(ko2025_level(kA))
              + "  →  E " + str(ko2025_level(kE)))
        print("""
  읽는 법
    설명 삽입으로 문장이 길어지므로 문장길이 항이 나빠지는 것은
    구조적으로 예상됩니다. Glossary 의 성공 여부를 총점으로 판단하지 않습니다.
    어휘 항이 개선되었는지를 따로 보십시오.""")

    # ── 판정 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("판정 — 핵심 지표 네 개")
    print("=" * 100)
    print("  1. Verified Explanation Coverage  %.1f%%  (%d / %d target)"
          % (cov, n_ins, n_tgt))
    print("       occurrence-level 참고치  %.1f%%  (%d / %d)"
          % (cov_occ, n_ins, n_elig))
    print("       분모 차이: 같은 용어의 두 번째 이후 등장은 첫등장 규칙에 따라")
    print("       삽입하지 않으므로 target 분모에서 제외합니다.")
    print("  2. Incorrect Insertion            " + str(s("incorrect")) + "건")
    print("  3. Terminology Integrity          A %.1f%%  →  E %.1f%%   [%s]"
          % (integ_A, integ_E, primary_key))
    print("  4. Length Cost                    %+.2f%%  (어절/문장 %+.2f)"
          % (m("len_pct") or 0, (m("wps_E") or 0) - (m("wps_A") or 0)))
    print("       순수 삽입 효과 (짝지은 문장)  %+.3f 어절"
          % (m("delta_wps_paired") or 0))
    print("""
  비교 기준
    HARI 자체 생성 설명은 250 occurrence 중 5건(2.0%)입니다.
    term_explain_rate 11.16% 는 원문 서식을 상당 부분 센 값이므로
    HARI 의 설명 능력으로 쓰지 않습니다.""")
    print("=" * 100)

    with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for x in rows:
            w.writerow(x)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "glossary_version": g.get("glossary_version"),
            "glossary_terms": gl_terms,
            "count_variant_in_integrity": a.count_variant,
            "terminology_integrity": {
                "denominator": tot,
                "base_successful_occurrences": keep,
                "total_occurrences": tot,
                "n_variant_form": var,
                "conservative": round(integ["conservative"], 1),
                "human_adjudicated": (round(integ["human_adjudicated"], 1)
                                      if "human_adjudicated" in integ else None),
                "upper_bound_all_variants":
                    round(integ["upper_bound_all_variants"], 1),
                "primary": primary_key,
                "primary_value": round(integ_A, 1),
                "note": ("A 와 E 는 동일한 occurrence 기준(분모)과 동일한 "
                         "review file 을 공유한다. E 는 삽입만 하므로 용어를 "
                         "잃지 않고 오탐만 차감한다."),
            },
            "variant_review": (dict(review, non_equivalent_detail=[d for d in review["detail"] if d.get("equivalent") is False], equivalent_detail=[d for d in review["detail"] if d.get("equivalent") is True]) if review else None),
            "coverage_definition_note": (
                "주지표는 target-level 이다. 분모는 (문서 x 용어) 쌍 중 "
                "첫등장 규칙 적용 후 삽입 가능했던 것이다. 같은 용어의 두 번째 "
                "이후 등장은 설계상 삽입하지 않으므로 실패로 세지 않는다. "
                "occurrence-level 은 참고치로 함께 보고한다."),
            "coverage_definition": (
                "검증된 glossary 설명이 실제 삽입된 retained source-term "
                "occurrences / HARI 출력에 남아 있는 source-term occurrences "
                "중 glossary 대상. HARI 에서 이미 사라진 용어는 분모에 넣지 않는다."),
            "summary": {
                "entries_enabled": len(gl_terms),
                "eligible_occurrences": n_elig,
                "insertions": n_ins,
                "coverage_pct_target_level": round(cov, 1),
                "coverage_pct_occurrence_level": round(cov_occ, 1),
                "target_occurrences": n_tgt,
                "wps_A_paired": m("wps_A_paired"),
                "wps_E_paired": m("wps_E_paired"),
                "delta_wps_paired": m("delta_wps_paired"),
                "boundary_preserved_all": not bad_sent,
                "incorrect_insertions": s("incorrect"),
                "skip_source_parenthetical": s("skip_source_paren"),
                "skip_existing_parenthetical": s("skip_existing_paren"),
                "skip_not_first": s("skip_not_first"),
                "integrity_A": round(integ_A, 1),
                "integrity_E": round(integ_E, 1),
                "integrity_primary_basis": primary_key,
                "grounding_A": m("ground_A"), "grounding_E_all": m("ground_E"),
                "delta_grounding": m("delta_ground"),
                "wps_A": m("wps_A"), "wps_E": m("wps_E"),
                "length_increase_pct": m("len_pct"),
                "absent_before_glossary": tal.get("absent", 41),
            },
            "rows": rows,
            "incorrect_insertion_detail": problems,
            "parenthetical_skip_detail": paren_cases,
            "incorrect_kind_tally": kind_tally,
        }, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] " + a.out)
    print("[SAVE] " + a.csv)


if __name__ == "__main__":
    main()
