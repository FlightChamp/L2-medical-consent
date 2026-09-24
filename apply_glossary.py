#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_glossary.py — 검증된 glossary 를 HARI P0 출력에 결정적으로 삽입
=====================================================================
A = 기존 HARI P0 출력 (outputs_prompt/{doc}__P0.json)
E = 동일한 A + verified glossary insertion

E 는 새 LLM generation 이 아니다. 따라서 A 와 E 의 차이는
glossary 후처리 효과만 반영한다.

DB 는 렌더링 방식을 알지 못한다. 표시 형태는 --render 로 결정한다.
    inline     불유합  →  불유합(뼈가 제대로 붙지 않음)
    footnote   본문은 그대로 두고 문서 끝에 용어 설명 목록을 붙인다
    none       삽입하지 않고 provenance 만 기록한다 (UI tooltip 용)

삽입 조건 (전부 만족해야 삽입):
    1. source_verified == true 이고 explanation_verified == true
    2. HARI 출력에 해당 용어가 실제로 존재
    3. 이미 같은 설명이 붙어 있지 않음
    4. source-parenthetical 이 아님
       (원문에서 온 괄호가 붙어 있으면 이중 괄호를 만들지 않는다.
        단 '괄호 존재 = 설명 충분' 으로 판정하지 않고 skip 으로 기록한다)
    5. longest-match 우선, 다른 단어 내부가 아님
    6. 문서 내 첫 등장에만
    7. 한 occurrence 에 둘 이상 적용하지 않음

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python apply_glossary.py                       # inline, 13문서
    python apply_glossary.py --render footnote
    python apply_glossary.py --render none
    python apply_glossary.py --docs doc1 doc10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, Splitter
except ImportError as e:
    sys.exit("[FATAL] prompt_ladder.py 필요: " + str(e))

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
HARI_P0 = os.path.join(HOME, "outputs_prompt", "{doc}__P0.json")
GLOSSARY = os.path.join(HOME, "glossary_v1.json")
OUTDIR = os.path.join(HOME, "outputs_glossary")

HANGUL = re.compile(r"[가-힣]")
JOSA = (
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "에게", "에는",
    "으로", "로", "와", "과", "도", "만", "부터", "까지", "라", "이라",
    "이나", "나", "이며", "며", "이고", "고", "인", "이란", "란",
    "처럼", "보다", "조차", "마저", "밖에", "이라고", "라고",
    "술", "시", "후", "전", "중", "및", "또는",
)
JOSA_RE = re.compile("^(?:" + "|".join(sorted(JOSA, key=len, reverse=True)) + ")")
# 용어 바로 뒤의 괄호
PAREN_AFTER = re.compile(r"^\s*[（(]\s*([^)）]{1,80})\s*[)）]")


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_A(doc: str, pattern: str = HARI_P0) -> Optional[str]:
    p = pattern.format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("out")


def _entry_source(e: dict) -> str:
    """출처 이름. v1.1 부터 source_name, 그 이전은 source 를 쓴다."""
    return e.get("source_name") or e.get("source") or ""


def load_glossary(path: str) -> Tuple[dict, List[dict]]:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    enabled = [e for e in g.get("entries", [])
               if e.get("source_verified") and e.get("explanation_verified")
               and e.get("easy_explanation")]
    return g, enabled


# ===========================================================================
# occurrence 탐색 — longest-match, 조사 허용, 겹침 제거
# ===========================================================================

def find_occurrences(text: str, terms: List[str]) -> List[Tuple[int, int, str]]:
    hits: List[Tuple[int, int, str]] = []
    for t in sorted(set(terms), key=len, reverse=True):
        start = 0
        while True:
            i = text.find(t, start)
            if i < 0:
                break
            start = i + 1
            if i > 0 and HANGUL.match(text[i - 1]):
                continue                      # 다른 단어 내부
            j = i + len(t)
            tail = text[j:j + 6]
            if tail and HANGUL.match(tail[0]) and not JOSA_RE.match(tail):
                continue                      # 더 긴 단어의 앞부분
            hits.append((i, j, t))
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: List[Tuple[int, int, str]] = []
    last = -1
    for s, e, t in hits:
        if s >= last:                         # overlap suppression
            kept.append((s, e, t))
            last = e
    return kept


def inside_parenthesis(text: str, pos: int) -> bool:
    """pos 가 괄호 안쪽인지. 앞쪽의 여는/닫는 괄호 수를 세어 판정한다.

    원문 '합병증(후유증)' 에서 후유증의 위치는 괄호 안이므로 True.
    여기에 설명을 넣으면 괄호가 중첩되어 서식이 깨진다."""
    depth = 0
    for ch in text[:pos]:
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth = max(0, depth - 1)
    return depth > 0


def paren_after(text: str, end: int) -> Optional[str]:
    m = PAREN_AFTER.match(text[end:end + 90])
    return m.group(1).strip() if m else None


# ===========================================================================
# 삽입
# ===========================================================================

def apply(doc: str, K: str, A: str, enabled: List[dict],
          render: str) -> dict:
    by_term = {e["term"]: e for e in enabled}
    occs = find_occurrences(A, list(by_term))

    insertions: List[dict] = []
    skipped: List[dict] = []
    seen_terms: Set[str] = set()
    plan: List[Tuple[int, int, str, str]] = []   # (start, end, term, explanation)

    for s, e, term in occs:
        ent = by_term[term]
        expl = ent["easy_explanation"]
        rec = {"term": term, "start": s, "end": e,
               "sentence_hint": A[max(0, s - 30):e + 30]}

        if term in seen_terms:
            skipped.append({**rec, "reason": "not_first_occurrence"})
            continue

        # 괄호 안쪽이면 삽입하지 않는다. 중첩 괄호를 만들기 때문이다.
        # 예) 원문 '합병증(후유증)' 의 후유증
        if inside_parenthesis(A, s):
            skipped.append({**rec,
                            "reason": "insertion_skipped_inside_parenthesis",
                            "note": ("이 위치는 괄호 안쪽입니다. 설명을 넣으면 "
                                     "괄호가 중첩되어 원문 서식이 깨집니다. "
                                     "같은 용어가 괄호 밖에 다시 나오면 "
                                     "그쪽에 삽입됩니다.")})
            continue

        pa = paren_after(A, e)
        if pa is not None:
            # 원문에도 같은 형태가 있으면 서식에서 온 것이다
            in_source = re.search(re.escape(term) + r"\s*[（(]", K) is not None
            reason = ("insertion_skipped_source_parenthetical" if in_source
                      else "insertion_skipped_existing_parenthetical")
            skipped.append({**rec, "reason": reason,
                            "existing_parenthetical": pa,
                            "parenthetical_in_source": in_source,
                            "note": ("괄호 존재를 '설명 충분' 으로 판정하지 않는다. "
                                     "UI tooltip 등 별도 표시 후보로 남긴다.")})
            seen_terms.add(term)
            continue

        if expl in A:
            skipped.append({**rec, "reason": "explanation_already_present"})
            seen_terms.add(term)
            continue

        plan.append((s, e, term, expl))
        seen_terms.add(term)

    # 뒤에서부터 삽입해 앞쪽 인덱스가 밀리지 않게 한다
    E = A
    for s, e, term, expl in sorted(plan, key=lambda x: -x[0]):
        E = E[:e] + "(" + expl + ")" + E[e:] if render == "inline" else E
        ent = by_term[term]
        src_name = _entry_source(ent)
        insertions.append({
            "term": term, "explanation": expl,
            "position_in_A": s, "first_occurrence": True,
            "render": render,
            # 두 키를 모두 기록한다. 읽는 쪽이 어느 이름을 쓰든 깨지지 않는다.
            "source": src_name,
            "source_name": src_name,
            "source_url": ent.get("source_url"),
            "source_type": ent.get("source_type"),
            "source_scope": ent.get("source_scope", "general"),
            "version_added": ent.get("version_added"),
            "source_verified": True, "explanation_verified": True,
        })
    insertions.reverse()

    if render == "footnote" and plan:
        lines = ["", "", "[용어 설명]"]
        for s, e, term, expl in sorted(plan, key=lambda x: x[0]):
            lines.append("- " + term + ": " + expl)
        E = A + "\n".join(lines)

    # ── 삽입 검증 (incorrect insertion 탐지) ──────────────────────────
    problems: List[dict] = []
    for ins in insertions:
        t, x = ins["term"], ins["explanation"]
        pat = re.escape(t) + r"\s*[（(]\s*" + re.escape(x)
        n = len(re.findall(pat, E))
        if n == 0:
            problems.append({"term": t, "kind": "insertion_missing"})
        elif n > 1:
            problems.append({"term": t, "kind": "duplicate_insertion",
                             "count": n})
        # 다른 단어 내부에 들어갔는지
        for m in re.finditer(re.escape(t) + r"\s*[（(]" + re.escape(x[:6]), E):
            i = m.start()
            if i > 0 and HANGUL.match(E[i - 1]):
                problems.append({"term": t, "kind": "substring_false_positive",
                                 "context": E[max(0, i - 20):i + 30]})
        # 이중 괄호
        for m in re.finditer(re.escape(t) + r"\s*[（(][^)）]*[)）]\s*[（(]", E):
            problems.append({"term": t, "kind": "double_parenthetical",
                             "context": E[m.start():m.start() + 70]})
        # 중첩 괄호 — 삽입 결과가 다른 괄호 안에 들어간 경우
        for m in re.finditer(re.escape(t) + r"\s*[（(]" + re.escape(x[:8]), E):
            if inside_parenthesis(E, m.start()):
                problems.append({"term": t, "kind": "nested_parenthesis",
                                 "context": E[max(0, m.start() - 20):
                                              m.start() + 50]})

    # ── 문장 구조 진단 ────────────────────────────────────────────────
    sa, se = Splitter.split(A), Splitter.split(E)
    diag = {"n_sentences_A": len(sa), "n_sentences_E": len(se),
            "sentence_count_preserved": len(sa) == len(se)}
    if not diag["sentence_count_preserved"]:
        diag["warning"] = ("glossary 삽입으로 문장 분할 결과가 달라졌습니다. "
                           "설명문에 문장 종결 어미가 포함되었을 수 있습니다.")

    return {
        "doc": doc,
        "route": "E",
        "source_generation": "outputs_prompt/" + doc + "__P0.json",
        "generation_note": "E 는 새 LLM generation 이 아니다. A 에 glossary 를 후처리로 삽입한 것이다.",
        "render": render,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "A_output": A,
        "E_output": E,
        "len_A": len(A), "len_E": len(E),
        "n_insertions": len(insertions),
        "insertions": insertions,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "incorrect_insertions": problems,
        "sentence_diagnostic": diag,
        "eligible_occurrences": len(occs),
    }


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", default=GLOSSARY)
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--render", choices=["inline", "footnote", "none"],
                    default="inline")
    ap.add_argument("--adir", default=os.path.join(HOME, "outputs_prompt"),
                    help="A(HARI P0) 결과 폴더")
    ap.add_argument("--apat", default=None,
                    help="A 파일 패턴. 기본 <adir>/{doc}__P0.json")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    if not a.apat:
        a.apat = os.path.join(a.adir, "{doc}__P0.json")

    if not os.path.exists(a.glossary):
        sys.exit("[FATAL] " + a.glossary + " 없음")
    g, enabled = load_glossary(a.glossary)

    if not enabled:
        n_all = len(g.get("entries", []))
        n_sv = sum(1 for e in g.get("entries", []) if e.get("source_verified"))
        n_ev = sum(1 for e in g.get("entries", [])
                   if e.get("explanation_verified"))
        sys.exit(
            "[FATAL] 삽입 가능한 glossary 항목이 없습니다. 중단합니다.\n"
            "  파일          " + a.glossary + "\n"
            "  version       " + str(g.get("glossary_version")) + "\n"
            "  entries       " + str(n_all) + "\n"
            "  source_verified=true       " + str(n_sv) + "\n"
            "  explanation_verified=true  " + str(n_ev) + "\n"
            "\n"
            "  삽입에는 두 값이 모두 true 여야 합니다.\n"
            "  확정본은 entries 6, 두 값 모두 6 입니다. 숫자가 다르면\n"
            "  이전 버전 파일이 올라간 것이니 재전송하십시오.")

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))

    outdir = os.path.join(a.outdir, "E")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 96)
    print("Glossary 적용 — A → E")
    print("=" * 96)
    print("  glossary        " + str(g.get("glossary_version")))
    print("  삽입 가능 항목    " + str(len(enabled)) + " / 전체 "
          + str(len(g.get("entries", []))))
    for e in enabled:
        print("     " + e["term"] + " — " + e["easy_explanation"])
    print("  render          " + a.render)
    print("  A 출력          " + a.apat)
    print("  문서            " + str(len(docs)) + "건")
    print("  출력            " + outdir + "/")
    print("\n  E 는 새 generation 이 아니라 A 의 후처리입니다.")

    rows = []
    print("\n  " + "문서".ljust(8) + "대상".rjust(5) + "삽입".rjust(5)
          + "skip".rjust(6) + "오탐".rjust(5) + "문장A".rjust(7)
          + "문장E".rjust(7) + "길이변화".rjust(9) + "   삽입된 용어")
    for d in docs:
        K, A = load_K(d, a.docdir), load_A(d, a.apat)
        if not K or not A:
            print("  " + d.ljust(8) + "  입력 없음 — 건너뜀")
            continue
        r = apply(d, K, A, enabled, a.render)
        with open(os.path.join(outdir, d + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({**r, "glossary_version": g.get("glossary_version"),
                       "glossary_entries_enabled":
                           [{"term": e["term"],
                             "source_name": _entry_source(e),
                             "source_url": e.get("source_url"),
                             "source_type": e.get("source_type"),
                             "source_scope": e.get("source_scope"),
                             "version_added": e.get("version_added"),
                             "source_verified": True,
                             "explanation_verified": True} for e in enabled],
                       "nature": g.get("nature")},
                      f, ensure_ascii=False, indent=2)
        dg = r["sentence_diagnostic"]
        pct = 100 * (r["len_E"] - r["len_A"]) / max(r["len_A"], 1)
        terms = ", ".join(i["term"] for i in r["insertions"])
        print("  " + d.ljust(8) + str(r["eligible_occurrences"]).rjust(5)
              + str(r["n_insertions"]).rjust(5) + str(r["n_skipped"]).rjust(6)
              + str(len(r["incorrect_insertions"])).rjust(5)
              + str(dg["n_sentences_A"]).rjust(7)
              + str(dg["n_sentences_E"]).rjust(7)
              + ("%+8.1f%%" % pct) + "   " + terms, flush=True)
        rows.append(r)

    # ── 요약 ──────────────────────────────────────────────────────────
    n_ins = sum(r["n_insertions"] for r in rows)
    n_skip = sum(r["n_skipped"] for r in rows)
    n_bad = sum(len(r["incorrect_insertions"]) for r in rows)
    n_elig = sum(r["eligible_occurrences"] for r in rows)
    sp = sum(1 for r in rows for s in r["skipped"]
             if s["reason"] == "insertion_skipped_source_parenthetical")
    ep = sum(1 for r in rows for s in r["skipped"]
             if s["reason"] == "insertion_skipped_existing_parenthetical")
    nf = sum(1 for r in rows for s in r["skipped"]
             if s["reason"] == "not_first_occurrence")
    ip = sum(1 for r in rows for s in r["skipped"]
             if s["reason"] == "insertion_skipped_inside_parenthesis")
    al = sum(1 for r in rows for s in r["skipped"]
             if s["reason"] == "explanation_already_present")
    bad_sent = [r["doc"] for r in rows
                if not r["sentence_diagnostic"]["sentence_count_preserved"]]

    print("\n" + "=" * 96)
    print("요약")
    print("=" * 96)
    print("  glossary 항목 (삽입 가능)        " + str(len(enabled)))
    print("  대상 occurrence                " + str(n_elig))
    print("  삽입 성공                       " + str(n_ins))
    print("  skip 합계                      " + str(n_skip))
    print("    첫 등장 아님                  " + str(nf))
    print("    원문 괄호 (source-paren)      " + str(sp))
    print("    출력에만 있는 괄호             " + str(ep))
    print("    괄호 안쪽 (중첩 방지)          " + str(ip))
    print("    설명이 이미 존재               " + str(al))
    print("  오탐 삽입                       " + str(n_bad))
    print("  문장 수가 달라진 문서             " + str(len(bad_sent))
          + ((" — " + ", ".join(bad_sent)) if bad_sent else ""))

    if n_bad:
        print("\n  오탐 삽입 상세")
        for r in rows:
            for p in r["incorrect_insertions"]:
                print("    [" + r["doc"] + "] " + p["kind"] + " — "
                      + p.get("term", "") + "  "
                      + str(p.get("context", ""))[:70])

    if sp or ep:
        print("\n  괄호 때문에 건너뛴 사례 (설명 충분으로 판정한 것이 아님)")
        n = 0
        for r in rows:
            for s in r["skipped"]:
                if s["reason"].startswith("insertion_skipped") and n < 8:
                    print("    [" + r["doc"] + "] " + s["term"]
                          + "(" + str(s.get("existing_parenthetical"))[:34] + ")"
                          + "  원문유래=" + str(s.get("parenthetical_in_source")))
                    n += 1

    print("\n다음\n  python eval_A_vs_E.py")
    print("=" * 96)


if __name__ == "__main__":
    main()
