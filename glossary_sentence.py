#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glossary_sentence.py — Glossary 삽입 방식 비교: 괄호형(inline) vs 문장 분리형(sentence)
==========================================================================================
교수님 지시 A1 · A2 대응 실험.

    A          HARI P0 출력 (outputs_prompt/{doc}__P0.json)         ← 기존, 수정 안 함
    E_inline   A + '용어(설명)'                (outputs_glossary_v11/E/)  ← 기존, 수정 안 함
    E_sent     A + 해당 문장 뒤에 '용어는 설명입니다.' 를 별도 문장으로 추가
                                              (outputs_glossary_v11_sent/E/)  ← 이번에 새로 생성

독립 변수는 '설명을 붙이는 형태' 하나뿐이다.
    · 같은 glossary(v1.1, 11개), 같은 A, 같은 삽입 대상(target)을 쓴다.
      삽입 대상 판정은 apply_glossary.apply() 를 그대로 호출해 얻으므로
      inline 과 sentence 의 대상 집합이 구조적으로 동일하다. (실행 중 재확인)
    · 평가는 기존 eval_A_vs_E.py 를 그대로 호출한다. 평가 코드를 바꾸지 않는다.

가설 (A2 분해 결과에서 도출)
    inline 방식의 Ko2025 하락(-0.85) 중 약 80%(-0.68)가 문장당 어절 수 증가에서 왔다.
    설명을 별도의 짧은 문장으로 빼면 기존 문장은 길어지지 않으므로
    어절 항의 악화가 사라지거나 오히려 개선될 것이다.
    대신 설명 문장은 원문에 없는 문장이므로 Grounding(NLI) 비용이 커질 수 있다.
    → Simplicity 와 Safety 를 둘 다 보고하고, 채택 여부를 자동으로 결정하지 않는다.

문장 분리형 형태
    '{용어}{은/는} {easy_explanation}입니다.'
    예) 출혈은 피가 혈관 밖으로 나오는 것입니다.
    조사 은/는 은 용어 마지막 글자의 받침으로 결정한다.
    삽입 위치는 용어가 처음 나온 문장의 바로 뒤.
    문장 경계는 평가기와 같은 prompt_ladder.Splitter.BOUND 로 찾는다.

사용법
    cd ~/이윤우 && source .venv/bin/activate
    python glossary_sentence.py                 # 생성 → 평가(NLI, GPU 0) → 비교 리포트
    python glossary_sentence.py --report-only   # 이미 만든 결과로 비교 리포트만
    python glossary_sentence.py --gpu 0         # NLI 에 쓸 GPU (1번은 대여 중이므로 0)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DEFAULT_HOME = os.path.expanduser("~/이윤우")


def _bootstrap(home: str) -> None:
    """프로젝트 폴더의 기존 모듈을 import 할 수 있게 한다."""
    for p in (home, os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.insert(0, p)


# ===========================================================================
# 문장 분리형 설명 생성
# ===========================================================================

def has_batchim(ch: str) -> bool:
    """한글 음절의 받침 유무. 한글이 아니면 False."""
    code = ord(ch) - 0xAC00
    return 0 <= code < 11172 and code % 28 != 0


def make_definition_sentence(term: str, explanation: str) -> str:
    """'출혈' + '피가 혈관 밖으로 나오는 것' → '출혈은 피가 혈관 밖으로 나오는 것입니다.'"""
    josa = "은" if has_batchim(term[-1]) else "는"
    expl = explanation.strip().rstrip(".")
    return f"{term}{josa} {expl}입니다."


def find_insertion_point(text: str, pos: int, bound_re) -> Tuple[int, str]:
    """pos 이후 첫 문장 경계를 찾아 (삽입 위치, 구분자)를 돌려준다.

    평가기와 같은 경계 규칙(Splitter.BOUND)을 쓴다.
        '다. '  → 마침표 바로 뒤에 ' ' + 설명문
        '다.\\n' → 마침표 바로 뒤에 '\\n' + 설명문
        '\\n'    → 줄바꿈 앞에 '\\n' + 설명문 (마침표 없는 목록 항목)
        경계 없음(마지막 문장) → 본문 끝(후행 공백 앞)에 '\\n' + 설명문
    """
    m = bound_re.search(text, pos)
    if not m:
        return len(text.rstrip()), "\n"
    tok = m.group(0)
    sep = "\n" if "\n" in tok else " "
    ip = m.start() + 1 if tok.startswith(".") else m.start()
    return ip, sep


def build_sentence_E(A: str, plan: List[dict], bound_re) -> Tuple[str, List[dict]]:
    """plan = [{'term','explanation','start','end'}, ...] (A 기준 위치)

    같은 문장에 용어가 여러 개면 등장 순서대로 그 문장 뒤에 이어 붙인다.
    뒤쪽 삽입 위치부터 처리해 앞쪽 인덱스가 밀리지 않게 한다."""
    groups: Dict[int, List[dict]] = {}
    seps: Dict[int, str] = {}
    for p in sorted(plan, key=lambda x: x["start"]):
        ip, sep = find_insertion_point(A, p["end"], bound_re)
        groups.setdefault(ip, []).append(p)
        seps[ip] = "\n" if (seps.get(ip) == "\n" or sep == "\n") else sep

    E = A
    placed: List[dict] = []
    for ip in sorted(groups, reverse=True):
        sep = seps[ip]
        chunk = ""
        for p in groups[ip]:
            sent = make_definition_sentence(p["term"], p["explanation"])
            chunk += sep + sent
            placed.append({**p, "inserted_sentence": sent,
                           "insertion_point_in_A": ip, "separator": repr(sep),
                           "sep_raw": sep})
        E = E[:ip] + chunk + E[ip:]
    placed.sort(key=lambda x: x["start"])
    return E, placed


def validate_sentence_E(A: str, E: str, placed: List[dict], splitter,
                        inside_parenthesis) -> List[dict]:
    """문장 분리형 삽입 검증. 하나라도 걸리면 incorrect insertion 으로 센다."""
    problems: List[dict] = []
    se = set(splitter.split(E))
    for p in placed:
        s, t = p["inserted_sentence"], p["term"]
        n = E.count(s)
        if n == 0:
            problems.append({"term": t, "kind": "insertion_missing"})
            continue
        if n > 1:
            problems.append({"term": t, "kind": "duplicate_insertion", "count": n})
        i = E.find(s)
        if inside_parenthesis(E, i):
            problems.append({"term": t, "kind": "nested_parenthesis",
                             "context": E[max(0, i - 20):i + 50]})
        # Splitter 는 '다. ' 경계의 마침표를 떼어 내므로 마침표 유무 둘 다 허용
        if s not in se and s.rstrip(".") not in se:
            problems.append({"term": t, "kind": "sentence_not_isolated",
                             "context": E[max(0, i - 40):i + len(s) + 10]})
        if re.search(r"\d", s):
            problems.append({"term": t, "kind": "digit_in_definition"})

    # 삽입문만 제거하면 A 와 정확히 같아야 한다 (A 의 글자를 하나도 바꾸지 않았는지)
    R = E
    for p in placed:
        R = R.replace(p["sep_raw"] + p["inserted_sentence"], "", 1)
    if R != A:
        problems.append({"term": "*", "kind": "a_text_altered"})

    # A 의 모든 문장이 E 에 그대로 남아 있어야 한다
    lost = [x for x in splitter.split(A) if x not in se]
    if lost:
        problems.append({"term": "*", "kind": "a_sentence_altered",
                         "count": len(lost), "context": lost[0][:80]})
    return problems


# ===========================================================================
# 1단계 — E_sent 생성
# ===========================================================================

def run_apply(a, home: str) -> None:
    import apply_glossary as AG
    from prompt_ladder import Splitter

    gpath = os.path.join(home, a.glossary)
    if not os.path.exists(gpath):
        sys.exit(f"[FATAL] {gpath} 없음")
    g, enabled = AG.load_glossary(gpath)
    if not enabled:
        sys.exit("[FATAL] 삽입 가능한 glossary 항목이 없습니다.")

    docdir = os.path.join(home, "docs")
    apat = os.path.join(home, a.adir, "{doc}__P0.json")
    inline_dir = os.path.join(home, a.inline_edir)
    outdir = os.path.join(home, a.outdir, "E")
    os.makedirs(outdir, exist_ok=True)

    docs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(docdir, "*.txt"))
                  if not os.path.basename(p).startswith("syn"))

    print("=" * 100)
    print("1단계 — 문장 분리형 E 생성 (새 LLM generation 아님, A 의 후처리)")
    print("=" * 100)
    print(f"  glossary   {g.get('glossary_version')}  삽입 가능 {len(enabled)}개")
    print(f"  A          {apat}")
    print(f"  출력       {outdir}/")
    print(f"  기존 inline 결과({inline_dir})는 읽기만 하고 수정하지 않습니다.\n")
    print("  " + "문서".ljust(8) + "대상".rjust(5) + "삽입".rjust(5) + "오탐".rjust(5)
          + "문장A".rjust(7) + "문장E".rjust(7) + "길이".rjust(9)
          + "  inline대상일치")

    n_same = n_cmp = 0
    for d in docs:
        K, A = AG.load_K(d, docdir), AG.load_A(d, apat)
        if not K or not A:
            print("  " + d.ljust(8) + "  입력 없음 — 건너뜀")
            continue

        # 삽입 대상 판정은 inline 과 완전히 같은 함수로 한다
        base = AG.apply(d, K, A, enabled, "none")
        plan = [{"term": i["term"], "explanation": i["explanation"],
                 "start": i["position_in_A"],
                 "end": i["position_in_A"] + len(i["term"])}
                for i in base["insertions"]]

        E, placed = build_sentence_E(A, plan, Splitter.BOUND)
        problems = validate_sentence_E(A, E, placed, Splitter,
                                       AG.inside_parenthesis)

        meta = {i["term"]: i for i in base["insertions"]}
        insertions = []
        for p in placed:
            m = meta[p["term"]]
            insertions.append({**{k: v for k, v in m.items() if k != "render"},
                               "render": "sentence",
                               "inserted_sentence": p["inserted_sentence"],
                               "insertion_point_in_A": p["insertion_point_in_A"],
                               "separator": p["separator"]})

        # inline 결과와 삽입 대상(용어 집합)이 같은지 확인
        same = "—"
        ip = os.path.join(inline_dir, d + ".json")
        if os.path.exists(ip):
            with open(ip, encoding="utf-8") as f:
                inl = json.load(f)
            t_inl = sorted(x["term"] for x in inl.get("insertions", []))
            t_sen = sorted(x["term"] for x in insertions)
            ok = (t_inl == t_sen and inl.get("A_output") == A)
            same = "일치" if ok else "불일치"
            n_cmp += 1
            n_same += int(ok)

        sa, se = Splitter.split(A), Splitter.split(E)
        rec = {
            "doc": d, "route": "E_sent", "render": "sentence",
            "source_generation": "outputs_prompt/" + d + "__P0.json",
            "generation_note": ("E_sent 는 새 LLM generation 이 아니다. A 의 문장은 "
                                "그대로 두고, 용어가 처음 나온 문장 뒤에 설명 문장을 "
                                "추가한 후처리이다."),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "A_output": A, "E_output": E,
            "len_A": len(A), "len_E": len(E),
            "n_insertions": len(insertions), "insertions": insertions,
            "n_skipped": base["n_skipped"], "skipped": base["skipped"],
            "incorrect_insertions": problems,
            "sentence_diagnostic": {"n_sentences_A": len(sa), "n_sentences_E": len(se),
                                    "added_definition_sentences": len(insertions)},
            "eligible_occurrences": base["eligible_occurrences"],
            "glossary_version": g.get("glossary_version"),
            "glossary_entries_enabled": [
                {"term": e["term"], "source_name": AG._entry_source(e),
                 "source_url": e.get("source_url"), "source_type": e.get("source_type"),
                 "source_scope": e.get("source_scope"),
                 "version_added": e.get("version_added"),
                 "source_verified": True, "explanation_verified": True}
                for e in enabled],
            "nature": g.get("nature"),
        }
        with open(os.path.join(outdir, d + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        pct = 100 * (len(E) - len(A)) / max(len(A), 1)
        print("  " + d.ljust(8) + str(base["eligible_occurrences"]).rjust(5)
              + str(len(insertions)).rjust(5) + str(len(problems)).rjust(5)
              + str(len(sa)).rjust(7) + str(len(se)).rjust(7)
              + ("%+8.1f%%" % pct) + "  " + same, flush=True)
        for q in problems:
            print("      [오탐] " + q["kind"] + " " + str(q.get("term"))
                  + "  " + str(q.get("context", ""))[:70])

    print(f"\n  inline 과 삽입 대상이 같은 문서: {n_same}/{n_cmp}")
    if n_cmp and n_same != n_cmp:
        print("  [경고] 대상이 다른 문서가 있습니다. 비교의 독립 변수가 하나가 아닙니다.")


# ===========================================================================
# 2단계 — 기존 평가기로 평가
# ===========================================================================

def run_eval(a, home: str) -> None:
    cmd = [sys.executable, "eval_A_vs_E.py",
           "--edir", os.path.join(a.outdir, "E"),
           "--glossary", a.glossary,
           "--gpu", str(a.gpu),
           "--out", a.eval_out, "--csv", a.csv]
    if os.path.exists(os.path.join(home, a.variant_review)):
        cmd += ["--variant-review", a.variant_review]
    else:
        print(f"[주의] {a.variant_review} 없음 — integrity 주지표가 conservative 로 바뀝니다.")
    print("\n" + "=" * 100)
    print("2단계 — 기존 eval_A_vs_E.py 로 평가 (평가 코드는 수정하지 않음)")
    print("=" * 100)
    print("$ " + " ".join(cmd) + "\n", flush=True)
    if subprocess.call(cmd, cwd=home) != 0:
        sys.exit("[FATAL] eval_A_vs_E.py 실패")


# ===========================================================================
# 3단계 — inline vs sentence 비교 리포트
# ===========================================================================

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def _f(v, d=2, w=11, sign=False):
    if v is None:
        return "—".rjust(w)
    return (("%+" if sign else "%") + f"{w}.{d}f") % v


def text_metrics(text: str, ko2025_index, mid_vocab_ratio, splitter) -> dict:
    ss = splitter.split(text)
    wps = (sum(len(x.split()) for x in ss) / len(ss)) if ss else None
    out = {"wps": wps, "n_sent": len(ss)}
    for pol in ("A", "B"):
        mv = mid_vocab_ratio(text, pol)
        out["mv_" + pol] = mv[0] if mv else None
        out["ko_" + pol] = ko2025_index(text, pol)
    return out


def run_report(a, home: str) -> None:
    from prompt_ladder import Splitter
    from simplicity_metrics import ko2025_index, mid_vocab_ratio, ko2025_level, KO2025

    p_inl = os.path.join(home, a.inline_eval)
    p_sen = os.path.join(home, a.eval_out)
    for p in (p_inl, p_sen):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] {p} 없음")
    with open(p_inl, encoding="utf-8") as f:
        I = json.load(f)
    with open(p_sen, encoding="utf-8") as f:
        S = json.load(f)
    si, ss = I["summary"], S["summary"]
    ti, ts = I.get("terminology_integrity") or {}, S.get("terminology_integrity") or {}

    # ── 텍스트에서 직접 재계산 (policy A/B 병기, 성분 분해) ──────────────
    idir = os.path.join(home, a.inline_edir)
    sdir = os.path.join(home, a.outdir, "E")
    per = []
    for sp in sorted(glob.glob(os.path.join(sdir, "*.json"))):
        d = os.path.splitext(os.path.basename(sp))[0]
        ip = os.path.join(idir, d + ".json")
        if not os.path.exists(ip):
            continue
        with open(sp, encoding="utf-8") as f:
            rs = json.load(f)
        with open(ip, encoding="utf-8") as f:
            ri = json.load(f)
        mA = text_metrics(rs["A_output"], ko2025_index, mid_vocab_ratio, Splitter)
        mI = text_metrics(ri["E_output"], ko2025_index, mid_vocab_ratio, Splitter)
        mS = text_metrics(rs["E_output"], ko2025_index, mid_vocab_ratio, Splitter)
        per.append({"doc": d, "A": mA, "I": mI, "S": mS})

    cv, cw = KO2025["coef_mid_vocab"], KO2025["coef_words_per_sent"]

    def decomp(key, pol="A"):
        dv = [cv * (r[key]["mv_" + pol] - r["A"]["mv_" + pol]) for r in per]
        dw = [cw * (r[key]["wps"] - r["A"]["wps"]) for r in per]
        dk = [r[key]["ko_" + pol] - r["A"]["ko_" + pol] for r in per]
        return st.fmean(dk), st.fmean(dv), st.fmean(dw)

    rows_i = {r["doc"]: r for r in I.get("rows", [])}
    rows_s = {r["doc"]: r for r in S.get("rows", [])}

    W = 38
    def row(label, vi, vs, dl=""):
        print("  " + label.ljust(W) + str(vi).rjust(14) + str(vs).rjust(14)
              + str(dl).rjust(14))

    print("\n" + "=" * 100)
    print("3단계 — 괄호형(inline) vs 문장 분리형(sentence) 비교")
    print("=" * 100)
    print(f"  문서 {len(per)}건 · glossary {ss.get('entries_enabled')}개 · "
          "A(기존 HARI 출력)는 두 방식이 동일")
    print("  " + "지표".ljust(W) + "괄호형".rjust(14) + "문장 분리형".rjust(14)
          + "차이(문장-괄호)".rjust(14))
    print("  " + "-" * (W + 42))

    print("\n  [Glossary 적용 — 두 방식이 같아야 정상]")
    row("target occurrence", si.get("target_occurrences"), ss.get("target_occurrences"))
    row("실제 삽입", si.get("insertions"), ss.get("insertions"))
    row("coverage (target-level)", _f(si.get("coverage_pct_target_level"), 1, 0) + "%",
        _f(ss.get("coverage_pct_target_level"), 1, 0) + "%")
    row("incorrect insertion", si.get("incorrect_insertions"),
        ss.get("incorrect_insertions"))

    print("\n  [Safety]")
    row("Terminology integrity [human]", _f(ti.get("human_adjudicated"), 1, 0) + "%",
        _f(ts.get("human_adjudicated"), 1, 0) + "%")
    row("Grounding A (공통)", _f(si.get("grounding_A"), 2, 0), _f(ss.get("grounding_A"), 2, 0))
    row("Grounding E", _f(si.get("grounding_E_all"), 2, 0),
        _f(ss.get("grounding_E_all"), 2, 0),
        _f((ss.get("grounding_E_all") or 0) - (si.get("grounding_E_all") or 0), 2, 0, True))
    row("  └ Grounding cost (E − A)", _f(si.get("delta_grounding"), 2, 0, True),
        _f(ss.get("delta_grounding"), 2, 0, True))
    gate_i = (si.get("grounding_E_all") or 0) >= 70
    gate_s = (ss.get("grounding_E_all") or 0) >= 70
    row("  └ Safety Gate Grounding ≥ 70", "통과" if gate_i else "미달",
        "통과" if gate_s else "미달")
    n_digit = 0
    for sp in glob.glob(os.path.join(home, a.outdir, "E", "*.json")):
        with open(sp, encoding="utf-8") as f:
            n_digit += sum(1 for q in json.load(f).get("incorrect_insertions", [])
                           if q.get("kind") == "digit_in_definition")
    row("설명문 속 숫자 (수치 보존 영향)", "0건", f"{n_digit}건")

    print("\n  [Simplicity — Ko2025, 값이 높을수록 쉬움]")
    kA = _mean([r["A"]["ko_A"] for r in per])
    kI = _mean([r["I"]["ko_A"] for r in per])
    kS = _mean([r["S"]["ko_A"] for r in per])
    row("Ko2025 A (기준선, policy A)", _f(kA, 2, 0), _f(kA, 2, 0))
    row("Ko2025 E (policy A)", _f(kI, 2, 0), _f(kS, 2, 0), _f(kS - kI, 2, 0, True))
    row("  └ 수준", str(ko2025_level(kI)), str(ko2025_level(kS)))
    kAB = _mean([r["A"]["ko_B"] for r in per])
    kIB = _mean([r["I"]["ko_B"] for r in per])
    kSB = _mean([r["S"]["ko_B"] for r in per])
    row("Ko2025 A (기준선, policy B)", _f(kAB, 2, 0), _f(kAB, 2, 0))
    row("Ko2025 E (policy B)", _f(kIB, 2, 0), _f(kSB, 2, 0), _f(kSB - kIB, 2, 0, True))
    row("평균 어절/문장 A", _f(_mean([r["A"]["wps"] for r in per]), 2, 0),
        _f(_mean([r["A"]["wps"] for r in per]), 2, 0))
    wI = _mean([r["I"]["wps"] for r in per])
    wS = _mean([r["S"]["wps"] for r in per])
    row("평균 어절/문장 E", _f(wI, 2, 0), _f(wS, 2, 0), _f(wS - wI, 2, 0, True))
    mvI = _mean([r["I"]["mv_A"] for r in per])
    mvS = _mean([r["S"]["mv_A"] for r in per])
    row("중급이상 어휘비율 E % (A)", _f(mvI, 2, 0), _f(mvS, 2, 0), _f(mvS - mvI, 2, 0, True))
    row("문장 수 E (평균)", _f(_mean([r["I"]["n_sent"] for r in per]), 1, 0),
        _f(_mean([r["S"]["n_sent"] for r in per]), 1, 0))
    row("길이 증가", _f(si.get("length_increase_pct"), 2, 0, True) + "%",
        _f(ss.get("length_increase_pct"), 2, 0, True) + "%")

    print("\n  [성분 분해 — ΔKo2025 = 어휘항 + 어절항  (A 대비, 문서 평균, policy A)]")
    for name, key in (("괄호형", "I"), ("문장 분리형", "S")):
        dk, dv, dw = decomp(key, "A")
        print(f"    {name:<8} ΔKo2025 {dk:+.2f} = 어휘항 {dv:+.2f} + 어절항 {dw:+.2f}")
    for name, key in (("괄호형", "I"), ("문장 분리형", "S")):
        dk, dv, dw = decomp(key, "B")
        print(f"    {name:<8} (policy B) ΔKo2025 {dk:+.2f} = 어휘항 {dv:+.2f} + 어절항 {dw:+.2f}")

    # 저장값과 재계산값 일치 확인 (계산 경로가 같은지)
    chk = []
    for r in per:
        if r["doc"] in rows_s and rows_s[r["doc"]].get("ko2025_E") is not None:
            chk.append(abs(rows_s[r["doc"]]["ko2025_E"] - r["S"]["ko_A"]))
    if chk:
        print(f"\n  [검증] eval 저장값 vs 텍스트 재계산 Ko2025 최대 오차: {max(chk):.4f}")

    print("\n  [문서별]")
    print("    " + "문서".ljust(7) + "ΔKo 괄호".rjust(10) + "ΔKo 문장".rjust(10)
          + "ΔGround 괄호".rjust(13) + "ΔGround 문장".rjust(13)
          + "어절E 괄호".rjust(11) + "어절E 문장".rjust(11))
    better = 0
    for r in per:
        d = r["doc"]
        di = r["I"]["ko_A"] - r["A"]["ko_A"]
        ds = r["S"]["ko_A"] - r["A"]["ko_A"]
        better += int(ds > di)
        gi = (rows_i.get(d) or {}).get("delta_ground")
        gs = (rows_s.get(d) or {}).get("delta_ground")
        print("    " + d.ljust(7) + _f(di, 2, 10, True) + _f(ds, 2, 10, True)
              + _f(gi, 2, 13, True) + _f(gs, 2, 13, True)
              + _f(r["I"]["wps"], 2, 11) + _f(r["S"]["wps"], 2, 11))
    print(f"\n  문장 분리형이 괄호형보다 Ko2025 가 높은 문서: {better}/{len(per)}")

    # ── 삽입 예시 3건 (자연스러운지 눈으로 확인) ──────────────────────
    print("\n  [문장 분리형 삽입 예시 — 자연스러운지 확인용]")
    shown = 0
    for sp in sorted(glob.glob(os.path.join(sdir, "*.json"))):
        with open(sp, encoding="utf-8") as f:
            rs = json.load(f)
        for ins in rs.get("insertions", [])[:1]:
            E = rs["E_output"]
            i = E.find(ins["inserted_sentence"])
            ctx = E[max(0, i - 60):i + len(ins["inserted_sentence"])].replace("\n", " ⏎ ")
            print(f"    [{rs['doc']}] …{ctx}")
            shown += 1
        if shown >= 3:
            break

    out = {
        "created": datetime.now(timezone.utc).isoformat(),
        "design": "independent variable = explanation form only (inline vs sentence)",
        "inline_eval": a.inline_eval, "sentence_eval": a.eval_out,
        "ko2025_policyA": {"A": kA, "inline": kI, "sentence": kS},
        "ko2025_policyB": {"A": kAB, "inline": kIB, "sentence": kSB},
        "wps": {"inline": wI, "sentence": wS},
        "decomposition_policyA": {
            "inline": dict(zip(("d_ko", "vocab_term", "eojeol_term"), decomp("I", "A"))),
            "sentence": dict(zip(("d_ko", "vocab_term", "eojeol_term"), decomp("S", "A")))},
        "grounding": {"A": ss.get("grounding_A"),
                      "inline_E": si.get("grounding_E_all"),
                      "sentence_E": ss.get("grounding_E_all")},
        "safety_gate_grounding70": {"inline": gate_i, "sentence": gate_s},
        "docs_sentence_better_ko2025": better, "n_docs": len(per),
        "per_doc": per,
    }
    op = os.path.join(home, "glossary_sentence_compare.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] " + op)
    print("""
  해석 시 주의
    · 두 방식의 차이는 '설명 형태' 하나뿐입니다. 삽입 대상·glossary·A 는 같습니다.
    · Ko2025 가 오르고 Grounding 이 내려가면 trade-off 입니다. 채택은 자동으로 정하지 않습니다.
    · Grounding 하락은 '원문에 없는 문장(설명문)이 추가된 것'을 NLI 가 미근거로 본 결과일 수
      있습니다. 환각과 구분해서 보고하십시오.
    · 실제 L2 환자 이해도를 측정한 것이 아닙니다. 13문서 밖으로 일반화하지 않습니다.""")
    print("=" * 100)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=DEFAULT_HOME)
    ap.add_argument("--glossary", default="glossary_v1_1.json")
    ap.add_argument("--adir", default="outputs_prompt")
    ap.add_argument("--inline-edir", default=os.path.join("outputs_glossary_v11", "E"))
    ap.add_argument("--inline-eval", default="eval_A_vs_E_v11.json")
    ap.add_argument("--outdir", default="outputs_glossary_v11_sent")
    ap.add_argument("--eval-out", default="eval_A_vs_E_v11_sent.json")
    ap.add_argument("--csv", default="eval_A_vs_E_v11_sent_per_doc.csv")
    ap.add_argument("--variant-review", default="variant_review.csv")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    home = os.path.abspath(os.path.expanduser(a.home))
    _bootstrap(home)
    os.chdir(home)

    if not a.report_only:
        run_apply(a, home)
        run_eval(a, home)
    run_report(a, home)


if __name__ == "__main__":
    main()
