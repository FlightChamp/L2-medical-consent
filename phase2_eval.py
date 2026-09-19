#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase2_eval.py — 2x2 평가 (Identity 포함 5 route)
==================================================
평가 대상:
    Identity   원문 그대로 (control, evaluator ceiling 기준)
    A  P0 / OFF   outputs_prompt/{doc}__P0.json      (기존 재사용)
    B  P0 / ON    outputs_2x2/B/{doc}.json
    C  P4 / OFF   outputs_2x2/C/{doc}.json
    D  P4 / ON    outputs_2x2/D/{doc}.json

보고 원칙:
    · Safety 와 Simplicity 를 분리해 보고한다. 단일 총점을 만들지 않는다.
    · Safety Gate 를 먼저 적용하고, 통과 후보 중에서만 Simplicity 를 비교한다.
    · Grounding 은 문서별 evaluator ceiling(= Identity 값) 대비로도 표시한다.
      identity_audit.json 이 있으면 그 값을 쓴다.
    · Protected 조건에서 bijection audit 에 실패해 fail-closed 된 문서는
      평이화 효과가 0 이므로 별도 집계한다.

Gate 기준 — 실험 전에 고정. 결과를 보고 바꾸지 않는다.
    수치보존 >= 90 / 수치환각 <= 0건 / Grounding >= 70

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python phase2_eval.py --conds A B
    python phase2_eval.py                 # Identity + A B C D 중 있는 것 전부
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import (
        extract_sections, Splitter, build_chunks, extract_terms, numeric_units,
    )
    from translate_verify2 import NLI
    from simplicity_metrics import simplicity, SPEC, arrow, unresolved
except ImportError as e:
    sys.exit(f"[FATAL] 상위 폴더 스크립트 필요: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
GATE = {"num_keep_min": 90.0, "n_halluc_max": 0, "ground_min": 70.0}

ROUTES = [
    ("Identity", None),
    ("A P0/OFF", os.path.join(HOME, "outputs_prompt", "{doc}__P0.json")),
    ("B P0/ON", os.path.join(HOME, "outputs_2x2", "B", "{doc}.json")),
    ("C P4/OFF", os.path.join(HOME, "outputs_2x2", "C", "{doc}.json")),
    ("D P4/ON", os.path.join(HOME, "outputs_2x2", "D", "{doc}.json")),
]
KEY = {"Identity": "Identity", "A P0/OFF": "A", "B P0/ON": "B",
       "C P4/OFF": "C", "D P4/ON": "D"}

BLANK_MARK = re.compile(r"\[미기재\]|\[\s*\]|\[시간\]|\[수술\s*이름\]")
ORIG_BLANK = re.compile(
    r"[（(\[〔]\s*[)）\]〕]|[_＿]{2,}"
    r"|약\s+(?=정도|가량|이내|이상|이하|동안|쯤)")


def _nows(t) -> str:
    return re.sub(r"\s+", "", str(t))


def numeric_keep_ws(src: str, out: str):
    """공백을 무시하고 비교한다. legacy 정의는 '1 시간'을 '1시간'으로
    추출한 뒤 원문에서 찾아 불일치가 났다."""
    nums = sorted(set(numeric_units(src)))
    if not nums:
        return None, 0, []
    o = _nows(out)
    miss = [x for x in nums if _nows(x) not in o]
    return round(100 * (len(nums) - len(miss)) / len(nums), 1), len(nums), miss[:6]


def numeric_halluc_ws(src: str, out: str):
    s_ = {_nows(x) for x in numeric_units(src)}
    return sorted({_nows(x) for x in numeric_units(out)} - s_)


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_route(pattern: Optional[str], doc: str, K: str) -> Optional[dict]:
    if pattern is None:
        return {"doc": doc, "src": K, "out": K, "audit_ok": True,
                "fail_closed_used_source": False}
    p = pattern.format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_ceiling() -> Dict[str, float]:
    p = os.path.join(HOME, "identity_audit.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("ceiling_by_doc", {})
    except Exception:
        return {}


def eval_doc(name: str, rec: dict, K: str, nli: NLI, a) -> dict:
    out = rec["out"]
    terms = extract_terms(K)
    ss, oo = Splitter.split(K), Splitter.split(out)
    nk, n_nums, miss = numeric_keep_ws(K, out)
    hal = numeric_halluc_ws(K, out)
    nb_src = len(ORIG_BLANK.findall(K))
    nb_out = len(BLANK_MARK.findall(out)) + len(ORIG_BLANK.findall(out))

    return {
        "route": name, "doc": rec.get("doc"),
        "audit_ok": rec.get("audit_ok", True),
        "fail_closed": rec.get("fail_closed_used_source", False),
        "n_placeholder": rec.get("n_placeholder", 0),
        "safety": {
            "ground": round(nli.rate(oo, build_chunks(ss), a.k, a.tau), 1)
                      if oo and ss else None,
            "cover": round(nli.rate(ss, build_chunks(oo), a.k, a.tau), 1)
                     if oo and ss else None,
            "term_keep": round(100 * sum(1 for t in terms if t in out)
                               / max(len(terms), 1), 1),
            "num_keep": nk, "n_nums": n_nums, "num_miss": miss,
            "n_halluc": len(hal), "halluc": hal[:6],
            "blank_src": nb_src, "blank_out": nb_out,
            "blank_keep": (round(100 * min(nb_out, nb_src) / nb_src, 1)
                           if nb_src else None),
        },
        "simplicity": simplicity(out, K),
    }


def gate_pass(s: dict) -> Tuple[bool, List[str]]:
    f = []
    if s.get("num_keep") is not None and s["num_keep"] < GATE["num_keep_min"]:
        f.append(f"수치보존 {s['num_keep']:.1f}")
    if s.get("n_halluc", 0) > GATE["n_halluc_max"]:
        f.append(f"수치환각 {s['n_halluc']}건")
    if s.get("ground") is not None and s["ground"] < GATE["ground_min"]:
        f.append(f"Grounding {s['ground']:.1f}")
    return (not f), f


# ===========================================================================

def report(rows: List[dict], routes: List[str], ceiling: Dict[str, float]):
    def avg(r, grp, k):
        v = [x[grp].get(k) for x in rows
             if x["route"] == r and x[grp].get(k) is not None]
        return st.fmean(v) if v else None

    def tot(r, grp, k):
        return sum(x[grp].get(k) or 0 for x in rows if x["route"] == r)

    def sel(r):
        return [x for x in rows if x["route"] == r]

    def f(v, w=13, d=2):
        return f"{'—':>{w}}" if v is None else f"{v:>{w}.{d}f}"

    W = 13
    hdr = f"  {'지표':<24}" + "".join(f"{r:>{W}}" for r in routes)

    # ── audit 상태 ──────────────────────────────────────────────────
    prot = [r for r in routes if r.endswith("/ON")]
    if prot:
        print("\n" + "=" * 96)
        print("0. bijection audit — Protected 조건")
        print("=" * 96)
        print(f"  {'Route':<12}{'문서':>6}{'통과':>6}{'fail-closed':>13}"
              f"{'placeholder 합':>16}")
        for r in prot:
            s = sel(r)
            fc = sum(1 for x in s if x["fail_closed"])
            print(f"  {r:<12}{len(s):>6}{len(s)-fc:>6}{fc:>13}"
                  f"{sum(x['n_placeholder'] for x in s):>16}")
        if any(x["fail_closed"] for x in rows):
            print("\n  fail-closed 문서는 원문을 그대로 두었으므로 "
                  "평이화 효과가 0 이고 Safety 는 Identity 와 같습니다.")

    # ── Safety ─────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("1. Safety")
    print("=" * 96)
    print(hdr)
    for label, key, d in [("Grounding ↑", "ground", 2),
                          ("Coverage ↑", "cover", 2),
                          ("용어 보존 ↑", "term_keep", 2),
                          ("수치 보존 ↑", "num_keep", 2),
                          ("빈칸 보존 ↑", "blank_keep", 1)]:
        print(f"  {label:<24}" + "".join(f(avg(r, "safety", key), W, d)
                                         for r in routes))
    print(f"  {'수치 환각 ↓ (총건)':<24}"
          + "".join(f"{tot(r,'safety','n_halluc'):>{W}}" for r in routes))
    nb = [x["safety"]["blank_src"] for x in rows if x["route"] == routes[0]]
    print(f"\n  빈칸 분모: 13문서 합계 {sum(nb)}곳 "
          f"({sum(1 for v in nb if v)}개 문서에만 존재)")
    print("  → 표본이 작아 빈칸 보존율은 단독 근거로 쓰지 않습니다.")

    # ── ceiling 대비 ───────────────────────────────────────────────
    if ceiling:
        print("\n" + "=" * 96)
        print("1-b. evaluator ceiling 대비 Grounding")
        print("=" * 96)
        print("  ceiling = 해당 문서에서 Identity(원문=원문)의 Grounding.")
        print("  원문 그대로도 100 이 되지 않으므로 이 값이 측정 상한입니다.")
        print(f"\n  {'문서':<8}{'ceiling':>9}"
              + "".join(f"{r:>{W}}" for r in routes[1:]))
        for doc in sorted({x["doc"] for x in rows if x["doc"]}):
            c = ceiling.get(doc)
            line = f"  {doc:<8}{(c if c else float('nan')):>9.1f}"
            for r in routes[1:]:
                v = next((x["safety"]["ground"] for x in rows
                          if x["route"] == r and x["doc"] == doc), None)
                line += (f"{'—':>{W}}" if v is None or not c
                         else f"{v/c:>{W}.2f}")
            print(line)
        print("\n  1.00 초과는 원문보다 충실하다는 뜻이 아니라, 추출로 뭉친 "
              "장문이 짧은 문장으로 나뉘어\n  NLI 가 판정 가능해졌다는 뜻입니다.")

    # ── Simplicity ─────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("2. Simplicity")
    print("=" * 96)
    print(hdr)
    order = ["ko2025_A", "ko2025_B", "mid_vocab_A", "mid_vocab_B",
             "words_per_sent", "chars_per_sent", "long_sent_ratio",
             "sent_split_ratio", "n_sentences", "length_change",
             "term_explain_rate", "hanja_ratio", "copy_similarity"]
    for k in order + [x for x in SPEC if x not in order]:
        if k not in SPEC:
            continue
        vals = [avg(r, "simplicity", k) for r in routes]
        if all(v is None for v in vals):
            continue
        print(f"  {arrow(k)} {k:<22}" + "".join(f(v, W, 2) for v in vals))

    # ── L2 수준 환산 ───────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("2-b. 고승연(2025) L2 이독성 수준 환산")
    print("=" * 96)
    print("  ≥45 쉬움 / 35–44.99 보통 / <35 어려움   (값이 높을수록 쉬움)")
    try:
        from simplicity_metrics import ko2025_level
    except ImportError:
        ko2025_level = lambda v: None
    print(f"\n  {'Route':<12}{'policy A':>11}{'수준':>8}"
          f"{'policy B':>11}{'수준':>8}")
    for r in routes:
        va, vb = avg(r, "simplicity", "ko2025_A"), avg(r, "simplicity", "ko2025_B")
        la = ko2025_level(va) or "—"
        lb = ko2025_level(vb) or "—"
        print(f"  {r:<12}{f(va,11,2)}{la:>8}{f(vb,11,2)}{lb:>8}")
    print("""
  미등재 어휘 처리 방식이 논문에 없어 두 정책을 병기합니다.
    policy A  미등재를 중급 이상으로 계산
    policy B  미등재를 분모에서 제외
  어느 한쪽만 인용하지 않습니다.""")

    # ── 이득 분해 ──────────────────────────────────────────────────
    base = "Identity"
    if base in routes:
        print("\n" + "=" * 96)
        print("2-c. L2 이독성 이득 분해 (policy A, Identity 대비)")
        print("=" * 96)
        print("  지수 = 61.994 − 0.261×중급이상어휘비율 − 1.045×평균어절수")
        b_mv = avg(base, "simplicity", "mid_vocab_A")
        b_wp = avg(base, "simplicity", "words_per_sent")
        b_ix = avg(base, "simplicity", "ko2025_A")
        print(f"\n  {'Route':<12}{'총 이득':>10}{'어휘 기여':>11}"
              f"{'문장길이 기여':>14}{'문장길이 몫':>12}")
        for r in routes[1:]:
            mv, wp, ix = (avg(r, "simplicity", "mid_vocab_A"),
                          avg(r, "simplicity", "words_per_sent"),
                          avg(r, "simplicity", "ko2025_A"))
            if None in (mv, wp, ix, b_mv, b_wp, b_ix):
                continue
            g_v = -0.261 * (mv - b_mv)
            g_s = -1.045 * (wp - b_wp)
            share = 100 * g_s / (g_v + g_s) if (g_v + g_s) else float("nan")
            print(f"  {r:<12}{ix-b_ix:>+10.2f}{g_v:>+11.2f}"
                  f"{g_s:>+14.2f}{share:>11.0f}%")
        print("\n  문장길이 몫이 크면 '문장은 짧아졌지만 어휘는 그대로'라는 뜻입니다.")

    # ── Gate ───────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("3. Safety Gate  (실험 전 고정)")
    print("=" * 96)
    print(f"  수치보존 ≥ {GATE['num_keep_min']} / "
          f"수치환각 ≤ {GATE['n_halluc_max']}건 / "
          f"Grounding ≥ {GATE['ground_min']}")
    print(f"\n  {'Route':<12}{'문서':>6}{'통과':>6}{'탈락':>6}  주요 탈락 사유")
    for r in routes:
        s = sel(r)
        ok = sum(1 for x in s if gate_pass(x["safety"])[0])
        why: Dict[str, int] = {}
        for x in s:
            for w in gate_pass(x["safety"])[1]:
                kk = w.split()[0]
                why[kk] = why.get(kk, 0) + 1
        rs = ", ".join(f"{k} {v}건" for k, v in
                       sorted(why.items(), key=lambda z: -z[1])[:3])
        print(f"  {r:<12}{len(s):>6}{ok:>6}{len(s)-ok:>6}  {rs}")

    # ── 종합 ───────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("4. Gate 통과율 × Simplicity")
    print("=" * 96)
    print(f"  {'Route':<12}{'Gate':>8}{'수치보존':>10}{'환각':>6}"
          f"{'ko2025_A':>11}{'어절/문장':>11}{'용어설명율':>12}{'Copy':>8}")
    for r in routes:
        s = sel(r)
        ok = sum(1 for x in s if gate_pass(x["safety"])[0])
        print(f"  {r:<12}{100*ok/max(len(s),1):>7.1f}%"
              + f(avg(r, "safety", "num_keep"), 10, 1)
              + f"{tot(r,'safety','n_halluc'):>6}"
              + f(avg(r, "simplicity", "ko2025_A"), 11, 2)
              + f(avg(r, "simplicity", "words_per_sent"), 11, 2)
              + f(avg(r, "simplicity", "term_explain_rate"), 12, 1)
              + f(avg(r, "simplicity", "copy_similarity"), 8, 3))
    print("""
  읽는 법
    · Gate 를 통과한 후보 중에서만 Simplicity 를 비교합니다.
      Gate 를 못 넘으면 평이성이 좋아도 채택 대상이 아닙니다.
    · Identity 는 Simplicity 개선이 0 인데 Safety 는 최상입니다.
      Fidelity 만으로 고르면 '아무것도 하지 않기' 가 이깁니다.
    · 단일 총점을 만들지 않습니다.""")

    print("\n" + "=" * 96)
    print("5. 미해결 지표")
    print("=" * 96)
    for m in unresolved():
        print(f"  · {m}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=None,
                    choices=["A", "B", "C", "D"])
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="phase2_metrics.json")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))

    want = set(a.conds) if a.conds else {"A", "B", "C", "D"}
    routes = [(n, p) for n, p in ROUTES
              if n == "Identity" or KEY[n] in want]
    # 파일이 하나도 없는 조건은 제외
    live = []
    for n, p in routes:
        if p is None:
            live.append((n, p))
            continue
        if any(os.path.exists(p.format(doc=d)) for d in docs):
            live.append((n, p))
        else:
            print(f"  [건너뜀] {n} — 결과 파일 없음")
    routes = live

    print("=" * 96)
    print("Phase 2 — 2x2 Safety–Simplicity 평가")
    print("=" * 96)
    print(f"  문서 {len(docs)}건 / Route {len(routes)}종")
    for n, _ in routes:
        print(f"    {n}")

    ceiling = load_ceiling()
    if ceiling:
        print(f"  evaluator ceiling: identity_audit.json 에서 "
              f"{len(ceiling)}문서 로드")
    else:
        print("  evaluator ceiling: identity_audit.json 없음 — 1-b 절 생략")

    nli = NLI(a.gpu, a.batch)
    rows = []
    print("\n  평가 중", end="", flush=True)
    for doc in docs:
        K = load_K(doc, a.docdir)
        if not K:
            continue
        for name, pat in routes:
            rec = load_route(pat, doc, K)
            if rec is None:
                continue
            rec.setdefault("doc", doc)
            if rec.get("src") and rec["src"] != K:
                print(f"\n  [경고] {doc}/{name}: src 불일치 — 비교 신뢰도 저하")
            rows.append(eval_doc(name, rec, K, nli, a))
        print(".", end="", flush=True)
    print()

    report(rows, [n for n, _ in routes], ceiling)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"gate": GATE, "routes": [n for n, _ in routes],
                   "ceiling": ceiling, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
