#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase1_eval.py — Safety–Simplicity 재평가 (Identity baseline 포함)
====================================================================
목적:
    기존 13문서 출력을 새 Simplicity 축으로 재평가한다. 새 generation 없음.
    Identity(아무것도 바꾸지 않음)를 control 로 넣어
    "Fidelity 만 최대화하면 Identity 가 이긴다"를 표에서 드러낸다.

평가 대상:
    Identity        원문 그대로               (out = src)
    HARI P0         outputs_prompt/{doc}__P0.json
    MedGemma        outputs_compare/google_medgemma-1.5-4b-it/{doc}.json
    Medical-Llama3  experiment_medllama/outputs/ko/{doc}.json

    SciBERT 는 생성형 모델이 아니므로 제외한다.

판단 원칙 (단일 총점을 만들지 않는다):
    1단계  Safety Gate      치명적 정보 훼손이 없는 후보만 통과
    2단계  Simplicity       통과한 후보 중 가장 쉽게 만든 것을 선택

    Safety Gate 항목 (하나라도 위반하면 평이성과 무관하게 탈락 후보):
      · 수치 보존율이 기준 미만
      · 수치 환각 발생
      · 빈칸 [미기재] 훼손
      · Grounding 이 기준 미만

전처리 동일성 보장:
    모든 모델의 src 가 같은 K 인지 assert 로 확인하고, 문서별 해시를 기록한다.
    (retrieval 실험에서 전처리 불일치로 결과가 무효화된 사례를 반영)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python phase1_eval.py --check-only      # 입력 정합성만 확인
    python phase1_eval.py                   # 전체 평가
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
        extract_sections, Splitter, build_chunks, extract_terms,
        numeric_units, numeric_halluc,
    )
    from translate_verify2 import NLI
    from simplicity_metrics import simplicity, SPEC, arrow, unresolved
except ImportError as e:
    sys.exit(f"[FATAL] 상위 폴더에 prompt_ladder / translate_verify2 / "
             f"simplicity_metrics 가 필요합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")

# 평가 대상 — (표시명, 경로 패턴 또는 None=Identity, 설명)
ROUTES = [
    ("Identity", None, "원문 그대로 (control)"),
    ("HARI P0", os.path.join(HOME, "outputs_prompt", "{doc}__P0.json"),
     "snuh/hari-q3-8b, greedy"),
    ("MedGemma", os.path.join(HOME, "outputs_compare",
                              "google_medgemma-1.5-4b-it", "{doc}.json"),
     "google/medgemma-1.5-4b-it"),
    ("Medical-Llama3", os.path.join(HOME, "experiment_medllama", "outputs",
                                    "ko", "{doc}.json"),
     "ruslanmv/Medical-Llama3-8B, greedy"),
]

# Safety Gate 기준 — 실험 전에 고정한다
GATE = {
    "num_keep_min": 90.0,      # 수치 보존율 %
    "n_halluc_max": 0,         # 문서당 수치 환각 허용 건수
    "ground_min": 70.0,        # Grounding %
}
# ── 수치 비교 (공백 정규화) ──────────────────────────────────────────
# prompt_ladder.numeric_units 는 "1 시간" 을 "1시간" 으로 추출한 뒤
# 원문에서 "1시간" 을 찾아 불일치가 난다. 비교 전에 공백을 지운다.
def _nows(t: str) -> str:
    return re.sub(r"\s+", "", str(t))


def numeric_keep_ws(src: str, out: str):
    """(보존율, 분모, 누락목록). 공백을 무시하고 비교한다."""
    nums = sorted(set(numeric_units(src)))
    if not nums:
        return None, 0, []
    o = _nows(out)
    miss = [x for x in nums if _nows(x) not in o]
    return round(100 * (len(nums) - len(miss)) / len(nums), 1), len(nums), miss[:6]


def numeric_halluc_ws(src: str, out: str):
    """출력에만 있는 수치. 공백을 무시한다."""
    s_ = {_nows(x) for x in numeric_units(src)}
    return sorted({_nows(x) for x in numeric_units(out)} - s_)


BLANK_MARK = re.compile(r"\[미기재\]|\[\s*\]|\[시간\]|\[수술\s*이름\]")
ORIG_BLANK = re.compile(
    r"[（(\[〔]\s*[)）\]〕]|[_＿]{2,}"
    r"|약\s+(?=정도|가량|이내|이상|이하|동안|쯤)")


# ===========================================================================

def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_route(name: str, pattern: Optional[str], doc: str,
               K: str) -> Optional[dict]:
    if pattern is None:                      # Identity
        return {"src": K, "out": K, "model": "(identity)"}
    p = pattern.format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def h12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# ===========================================================================

def check_inputs(docs: List[str], a) -> Tuple[Dict[str, str], List[str]]:
    """모든 route 의 src 가 동일한 K 인지 확인한다."""
    print("=" * 96)
    print("0. 입력 정합성 확인 — 모든 모델이 같은 원문을 봤는가")
    print("=" * 96)
    Ks: Dict[str, str] = {}
    problems: List[str] = []
    print(f"  {'문서':<8}{'K 해시':>14}{'K 길이':>8}"
          + "".join(f"{n:>18}" for n, _, _ in ROUTES[1:]))
    for doc in docs:
        K = load_K(doc, a.docdir)
        if K is None:
            problems.append(f"{doc}: K 생성 실패")
            continue
        Ks[doc] = K
        line = f"  {doc:<8}{h12(K):>14}{len(K):>8}"
        for name, pat, _ in ROUTES[1:]:
            r = load_route(name, pat, doc, K)
            if r is None:
                line += f"{'파일없음':>18}"
                problems.append(f"{doc}/{name}: 파일 없음")
            elif r.get("src") != K:
                line += f"{'src 불일치':>18}"
                problems.append(
                    f"{doc}/{name}: src 불일치 "
                    f"({len(r.get('src',''))}자 vs K {len(K)}자)")
            else:
                line += f"{'일치':>18}"
        print(line)

    if problems:
        print(f"\n  [문제 {len(problems)}건]")
        for p in problems[:15]:
            print(f"    · {p}")
        print("\n  src 가 다르면 비교가 성립하지 않습니다. 해당 모델을 재생성하십시오.")
    else:
        print("\n  모든 route 가 동일한 K 를 사용했습니다.")
    return Ks, problems


# ===========================================================================

def eval_doc(name: str, rec: dict, K: str, nli: NLI, a) -> dict:
    src, out = K, rec["out"]
    terms = extract_terms(src)
    nums = sorted(set(numeric_units(src)))
    ss, oo = Splitter.split(src), Splitter.split(out)

    n_blank_src = len(ORIG_BLANK.findall(src))
    n_blank_out = len(BLANK_MARK.findall(out)) + len(ORIG_BLANK.findall(out))

    nk_ws, n_nums_ws, miss_ws = numeric_keep_ws(src, out)
    hal_ws = numeric_halluc_ws(src, out)

    safety = {
        "ground": round(nli.rate(oo, build_chunks(ss), a.k, a.tau), 1)
                  if oo and ss else None,
        "num_keep_ws": nk_ws,
        "n_halluc_ws": len(hal_ws),
        "halluc_ws": hal_ws[:6],
        "num_miss_ws": miss_ws,
        "cover": round(nli.rate(ss, build_chunks(oo), a.k, a.tau), 1)
                 if oo and ss else None,
        "term_keep": round(100 * sum(1 for t in terms if t in out)
                           / max(len(terms), 1), 1),
        "num_keep": round(100 * sum(1 for x in nums if x in out) / len(nums), 1)
                    if nums else None,
        "n_nums": len(nums),
        "n_halluc": len(numeric_halluc(src, out)),
        "halluc": numeric_halluc(src, out)[:6],
        "blank_src": n_blank_src,
        "blank_out": n_blank_out,
        "blank_keep": (round(100 * min(n_blank_out, n_blank_src)
                             / n_blank_src, 1) if n_blank_src else None),
    }
    return {"route": name, "doc": rec.get("doc"),
            "safety": safety, "simplicity": simplicity(out, src)}


def gate_pass(safety: dict) -> Tuple[bool, List[str]]:
    """Gate 는 공백 정규화 지표(_ws)로 판정한다. legacy 는 참고용."""
    fails = []
    nk = safety.get("num_keep_ws")
    if nk is not None and nk < GATE["num_keep_min"]:
        fails.append(f"수치보존 {nk:.1f} < {GATE['num_keep_min']}")
    if safety.get("n_halluc_ws", 0) > GATE["n_halluc_max"]:
        fails.append(f"수치환각 {safety['n_halluc_ws']}건")
    g = safety.get("ground")
    if g is not None and g < GATE["ground_min"]:
        fails.append(f"Grounding {g:.1f} < {GATE['ground_min']}")
    return (not fails), fails


# ===========================================================================

def report(rows: List[dict]):
    routes = [n for n, _, _ in ROUTES]

    def avg(route, group, key):
        v = [r[group].get(key) for r in rows
             if r["route"] == route and r[group].get(key) is not None]
        return st.fmean(v) if v else None

    def tot(route, group, key):
        return sum(r[group].get(key) or 0 for r in rows if r["route"] == route)

    def n_of(route):
        return sum(1 for r in rows if r["route"] == route)

    def f(v, w=14, d=2):
        return f"{'—':>{w}}" if v is None else f"{v:>{w}.{d}f}"

    print("\n" + "=" * 96)
    print("1. Safety")
    print("=" * 96)
    print(f"  {'지표':<26}" + "".join(f"{r:>16}" for r in routes))
    for label, key, d in [("Grounding ↑", "ground", 2),
                          ("Coverage ↑", "cover", 2),
                          ("용어 보존 ↑", "term_keep", 2),
                          ("수치 보존 ↑ (보정)", "num_keep_ws", 2),
                          ("빈칸 보존 ↑", "blank_keep", 1)]:
        print(f"  {label:<26}"
              + "".join(f(avg(r, "safety", key), 16, d) for r in routes))
    print(f"  {'수치 환각 ↓ (보정, 총건)':<26}"
          + "".join(f"{tot(r,'safety','n_halluc_ws'):>16}" for r in routes))
    print(f"\n  {'[참고] 수치 보존 (legacy)':<26}"
          + "".join(f(avg(r, "safety", "num_keep"), 16, 2) for r in routes))
    print(f"  {'[참고] 수치 환각 (legacy)':<26}"
          + "".join(f"{tot(r,'safety','n_halluc'):>16}" for r in routes))
    print("""
  legacy 는 "1 시간" 을 "1시간" 으로 추출한 뒤 원문에서 찾아 불일치가 나는
  기존 정의입니다. 원문 띄어쓰기를 보존한 텍스트가 벌을 받으므로
  Identity 가 100 이 되지 않습니다. 보정 지표를 기준으로 판단하십시오.""")

    # ── Identity 를 측정 상한으로 표시 ──────────────────────────────
    print("\n" + "=" * 96)
    print("1-b. Identity 대비 — 측정 상한을 기준으로 본 값")
    print("=" * 96)
    print("""  Identity 는 원문 그대로이므로 Grounding 이 100 이어야 하지만 실제로는
  그렇지 않습니다. 동의서의 상당 부분이 서식 항목이라 NLI 가 명제로 판정할 수
  없기 때문입니다. 따라서 Identity 값이 이 문서 유형에서의 **측정 상한**입니다.""")
    print(f"\n  {'지표':<26}" + "".join(f"{r:>16}" for r in routes))
    for label, key in [("Grounding", "ground"), ("Coverage", "cover")]:
        base = avg("Identity", "safety", key)
        line = f"  {label} / Identity".ljust(28)
        for r in routes:
            v = avg(r, "safety", key)
            line += (f"{'—':>16}" if v is None or not base
                     else f"{v/base:>16.2f}")
        print(line)
    print("""
  1.00 을 넘는다는 것은 원문보다 충실하다는 뜻이 아닙니다.
  서식 항목이 문장으로 바뀌어 NLI 가 판정 가능한 형태가 되었다는 뜻입니다.
  Grounding 은 충실도 지표이자 '문장다움' 지표이기도 합니다.""")

    print("\n" + "=" * 96)
    print("2. Simplicity")
    print("=" * 96)
    print(f"  {'지표':<26}" + "".join(f"{r:>16}" for r in routes))
    for key in SPEC:
        d, cls, desc = SPEC[key]
        vals = [avg(r, "simplicity", key) for r in routes]
        if all(v is None for v in vals):
            continue
        print(f"  {arrow(key)} {key:<24}"
              + "".join(f(v, 16, 2) for v in vals))

    print("\n" + "=" * 96)
    print("3. Safety Gate  (실험 전 고정 기준)")
    print("=" * 96)
    print(f"  기준: 수치보존 ≥ {GATE['num_keep_min']} / "
          f"수치환각 ≤ {GATE['n_halluc_max']}건 / Grounding ≥ {GATE['ground_min']}")
    print(f"\n  {'Route':<18}{'문서':>6}{'통과':>7}{'탈락':>7}  주요 탈락 사유")
    for r in routes:
        sel = [x for x in rows if x["route"] == r]
        ok = sum(1 for x in sel if gate_pass(x["safety"])[0])
        reasons: Dict[str, int] = {}
        for x in sel:
            for why in gate_pass(x["safety"])[1]:
                kind = why.split()[0]
                reasons[kind] = reasons.get(kind, 0) + 1
        rs = ", ".join(f"{k} {v}건" for k, v in
                       sorted(reasons.items(), key=lambda x: -x[1])[:3])
        print(f"  {r:<18}{len(sel):>6}{ok:>7}{len(sel)-ok:>7}  {rs}")

    print("\n" + "=" * 96)
    print("4. Safety–Simplicity 관계")
    print("=" * 96)
    print(f"  {'Route':<18}{'Gate 통과율':>13}{'수치보존 ↑':>12}{'Copy ↓':>10}"
          f"{'어절/문장 ↓':>13}{'문장분할 ↑':>12}{'용어설명율 ↑':>13}")
    for r in routes:
        sel = [x for x in rows if x["route"] == r]
        ok = sum(1 for x in sel if gate_pass(x["safety"])[0])
        print(f"  {r:<18}{100*ok/max(len(sel),1):>12.1f}%"
              + f(avg(r, "safety", "num_keep_ws"), 12, 1)
              + f(avg(r, "simplicity", "copy_similarity"), 10, 3)
              + f(avg(r, "simplicity", "words_per_sent"), 13, 2)
              + f(avg(r, "simplicity", "sent_split_ratio"), 12, 2)
              + f(avg(r, "simplicity", "term_explain_rate"), 13, 1))

    id_copy = avg("Identity", "simplicity", "copy_similarity")
    print(f"""
  읽는 법
    · Identity 는 Copy similarity 가 {id_copy if id_copy is None else f'{id_copy:.2f}'} 이고
      평이화를 전혀 하지 않았으므로 Simplicity 개선이 0 입니다.
      그럼에도 Safety 지표에서는 가장 유리합니다.
    · 따라서 Fidelity 만으로 모델을 고르면 '아무것도 하지 않기' 가 최적이 됩니다.
      이것이 Safety 와 Simplicity 를 별도 축으로 두어야 하는 이유입니다.
    · 단일 총점을 만들지 않습니다. Gate 를 통과한 후보 중에서만 Simplicity 를 비교합니다.""")

    print("\n" + "=" * 96)
    print("5. 미해결 지표")
    print("=" * 96)
    for m in unresolved():
        print(f"  · {m}")
    print("=" * 96)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default="phase1_metrics.json")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")

    print("=" * 96)
    print("Phase 1 — Safety–Simplicity 재평가")
    print("=" * 96)
    print(f"  문서 {len(docs)}건 / Route {len(ROUTES)}종 (새 generation 없음)")
    for n, p, d in ROUTES:
        print(f"    {n:<18}{d}")

    Ks, problems = check_inputs(docs, a)
    if a.check_only:
        return
    if problems:
        print("\n  정합성 문제가 있어 중단합니다. --check-only 로 확인 후 재생성하십시오.")
        sys.exit(1)

    nli = NLI(a.gpu, a.batch)
    rows = []
    print("\n  평가 중", end="", flush=True)
    for doc in docs:
        K = Ks[doc]
        for name, pat, _ in ROUTES:
            rec = load_route(name, pat, doc, K)
            if rec is None:
                continue
            rec.setdefault("doc", doc)
            rows.append(eval_doc(name, rec, K, nli, a))
        print(".", end="", flush=True)
    print()

    report(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"gate": GATE, "routes": [r[0] for r in ROUTES],
                   "doc_hash": {d: h12(k) for d, k in Ks.items()},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
