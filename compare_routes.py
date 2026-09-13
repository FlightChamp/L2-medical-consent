#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_routes.py — Route A / B / C 비교 및 오류 전파 분석
===========================================================
비교 대상 (모두 같은 K 에서 출발, 최종 산출물은 영어):
    Route A   K → HARI 한국어 평이화 → 영어        {doc}__A_en.json
    Route B   K → 영어(E0) → MedGemma 영어 평이화  {doc}__E1.json
    Route C   K → 영어(E0)                         {doc}__E0.json

검증 방향 세 가지:
    K  ↔ E0    1차 번역에서 생긴 손실
    E0 ↔ E1    MedGemma 평이화에서 생긴 손실·창작   (영어 단일언어)
    K  ↔ E1    최종 결과가 최초 원문을 얼마나 보존하는지  ★핵심

    중간 단계가 각각 좋아 보여도 최종이 원문에서 멀어질 수 있으므로
    K ↔ E1 을 최종 판단 근거로 삼는다.

교차 언어 검색 주의:
    한국어와 영어는 문자 n-gram 겹침이 0 이라 Jaccard 상위 k 선택이 작동하지 않는다.
    (이전 실험에서 이 때문에 점수가 24~35 로 나왔다가 수정 후 70~93 이 되었다)
    따라서 교차 언어 방향에서는 k=0, 즉 전체 청크를 비교한다.

지표 적용 가능성:
    Grounding / Coverage      세 방향 모두 가능
    Numeric preservation      세 방향 모두 가능 (숫자는 언어 무관)
    Numeric hallucination     세 방향 모두 가능
    Length change             세 방향 모두 가능
    Copy similarity           E0↔E1 만 가능 (교차 언어는 문자 겹침이 0)
    Top-20 어휘 보존          한국어 전용이므로 교차 방향 적용 불가
    Terminology               K↔E0, K↔E1 만 가능 (한-영 사전 29개)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python compare_routes.py --docs doc7 doc12 doc11      # smoke
    python compare_routes.py                              # 전체
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from prompt_ladder import (
        extract_sections, Splitter, build_chunks, jaccard3,
    )
    # NLI 는 translate_verify2 쪽을 쓴다.
    # prompt_ladder.NLI.rate 는 k=0 분기가 없어 topk 가 빈 리스트를 돌려주고
    # 교차 언어 비교가 전부 0 이 된다.
    from translate_verify2 import (
        NLI, ko_is_pred, fo_is_pred, Splitter as FSplitter,
        load_terms, numbers_of, build_chunks as build_chunks2,
    )
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py / translate_verify2.py 필요: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
PIVOTDIR = os.path.join(HOME, "outputs_medgemma_pivot")

# HARI 기준선 (13문서, 한국어 평이화 단계. 참고용)
HARI_KO_BASELINE = {
    "ground": 83.64, "cover": 63.20, "term_keep": 39.62,
    "copy_sim": 0.30, "num_keep": 99.45, "n_halluc": 2, "delta": -10.35,
}


# ===========================================================================

def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_stage(doc: str, stage: str) -> Optional[dict]:
    p = os.path.join(PIVOTDIR, f"{doc}__{stage}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ko_sents(t: str) -> List[str]:
    """한국어 — 서술문만 (NLI 는 명제에만 적용된다)"""
    return [s for s in Splitter.split(t) if ko_is_pred(s)]


def en_sents(t: str) -> List[str]:
    return [s for s in FSplitter.foreign(t, "en") if fo_is_pred(s, "en")]


def num_metrics(src: str, out: str) -> Tuple[Optional[float], int, List[str]]:
    """(보존율, 환각 개수, 환각 목록). 숫자만 비교하므로 언어 무관."""
    s, o = numbers_of(src), numbers_of(out)
    keep = 100 * len(s & o) / len(s) if s else None
    hal = sorted(o - s)
    return keep, len(hal), hal


# ===========================================================================

class Scorer:
    def __init__(self, nli: NLI, tau: float):
        self.nli = nli
        self.tau = tau

    def pair(self, a_sents: List[str], b_sents: List[str],
             cross: bool) -> Dict[str, Optional[float]]:
        """a=소스, b=산출물. cross=True 면 전체 청크 비교(k=0)."""
        k = 0 if cross else 5
        if not a_sents or not b_sents:
            return {"ground": None, "cover": None, "min": None}
        g = self.nli.rate(b_sents, build_chunks(a_sents), k, self.tau)
        c = self.nli.rate(a_sents, build_chunks(b_sents), k, self.tau)
        return {"ground": round(g, 1), "cover": round(c, 1),
                "min": round(min(g, c), 1)}


def analyze_doc(doc: str, a, sc: Scorer, terms: Dict) -> Optional[dict]:
    K = load_K(doc, a.docdir)
    e0, e1, aen = load_stage(doc, "E0"), load_stage(doc, "E1"), load_stage(doc, "A_en")
    if not (K and e0):
        return None
    E0 = e0["out"]
    E1 = e1["out"] if e1 else None
    AEN = aen["out"] if aen else None
    S_ko = aen["src"] if aen else None          # HARI 한국어 평이화문

    ks, e0s = ko_sents(K), en_sents(E0)
    r: dict = {"doc": doc,
               "len": {"K": len(K), "E0": len(E0),
                       "E1": len(E1) if E1 else None,
                       "A_en": len(AEN) if AEN else None,
                       "S_ko": len(S_ko) if S_ko else None},
               "n_sents": {"K_pred": len(ks), "E0_pred": len(e0s)}}

    # ── 방향별 NLI ────────────────────────────────────────────────────
    r["K_E0"] = sc.pair(ks, e0s, cross=True)
    if E1:
        e1s = en_sents(E1)
        r["n_sents"]["E1_pred"] = len(e1s)
        r["E0_E1"] = sc.pair(e0s, e1s, cross=False)   # 영어-영어
        r["K_E1"] = sc.pair(ks, e1s, cross=True)
    if AEN:
        aens = en_sents(AEN)
        r["n_sents"]["A_en_pred"] = len(aens)
        r["K_A_en"] = sc.pair(ks, aens, cross=True)

    # ── 수치 ──────────────────────────────────────────────────────────
    r["num"] = {}
    for tag, s, o in [("K_E0", K, E0),
                      ("E0_E1", E0, E1), ("K_E1", K, E1),
                      ("K_A_en", K, AEN)]:
        if o is None:
            continue
        keep, nh, hal = num_metrics(s, o)
        r["num"][tag] = {"keep": round(keep, 1) if keep is not None else None,
                         "n_halluc": nh, "halluc": hal[:8]}

    # ── 용어 (한-영 사전. 교차 방향만) ─────────────────────────────────
    present = [t for t in terms if t in K]
    r["term"] = {"n_present": len(present)}
    for tag, o in [("K_E0", E0), ("K_E1", E1), ("K_A_en", AEN)]:
        if o is None:
            continue
        hit = miss = 0
        missed = []
        for t in present:
            c = terms[t].get("en")
            if not c:
                continue
            if c.lower() in o.lower():
                hit += 1
            else:
                miss += 1
                missed.append(f"{t}→{c}")
        r["term"][tag] = {
            "acc": round(100 * hit / (hit + miss), 1) if hit + miss else None,
            "n": hit + miss, "missed": missed[:8]}

    # ── 복사유사도 (영어-영어만) ───────────────────────────────────────
    r["copy_sim"] = {"E0_E1": round(jaccard3(E0, E1), 3) if E1 else None}

    # ── 분량 ──────────────────────────────────────────────────────────
    r["delta"] = {
        "K_E0": round(100 * (len(E0) - len(K)) / max(len(K), 1), 1),
        "E0_E1": round(100 * (len(E1) - len(E0)) / max(len(E0), 1), 1) if E1 else None,
        "K_E1": round(100 * (len(E1) - len(K)) / max(len(K), 1), 1) if E1 else None,
        "K_A_en": round(100 * (len(AEN) - len(K)) / max(len(K), 1), 1) if AEN else None,
    }
    return r


# ===========================================================================

def fmt(v, w=7, d=1):
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.{d}f}" if isinstance(v, float) else f"{v:>{w}}"


def report(rows: List[dict]):
    if not rows:
        print("결과 없음")
        return

    def avg(path: List, rows_=None):
        vals = []
        for r in (rows_ or rows):
            cur = r
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if cur is not None:
                vals.append(cur)
        return st.fmean(vals) if vals else None

    def tot(path: List):
        s = 0
        for r in rows:
            cur = r
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if cur is not None:
                s += cur
        return s

    print("\n" + "=" * 100)
    print("1. 문서별 — 최종 결과가 원문을 얼마나 보존하는가 (K ↔ 최종 영어)")
    print("=" * 100)
    print(f"  {'문서':<7}"
          f"{'A 근거':>8}{'A 커버':>8}{'A MIN':>8}  "
          f"{'B 근거':>8}{'B 커버':>8}{'B MIN':>8}  "
          f"{'C 근거':>8}{'C 커버':>8}{'C MIN':>8}")
    for r in rows:
        a_, b_, c_ = r.get("K_A_en", {}), r.get("K_E1", {}), r.get("K_E0", {})
        print(f"  {r['doc']:<7}"
              + fmt(a_.get("ground"), 8) + fmt(a_.get("cover"), 8) + fmt(a_.get("min"), 8) + "  "
              + fmt(b_.get("ground"), 8) + fmt(b_.get("cover"), 8) + fmt(b_.get("min"), 8) + "  "
              + fmt(c_.get("ground"), 8) + fmt(c_.get("cover"), 8) + fmt(c_.get("min"), 8))

    print("\n" + "=" * 100)
    print("2. Route 비교 — 평균")
    print("=" * 100)
    print(f"  {'지표':<24}{'A (HARI KO→EN)':>18}{'B (EN+MedGemma)':>18}"
          f"{'C (EN only)':>16}")
    spec = [
        ("Grounding (K↔최종)", ["K_A_en", "ground"], ["K_E1", "ground"], ["K_E0", "ground"]),
        ("Coverage (K↔최종)", ["K_A_en", "cover"], ["K_E1", "cover"], ["K_E0", "cover"]),
        ("Bidirectional MIN", ["K_A_en", "min"], ["K_E1", "min"], ["K_E0", "min"]),
        ("Numeric preservation", ["num", "K_A_en", "keep"], ["num", "K_E1", "keep"], ["num", "K_E0", "keep"]),
        ("Terminology accuracy", ["term", "K_A_en", "acc"], ["term", "K_E1", "acc"], ["term", "K_E0", "acc"]),
        ("Length change vs K %", ["delta", "K_A_en"], ["delta", "K_E1"], ["delta", "K_E0"]),
    ]
    for label, pa, pb, pc in spec:
        va, vb, vc = avg(pa), avg(pb), avg(pc)
        print(f"  {label:<24}" + fmt(va, 18, 2) + fmt(vb, 18, 2) + fmt(vc, 16, 2))
    print(f"  {'Numeric hallucination':<24}"
          f"{tot(['num','K_A_en','n_halluc']):>18}"
          f"{tot(['num','K_E1','n_halluc']):>18}"
          f"{tot(['num','K_E0','n_halluc']):>16}   (총 건수)")

    print("\n  적용 불가 지표")
    print("    Copy similarity   교차 언어는 문자 겹침이 0 이므로 K↔최종에 적용 불가")
    print("    Top-20 어휘보존   한국어 문자열 기준이므로 영어 산출물에 적용 불가")
    cs = avg(["copy_sim", "E0_E1"])
    print(f"    → 대신 E0↔E1 복사유사도(영어-영어): {cs:.3f}" if cs is not None else "")

    print("\n" + "=" * 100)
    print("3. 오류 전파 — 어느 단계에서 잃었는가 (Route B)")
    print("=" * 100)
    print(f"  {'단계':<22}{'Grounding':>12}{'Coverage':>12}"
          f"{'수치보존':>12}{'용어정확':>12}{'분량변화':>12}")
    stages = [
        ("K → E0  (1차 번역)", ["K_E0", "ground"], ["K_E0", "cover"],
         ["num", "K_E0", "keep"], ["term", "K_E0", "acc"], ["delta", "K_E0"]),
        ("E0 → E1 (평이화)", ["E0_E1", "ground"], ["E0_E1", "cover"],
         ["num", "E0_E1", "keep"], None, ["delta", "E0_E1"]),
        ("K → E1  (최종)", ["K_E1", "ground"], ["K_E1", "cover"],
         ["num", "K_E1", "keep"], ["term", "K_E1", "acc"], ["delta", "K_E1"]),
    ]
    for label, pg, pc2, pn, pt, pd in stages:
        line = f"  {label:<22}" + fmt(avg(pg), 12, 1) + fmt(avg(pc2), 12, 1) \
               + fmt(avg(pn), 12, 1)
        line += fmt(avg(pt), 12, 1) if pt else f"{'—':>12}"
        line += fmt(avg(pd), 12, 1)
        print(line)

    t0 = avg(["term", "K_E0", "acc"])
    t1 = avg(["term", "K_E1", "acc"])
    if t0 is not None and t1 is not None:
        print(f"\n  용어 손실 분해")
        print(f"    1차 번역에서 잃음 : {100-t0:>5.1f}%p  (사전 용어 기준)")
        print(f"    평이화에서 추가 손실: {t0-t1:>5.1f}%p")
        print(f"    최종 잔존          : {t1:>5.1f}%")
        print("    → 1차 번역에서 사라진 정보는 이후 단계에서 복원되지 않습니다.")

    # 문서별 누락 용어 예시
    ex = [r for r in rows if r.get("term", {}).get("K_E0", {}).get("missed")]
    if ex:
        print(f"\n  1차 번역에서 누락된 용어 예시")
        for r in ex[:4]:
            m = r["term"]["K_E0"]["missed"]
            print(f"    {r['doc']}: {', '.join(m[:5])}")

    print("\n" + "=" * 100)
    print("4. 판정")
    print("=" * 100)
    a_min, b_min, c_min = (avg(["K_A_en", "min"]), avg(["K_E1", "min"]),
                           avg(["K_E0", "min"]))
    a_t, b_t = avg(["term", "K_A_en", "acc"]), avg(["term", "K_E1", "acc"])
    a_n, b_n = avg(["num", "K_A_en", "keep"]), avg(["num", "K_E1", "keep"])

    if None in (a_min, b_min):
        print("  A 또는 B 산출물이 없어 판정할 수 없습니다.")
    else:
        d = b_min - a_min
        print(f"  Bidirectional MIN   A {a_min:.1f}  vs  B {b_min:.1f}   ({d:+.1f})")
        if c_min is not None:
            print(f"  대조군 C            {c_min:.1f}   "
                  f"(B−C = {b_min-c_min:+.1f} ← MedGemma 평이화 단계의 순효과)")
        if a_t is not None and b_t is not None:
            print(f"  Terminology         A {a_t:.1f}  vs  B {b_t:.1f}   ({b_t-a_t:+.1f})")
        if a_n is not None and b_n is not None:
            print(f"  Numeric             A {a_n:.1f}  vs  B {b_n:.1f}   ({b_n-a_n:+.1f})")

        print()
        if d >= 5 and (b_t is None or a_t is None or b_t >= a_t - 5):
            print("  → B 우세. 영어 피벗 구조를 후속 후보로 검토할 수 있습니다.")
            print("     단 변환 단계가 하나 늘어나므로 3번 오류 전파 분석을 함께 보십시오.")
        elif abs(d) < 5:
            print("  → 비슷함. 구조 복잡도와 추가 오류 전파 위험을 고려하면")
            print("     기존 HARI 구조를 유지하는 편이 낫습니다.")
        else:
            print("  → B 열세. 영어 피벗 실험을 기각하고 기존 HARI 구조를 유지합니다.")

        if c_min is not None and b_min - c_min < 2:
            print("\n  [주의] B 와 C 의 차이가 작습니다.")
            print("         B 의 성능이 MedGemma 평이화가 아니라 영어 피벗 자체에서")
            print("         왔을 가능성이 있습니다.")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="compare_routes_metrics.json")
    a = ap.parse_args()

    docs = a.docs or sorted({
        os.path.basename(p).split("__")[0]
        for p in glob.glob(os.path.join(PIVOTDIR, "*__E0.json"))})
    if not docs:
        sys.exit(f"[FATAL] {PIVOTDIR} 에 결과가 없습니다. "
                 "medgemma_en_pivot.py 를 먼저 실행하세요.")

    print("=" * 100)
    print("Route A / B / C 비교")
    print("=" * 100)
    print(f"  문서 {len(docs)}건: {', '.join(docs)}")
    print("  A: K → HARI 한국어 평이화 → 영어")
    print("  B: K → 영어(E0) → MedGemma 영어 평이화(E1)")
    print("  C: K → 영어(E0)")
    print("  교차 언어 방향은 k=0 (전체 청크 비교) — 문자 n-gram 검색이 무효이므로")

    terms = load_terms()
    print(f"  용어 사전 {len(terms)}개 (한-영)")
    nli = NLI(a.gpu, a.batch)
    sc = Scorer(nli, a.tau)

    rows = []
    for d in docs:
        r = analyze_doc(d, a, sc, terms)
        if r:
            rows.append(r)
            print(f"  {d} 완료", end="  ", flush=True)
    print()

    report(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"baseline_hari_ko": HARI_KO_BASELINE, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
