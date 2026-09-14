#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_medllama.py — Medical-Llama3 평가
===========================================
한국어(--lang ko):
    HARI baseline 과 동일한 지표로 비교한다. 지표 함수를 prompt_ladder.py 에서
    직접 import 하므로 계산 방식이 한 글자도 다르지 않다.

    HARI baseline (13문서, whole-document, P0):
        Grounding 83.64 / Coverage 63.20 / Top-20 retention 39.62
        Copy similarity 0.30 / Numeric preservation 99.45 / Numeric hallucination 2
        Length change -10.35

영어(--lang en):
    K ↔ E0 ↔ E1 세 방향을 본다. MedGemma 영어 피벗 실험과 동일한 구조이며
    교차 언어 방향은 k=0(전체 청크)을 쓴다.

지표 분류를 함께 표시한다:
    표준 지표        Recall, Specificity, AUROC
    프로젝트 정의    Grounding, Coverage, Top-20 retention, Copy similarity,
                     Numeric preservation/hallucination

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python experiment_medllama/evaluate_medllama.py --lang ko
    python experiment_medllama/evaluate_medllama.py --lang en
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import (
        Splitter, build_chunks, jaccard3, extract_terms, numeric_units,
        numeric_halluc, readability, extract_sections,
    )
    from translate_verify2 import NLI, ko_is_pred, fo_is_pred, Splitter as FSplitter
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py / translate_verify2.py 필요: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
PIVOTDIR = os.path.join(HOME, "outputs_medgemma_pivot")
OUTDIR = os.path.join(HERE, "outputs")

HARI_BASELINE = {
    "ground": 83.64, "cover": 63.20, "term_keep": 39.62,
    "copy_sim": 0.30, "num_keep": 99.45, "n_halluc": 2, "delta": -10.35,
}


def load_outputs(lang: str) -> List[dict]:
    d = os.path.join(OUTDIR, lang)
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def load_K(doc: str) -> Optional[str]:
    p = os.path.join(DOCDIR, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def eval_ko(recs: List[dict], nli: NLI, a) -> List[dict]:
    rows = []
    print(f"\n  {'문서':<7}{'분량':>8}{'용어':>7}{'수치':>7}{'환각':>6}"
          f"{'근거':>7}{'커버':>7}{'복사':>7}{'한글비':>8}")
    for r in recs:
        src, out = r["src"], r["out"]
        terms = extract_terms(src)
        nums = sorted(set(numeric_units(src)))
        ss, oo = Splitter.split(src), Splitter.split(out)
        ko = len(re.findall(r"[가-힣]", out))
        row = {
            "doc": r["doc"],
            "delta": 100 * (len(out) - len(src)) / max(len(src), 1),
            "term_keep": round(100 * sum(1 for t in terms if t in out)
                               / max(len(terms), 1), 1),
            "num_keep": round(100 * sum(1 for x in nums if x in out)
                              / len(nums), 1) if nums else None,
            "n_halluc": len(numeric_halluc(src, out)),
            "ground": round(nli.rate(oo, build_chunks(ss), a.k, a.tau), 1)
                      if oo and ss else None,
            "cover": round(nli.rate(ss, build_chunks(oo), a.k, a.tau), 1)
                     if oo and ss else None,
            "copy_sim": round(jaccard3(src, out), 3),
            "ko_ratio": round(100 * ko / max(len(out), 1), 1),
            "runtime": r.get("runtime_sec"),
        }
        rows.append(row)
        nk = f"{row['num_keep']:.1f}" if row["num_keep"] is not None else "—"
        g = f"{row['ground']:.1f}" if row["ground"] is not None else "—"
        c = f"{row['cover']:.1f}" if row["cover"] is not None else "—"
        print(f"  {row['doc']:<7}{row['delta']:>+7.0f}%{row['term_keep']:>7.1f}"
              f"{nk:>7}{row['n_halluc']:>6}{g:>7}{c:>7}"
              f"{row['copy_sim']:>7.2f}{row['ko_ratio']:>7.1f}%", flush=True)
    return rows


def report_ko(rows: List[dict], recs: List[dict]):
    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return st.fmean(v) if v else float("nan")

    tot_h = sum(r["n_halluc"] for r in rows)
    print("\n" + "=" * 84)
    print(f"HARI 8B vs Medical-Llama3 8B  (13문서, whole-document, P0)")
    print("=" * 84)
    print(f"  {'Metric':<26}{'HARI 8B':>14}{'Medical-Llama3 8B':>20}{'차이':>12}")
    pairs = [("Grounding", "ground", HARI_BASELINE["ground"]),
             ("Coverage", "cover", HARI_BASELINE["cover"]),
             ("Top-20 retention", "term_keep", HARI_BASELINE["term_keep"]),
             ("Copy similarity", "copy_sim", HARI_BASELINE["copy_sim"]),
             ("Numeric preservation", "num_keep", HARI_BASELINE["num_keep"]),
             ("Length change", "delta", HARI_BASELINE["delta"])]
    for label, key, base in pairs:
        v = avg(key)
        print(f"  {label:<26}{base:>14.2f}{v:>20.2f}{v-base:>+12.2f}")
    print(f"  {'Numeric hallucination':<26}{HARI_BASELINE['n_halluc']:>14}"
          f"{tot_h:>20}{tot_h-HARI_BASELINE['n_halluc']:>+12}")

    print(f"\n  {'참고':<26}{'':>14}{'Medical-Llama3':>20}")
    print(f"  {'한글 비율 %':<26}{'':>14}{avg('ko_ratio'):>20.2f}")
    rt = [r["runtime"] for r in rows if r.get("runtime")]
    if rt:
        print(f"  {'문서당 생성 시간(초)':<26}{'':>14}{st.fmean(rt):>20.1f}")

    ts = {r.get("chat_template_source") for r in recs}
    print(f"\n  chat template: {', '.join(str(t) for t in ts)}")

    print("\n" + "=" * 84)
    print("Stop / Go 판정")
    print("=" * 84)
    cs, kor, g = avg("copy_sim"), avg("ko_ratio"), avg("ground")
    fails = []
    if cs >= 0.6:
        fails.append(f"Copy similarity {cs:.2f} ≥ 0.6 — 원문 전사 수준")
    if kor < 40:
        fails.append(f"한글 비율 {kor:.1f}% < 40% — 한국어로 출력하지 않음")
    if g == g and g < HARI_BASELINE["ground"] - 10:
        fails.append(f"Grounding {g:.1f} — HARI 대비 10 이상 낮음")
    if tot_h > HARI_BASELINE["n_halluc"] * 2:
        fails.append(f"Numeric hallucination {tot_h}건 — HARI({HARI_BASELINE['n_halluc']})의 2배 초과")
    if fails:
        print("  FAIL")
        for f in fails:
            print(f"    · {f}")
        print("\n  → 한국어 평이화 미채택. 영어 조건에서 1회 추가 실험합니다.")
    else:
        print("  GO — HARI 와 경쟁 가능한 수준입니다. 지표별로 상세 비교하십시오.")
    print("""
  지표 분류
    프로젝트 정의 : Grounding, Coverage, Top-20 retention, Copy similarity,
                    Numeric preservation / hallucination
    (표준 통계 지표인 Recall/Specificity/AUROC 는 이 표에 포함되지 않습니다)""")
    print("=" * 84)


def eval_en(recs: List[dict], nli: NLI, a) -> List[dict]:
    """K ↔ E0 ↔ E1 세 방향. 교차 언어는 k=0."""
    rows = []
    print(f"\n  {'문서':<7}{'K↔E0 MIN':>11}{'E0↔E1 MIN':>11}{'K↔E1 MIN':>11}"
          f"{'수치(K→E1)':>12}{'환각':>6}{'복사(E0E1)':>12}{'분량(K→E1)':>12}")
    for r in recs:
        doc, E0, E1 = r["doc"], r["src"], r["out"]
        K = load_K(doc)
        if not K:
            continue
        ks = [s for s in Splitter.split(K) if ko_is_pred(s)]
        e0s = [s for s in FSplitter.foreign(E0, "en") if fo_is_pred(s, "en")]
        e1s = [s for s in FSplitter.foreign(E1, "en") if fo_is_pred(s, "en")]

        def pair(A, B, cross):
            if not A or not B:
                return None
            k = 0 if cross else 5
            g = nli.rate(B, build_chunks(A), k, a.tau)
            c = nli.rate(A, build_chunks(B), k, a.tau)
            return round(min(g, c), 1)

        ns_K = {m.group(1) for m in re.finditer(r"(?<![\d.])(\d{1,4})(?![\d.])", K)}
        ns_E1 = {m.group(1) for m in re.finditer(r"(?<![\d.])(\d{1,4})(?![\d.])", E1)}
        row = {
            "doc": doc,
            "K_E0": pair(ks, e0s, True),
            "E0_E1": pair(e0s, e1s, False),
            "K_E1": pair(ks, e1s, True),
            "num_keep": round(100 * len(ns_K & ns_E1) / len(ns_K), 1) if ns_K else None,
            "n_halluc": len(ns_E1 - ns_K),
            "copy_E0E1": round(jaccard3(E0, E1), 3),
            "delta_K_E1": round(100 * (len(E1) - len(K)) / max(len(K), 1), 1),
        }
        rows.append(row)
        f = lambda v, w=11: (f"{v:>{w}.1f}" if isinstance(v, float) else f"{'—':>{w}}")
        print(f"  {doc:<7}{f(row['K_E0'])}{f(row['E0_E1'])}{f(row['K_E1'])}"
              f"{f(row['num_keep'],12)}{row['n_halluc']:>6}"
              f"{row['copy_E0E1']:>12.2f}{row['delta_K_E1']:>11.1f}%", flush=True)
    return rows


def report_en(rows: List[dict]):
    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return st.fmean(v) if v else float("nan")

    print("\n" + "=" * 84)
    print("Medical-Llama3 영어 피벗 — 단계별")
    print("=" * 84)
    print(f"  {'단계':<24}{'Bidirectional MIN':>20}")
    for label, key in [("K → E0  (1차 번역)", "K_E0"),
                       ("E0 → E1 (평이화)", "E0_E1"),
                       ("K → E1  (최종)", "K_E1")]:
        print(f"  {label:<24}{avg(key):>20.2f}")
    print(f"\n  {'수치보존 (K→E1)':<24}{avg('num_keep'):>20.2f}")
    print(f"  {'수치환각 (총 건수)':<24}"
          f"{sum(r['n_halluc'] for r in rows):>20}")
    print(f"  {'복사유사도 (E0↔E1)':<24}{avg('copy_E0E1'):>20.3f}")
    print(f"  {'분량 변화 (K→E1)':<24}{avg('delta_K_E1'):>19.1f}%")
    print("""
  참고 — MedGemma 영어 피벗 실험 (동일 구조, 13문서)
    Route B (MedGemma)  Bidirectional MIN 63.16  수치보존 69.76  환각 5건
    Route C (번역만)     Bidirectional MIN 83.65  수치보존 100.0  환각 1건
    Route A (HARI)      Bidirectional MIN 64.87  수치보존 81.48  환각 0건""")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = load_outputs(a.lang)
    if not recs:
        sys.exit(f"[FATAL] {OUTDIR}/{a.lang}/ 에 결과가 없습니다. "
                 f"run_medllama.py --lang {a.lang} 를 먼저 실행하세요.")
    print("=" * 84)
    print(f"Medical-Llama3 평가 — {a.lang.upper()}  ({len(recs)}건)")
    print("=" * 84)

    nli = NLI(a.gpu, a.batch)
    if a.lang == "ko":
        rows = eval_ko(recs, nli, a)
        report_ko(rows, recs)
    else:
        rows = eval_en(recs, nli, a)
        report_en(rows)

    out = a.out or os.path.join(OUTDIR, f"evaluate_medllama_{a.lang}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"lang": a.lang, "model": recs[0].get("model"),
                   "hari_baseline": HARI_BASELINE, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
