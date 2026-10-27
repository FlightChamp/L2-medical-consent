#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standard_metrics.py — 연구 1: 표준 지표 재계산 (교수님 요청 1·2번)
====================================================================
"모델 비교 표의 지표가 문헌에서 쓰는 표준 지표인가?"에 답하기 위해,
저장된 출력만으로(새 생성 없음) 표준 정의 그대로의 지표를 계산해
우리 지표와 나란히 보고한다. 기존 지표·결과는 하나도 바꾸지 않는다.

계산하는 것 (문서 단위, 13문서 × 경로)

  [의미 보존 — Laban et al. 2022, SummaC]
    SummaC-ZS (P)   출력 문장마다 '원문 전체 문장' 중 최대 함의 확률 → 출력 문장 평균 × 100
                    (SummaC-ZS 원 정의: 문장 단위, 함의 점수, 열 최대 → 평균. 임계값·검색 없음)
    SummaC-ZS (R)   원문 문장마다 출력 전체 문장 중 최대 함의 확률 → 원문 문장 평균 × 100
                    (Cripwell et al.이 지적한 'SummaC 는 짧고 보수적인 출력을 선호' 를
                     보완하는 recall 방향. 그 논문의 정확한 변형과 같다고 주장하지 않는다)
    우리 Grounding  기존 정의 그대로 (상위 5개 1~3문장 묶음 최대 함의 ≥ 0.5 인 출력 문장 비율)
    우리 Cover      기존 정의 그대로 (방향만 반대)

  [보수성·길이 — Alva-Manchego et al. 2019, EASSE 의 참조 없는(QE) 지표를 문서 단위로]
    compression     출력 글자 수 ÷ 원문 글자 수   (1 미만 = 줄어듦)
    Levenshtein     1 − 편집거리 ÷ 긴 쪽 길이   (공백 정규화한 문서 전체 문자열)
    exact copies    원문 문장과 똑같은 출력 문장 비율 %
    split ratio     출력 문장 수 ÷ 원문 문장 수
    added / deleted 출력 어절 중 원문에 없던 비율 % / 원문 어절 중 출력에서 사라진 비율 %
    우리 Copy       기존 정의 그대로 (문자 3-gram Jaccard)

  [Safety Gate — phase2_eval.py 의 함수를 그대로 import]
    수치 보존 ≥ 90 / 수치 환각 0건 / Grounding ≥ 70  → 문서 단위 판정, 통과율 = 통과 문서 ÷ 문서 수

경로는 서버에 있는 결과를 자동으로 찾는다
    Identity(원문 그대로) · HARI P0 · outputs_compare/<모델>/ · outputs_2x2/{B,C,D}
    · Glossary 괄호형 · 문장 분리형 · '여기서' 원안

사용법
    cd ~/이윤우 && source .venv/bin/activate
    python standard_metrics.py                  # 전체 (GPU 0)
    python standard_metrics.py --list           # 찾은 경로만 보기
    python standard_metrics.py --routes Identity HARI_P0
    python standard_metrics.py --report-only    # 캐시로 리포트만

결과는 outputs_standard_metrics/ (git 제외 폴더) 와
standard_metrics.json / standard_metrics_per_doc.csv (수치만, 본문 없음) 에 저장.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
from typing import Dict, List, Optional, Tuple

HOME = os.environ.get("L2_HOME", os.path.expanduser("~/이윤우"))
for _p in (HOME, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt_ladder import extract_sections, Splitter, build_chunks, jaccard3   # noqa: E402

CACHE_DIR = os.path.join(HOME, "outputs_standard_metrics", "cache")
METRIC_VERSION = "v1"


# ===========================================================================
# 경로 탐색
# ===========================================================================

def discover_routes() -> Dict[str, dict]:
    R: Dict[str, dict] = {"Identity": {"kind": "identity", "pattern": None}}
    p0 = os.path.join(HOME, "outputs_prompt", "{doc}__P0.json")
    if glob.glob(p0.format(doc="*")):
        R["HARI_P0"] = {"kind": "json", "pattern": p0}
    for d in sorted(glob.glob(os.path.join(HOME, "outputs_compare", "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "doc*.json")):
            R["cmp:" + os.path.basename(d)] = {
                "kind": "json", "pattern": os.path.join(d, "{doc}.json")}
    for c in ("B", "C", "D"):
        d = os.path.join(HOME, "outputs_2x2", c)
        if glob.glob(os.path.join(d, "doc*.json")):
            R["2x2_" + c] = {"kind": "json", "pattern": os.path.join(d, "{doc}.json")}
    for name, sub in (("Gloss_inline", "outputs_glossary_v11"),
                      ("Gloss_sent", "outputs_glossary_v11_sent"),
                      ("Gloss_yeogi", "outputs_glossary_v11_sent_yeogi")):
        d = os.path.join(HOME, sub, "E")
        if glob.glob(os.path.join(d, "doc*.json")):
            R[name] = {"kind": "json", "pattern": os.path.join(d, "{doc}.json")}
    return R


def load_K(doc: str) -> Optional[str]:
    p = os.path.join(HOME, "docs", doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_out(route: dict, doc: str, K: str) -> Optional[str]:
    if route["kind"] == "identity":
        return K
    p = route["pattern"].format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    return r.get("E_output") or r.get("out")


# ===========================================================================
# NLI 쌍 계산 (기존 NLI 모델·토크나이저를 그대로 사용)
# ===========================================================================

def entail_probs(nli, prem: List[str], hyp: List[str]) -> List[float]:
    torch = nli.torch
    out: List[float] = []
    for i in range(0, len(prem), nli.bs):
        enc = nli.tok(prem[i:i + nli.bs], hyp[i:i + nli.bs], truncation=True,
                      padding=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(nli.dev) for k, v in enc.items()}
        with torch.no_grad():
            p = torch.softmax(nli.model(**enc).logits, dim=-1)
        out.extend(p[:, nli.i_ent].tolist())
    return out


def summac_zs(nli, premises: List[str], hyps: List[str]) -> Optional[float]:
    """SummaC-ZS: 각 가설 문장에 대해 모든 전제 문장 중 최대 함의 확률, 그 평균 × 100."""
    if not premises or not hyps:
        return None
    P, H, owner = [], [], []
    for i, h in enumerate(hyps):
        for s in premises:
            P.append(s); H.append(h); owner.append(i)
    sc = entail_probs(nli, P, H)
    best = [0.0] * len(hyps)
    for o, s in zip(owner, sc):
        if s > best[o]:
            best[o] = s
    return 100 * st.fmean(best)


# ===========================================================================
# EASSE 식 보수성·길이 지표 (참조 없음)
# ===========================================================================

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def levenshtein_similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein as L
        d = L.distance(a, b)
    except Exception:
        try:
            import Levenshtein as L2
            d = L2.distance(a, b)
        except Exception:
            if len(a) < len(b):
                a, b = b, a
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i] + [0] * len(b)
                for j, cb in enumerate(b, 1):
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
                prev = cur
            d = prev[-1]
    return 1 - d / max(len(a), len(b))


def easse_like(K: str, out: str, ks: List[str], oo: List[str]) -> dict:
    src_set = {_norm(s) for s in ks}
    kw = [w for w in K.split() if w.strip()]
    ow = [w for w in out.split() if w.strip()]
    kset, oset = set(kw), set(ow)
    return {
        "compression": len(_norm(out)) / max(len(_norm(K)), 1),
        "levenshtein_sim": levenshtein_similarity(K, out),
        "exact_copies": 100 * sum(1 for s in oo if _norm(s) in src_set) / max(len(oo), 1),
        "split_ratio": len(oo) / max(len(ks), 1),
        "added_pct": 100 * sum(1 for w in ow if w not in kset) / max(len(ow), 1),
        "deleted_pct": 100 * sum(1 for w in kw if w not in oset) / max(len(kw), 1),
        "copy_jaccard3": jaccard3(K, out),
    }


# ===========================================================================
# 통계
# ===========================================================================

def _rank(v: List[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 다시 계산")
    a = ap.parse_args()

    from phase2_eval import numeric_keep_ws, numeric_halluc_ws, GATE, gate_pass

    routes = discover_routes()
    if a.routes:
        routes = {k: v for k, v in routes.items() if k in a.routes}
    docs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(HOME, "docs", "*.txt"))
                  if not os.path.basename(p).startswith("syn"))

    print("=" * 100)
    print("연구 1 — 표준 지표 재계산 (저장된 출력만 사용, 새 생성 없음)")
    print("=" * 100)
    print(f"  문서 {len(docs)}건 · 경로 {len(routes)}개")
    for k, v in routes.items():
        n = sum(1 for d in docs if v["kind"] == "identity"
                or os.path.exists(v["pattern"].format(doc=d)))
        print(f"    {k:<34} {n}/{len(docs)}문서")
    if a.list:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    nli = None
    rows: List[dict] = []
    t0 = time.time()
    for rname, route in routes.items():
        print(f"\n  [{rname}] ", end="", flush=True)
        for d in docs:
            K = load_K(d)
            if not K:
                continue
            out = load_out(route, d, K)
            if not out:
                print("-", end="", flush=True)
                continue
            h = hashlib.sha1((METRIC_VERSION + str(a.k) + str(a.tau) + K + "\0" + out)
                             .encode("utf-8")).hexdigest()[:16]
            cp = os.path.join(CACHE_DIR, f"{re.sub(r'[^A-Za-z0-9._-]', '_', rname)}__{d}.json")
            cached = None
            if os.path.exists(cp) and not a.force:
                with open(cp, encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("hash") != h:
                    cached = None
            if cached is None:
                if a.report_only:
                    print("?", end="", flush=True)
                    continue
                if nli is None:
                    from translate_verify2 import NLI
                    nli = NLI(a.gpu, a.batch)
                ks, oo = Splitter.split(K), Splitter.split(out)
                nk, n_nums, _ = numeric_keep_ws(K, out)
                hal = numeric_halluc_ws(K, out)
                rec = {
                    "hash": h, "route": rname, "doc": d,
                    "n_src_sents": len(ks), "n_out_sents": len(oo),
                    "summac_zs_P": summac_zs(nli, ks, oo),
                    "summac_zs_R": summac_zs(nli, oo, ks),
                    "ground_ours": nli.rate(oo, build_chunks(ks), a.k, a.tau) if oo and ks else None,
                    "cover_ours": nli.rate(ks, build_chunks(oo), a.k, a.tau) if oo and ks else None,
                    "num_keep": nk, "n_nums": n_nums, "n_halluc": len(hal),
                    **easse_like(K, out, ks, oo),
                }
                ok, why = gate_pass({"num_keep": rec["num_keep"], "n_halluc": rec["n_halluc"],
                                     "ground": rec["ground_ours"]})
                rec["gate_pass"], rec["gate_fail_reasons"] = ok, why
                with open(cp, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                cached = rec
            rows.append(cached)
            print(".", end="", flush=True)
    print(f"\n\n  소요 {time.time() - t0:.0f}초")
    if not rows:
        sys.exit("[FATAL] 계산된 결과가 없습니다.")

    # ── 경로별 평균 ─────────────────────────────────────────────────────
    order = [r for r in routes if any(x["route"] == r for x in rows)]
    def M(r, k):
        v = [x[k] for x in rows if x["route"] == r and x.get(k) is not None]
        return st.fmean(v) if v else None
    def gate(r):
        s = [x for x in rows if x["route"] == r]
        return sum(1 for x in s if x["gate_pass"]), len(s)
    def f(v, w=8, d=2):
        return ("—".rjust(w)) if v is None else (f"{v:{w}.{d}f}")

    print("\n" + "=" * 100)
    print("표 1 — 의미 보존 (값이 높을수록 원문에 충실)")
    print("=" * 100)
    print("  " + "경로".ljust(34) + "SummaC-ZS P".rjust(12) + "SummaC-ZS R".rjust(12)
          + "우리Ground".rjust(11) + "우리Cover".rjust(10) + "수치보존".rjust(9)
          + "환각".rjust(6) + "Gate통과".rjust(10))
    for r in order:
        g_ok, g_n = gate(r)
        hal = sum(x["n_halluc"] for x in rows if x["route"] == r)
        print("  " + r.ljust(34) + f(M(r, "summac_zs_P"), 12) + f(M(r, "summac_zs_R"), 12)
              + f(M(r, "ground_ours"), 11) + f(M(r, "cover_ours"), 10)
              + f(M(r, "num_keep"), 9, 1) + f"{hal:>6}" + f"{g_ok:>5}/{g_n:<4}")

    print("\n" + "=" * 100)
    print("표 2 — 보수성·길이 (EASSE 참조 없는 지표의 문서 단위 적용)")
    print("=" * 100)
    print("  " + "경로".ljust(34) + "compress".rjust(9) + "Lev.sim".rjust(9)
          + "exact%".rjust(8) + "split".rjust(7) + "added%".rjust(8)
          + "deleted%".rjust(9) + "우리Copy".rjust(9))
    for r in order:
        print("  " + r.ljust(34) + f(M(r, "compression"), 9, 3) + f(M(r, "levenshtein_sim"), 9, 3)
              + f(M(r, "exact_copies"), 8, 1) + f(M(r, "split_ratio"), 7, 2)
              + f(M(r, "added_pct"), 8, 1) + f(M(r, "deleted_pct"), 9, 1)
              + f(M(r, "copy_jaccard3"), 9, 3))

    # ── 일치도 ─────────────────────────────────────────────────────────
    gen = [x for x in rows if x["route"] != "Identity"]
    sp_doc = spearman([x["ground_ours"] for x in gen], [x["summac_zs_P"] for x in gen])
    sp_cov = spearman([x["cover_ours"] for x in gen], [x["summac_zs_R"] for x in gen])
    sp_copy = spearman([x["copy_jaccard3"] for x in gen], [x["levenshtein_sim"] for x in gen])
    gen_routes = [r for r in order if r != "Identity"]
    sp_route = spearman([M(r, "ground_ours") for r in gen_routes],
                        [M(r, "summac_zs_P") for r in gen_routes])
    print("\n" + "=" * 100)
    print("표 3 — 우리 지표와 표준 지표의 일치도 (Spearman, Identity 제외)")
    print("=" * 100)
    def sp(v):
        return "—" if v is None else f"{v:+.3f}"
    print(f"  우리 Grounding vs SummaC-ZS P   문서 단위 {sp(sp_doc)}   경로 순위 {sp(sp_route)}"
          f"  (경로 {len(gen_routes)}개)")
    print(f"  우리 Cover     vs SummaC-ZS R   문서 단위 {sp(sp_cov)}")
    print(f"  우리 Copy      vs Levenshtein   문서 단위 {sp(sp_copy)}")
    rk_g = sorted(gen_routes, key=lambda r: -(M(r, "ground_ours") or 0))
    rk_s = sorted(gen_routes, key=lambda r: -(M(r, "summac_zs_P") or 0))
    print("\n  순위 (높은 순)")
    print("    " + "우리 Grounding".ljust(36) + "SummaC-ZS P")
    for i in range(len(gen_routes)):
        print(f"    {i+1}. {rk_g[i]:<33} {i+1}. {rk_s[i]}")

    # ── Gate 상세 ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Safety Gate — 문서 단위 판정 (phase2_eval.gate_pass 그대로)")
    print("=" * 100)
    print(f"  기준: 수치보존 ≥ {GATE['num_keep_min']} / 수치환각 ≤ {GATE['n_halluc_max']}건 / "
          f"Grounding ≥ {GATE['ground_min']}")
    print("  통과율 = 세 조건을 모두 만족한 문서 수 ÷ 문서 수")
    for r in order:
        g_ok, g_n = gate(r)
        fails = [f"{x['doc']}({', '.join(x['gate_fail_reasons'])})"
                 for x in rows if x["route"] == r and not x["gate_pass"]]
        print(f"  {r:<34} {g_ok}/{g_n} = {100*g_ok/max(g_n,1):5.1f}%   "
              + ("탈락: " + "; ".join(fails) if fails else "전부 통과"))

    # ── 저장 (수치만) ──────────────────────────────────────────────────
    keys = ["route", "doc", "n_src_sents", "n_out_sents", "summac_zs_P", "summac_zs_R",
            "ground_ours", "cover_ours", "num_keep", "n_nums", "n_halluc", "gate_pass",
            "compression", "levenshtein_sim", "exact_copies", "split_ratio",
            "added_pct", "deleted_pct", "copy_jaccard3"]
    with open(os.path.join(HOME, "standard_metrics_per_doc.csv"), "w",
              encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k) for k in keys})
    summary = {r: {k: M(r, k) for k in keys[4:] if k != "gate_pass"} for r in order}
    for r in order:
        g_ok, g_n = gate(r)
        summary[r]["gate_pass_rate"] = 100 * g_ok / max(g_n, 1)
        summary[r]["gate_pass"] = f"{g_ok}/{g_n}"
    with open(os.path.join(HOME, "standard_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "metric_version": METRIC_VERSION,
            "definitions": {
                "summac_zs_P": "Laban et al. 2022 SummaC-ZS: sentence-level, entailment prob, max over source sentences, mean over output sentences, x100",
                "summac_zs_R": "same with direction reversed (recall-oriented); not claimed identical to Cripwell et al.'s variant",
                "ground_ours": "project metric: % output sentences whose max entailment over top-k lexical 1-3 sentence source chunks >= tau",
                "compression": "chars(out)/chars(src), whitespace-normalised",
                "levenshtein_sim": "1 - edit distance / max length, whole document",
                "exact_copies": "% output sentences identical to a source sentence",
                "gate": "per-document: num_keep>=90, n_halluc<=0, ground_ours>=70 (phase2_eval.GATE)",
            },
            "params": {"k": a.k, "tau": a.tau},
            "spearman": {"ground_vs_summacP_doc": sp_doc, "ground_vs_summacP_route": sp_route,
                         "cover_vs_summacR_doc": sp_cov, "copy_vs_levenshtein_doc": sp_copy},
            "summary_by_route": summary,
        }, fh, ensure_ascii=False, indent=2)
    print("\n[SAVE] standard_metrics.json / standard_metrics_per_doc.csv (수치만, 본문 없음)")
    print("[CACHE] outputs_standard_metrics/cache/ — 다시 실행하면 바뀐 문서만 계산")


if __name__ == "__main__":
    main()
