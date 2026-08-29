#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expansion_audit.py — 팽창률 감사 (기준을 먼저 정하지 않는다)
=============================================================
목표:
    "몇 배 이상이면 오류"를 미리 정하지 않는다.
    지금까지 임계값을 먼저 잡았다가 두 번 되돌렸다(tau* 규칙, min-hangul 25).
    데이터를 보고 정한다.

    ① 기존 출력 전부에서 섹션 단위 길이비 수집
    ② 분포 확인 — 원문 길이 구간별로 나눠 본다
    ③ 팽창률이 품질 저하와 실제로 연관되는지 확인 (근거율·수치환각과의 상관)
    ④ 이상치 기준 후보 도출 (IQR / 백분위, 둘 다 제시)
    ⑤ 맹검 검토 시트 생성 — 사람이 유형을 판정한다
    ⑥ WARN 가드 구현은 ⑤ 결과를 보고 붙인다 (이 스크립트에 없음)

왜 섹션 단위인가:
    문서 단위로 보면 이상치가 희석된다.
    doc5 는 문서 전체로 5.7배지만 6번 섹션 하나가 1,188배다.

왜 원문 길이 구간별인가:
    6자가 30자 되는 5배는 정상일 수 있고,
    3,000자가 6,000자 되는 2배는 비정상일 수 있다.

왜 맹검인가:
    팽창률 높은 것만 보여주면 '크니까 문제겠지'로 판정이 쏠린다.
    이상치 후보와 정상 구간 대조군을 섞고 순서를 무작위로 한다.
    정답은 별도 파일에 두어 판정 후 대조한다.

팽창의 세 유형 (사람이 가릴 것):
    부연설명   정상. "갑상선은 목 앞부분의 나비 모양 기관입니다"
    서식재구성 정상. 항목 나열을 문장으로 풀어쓰기
    내용창작   오류. 빈칸을 "1시간"으로 채움

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python expansion_audit.py                # 근거율 포함 (GPU, 수 분)
    python expansion_audit.py --no-nli       # 길이·수치만 (GPU 불필요, 즉시)
    python expansion_audit.py --review-n 30
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
OUTDIR = os.path.expanduser("~/이윤우/outputs")

# 원문 한글 길이 구간. 짧은 쪽을 촘촘히 나눈다.
BUCKETS = [(0, 15), (15, 50), (50, 150), (150, 400), (400, 10 ** 9)]
BUCKET_LABELS = ["0-15자", "15-50자", "50-150자", "150-400자", "400자+"]


# ===========================================================================
# 텍스트 유틸 (기존 스크립트와 동일 기준)
# ===========================================================================

def clean(t) -> str:
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls.BOUND.split(str(t)):
            s = clean(p or "")
            if s and not cls.NUM_PREFIX.match(s) and len(s) >= 10:
                out.append(s)
        return out


def build_chunks(sents: Sequence[str], span: int = 3) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for k in range(1, span + 1):
        for i in range(0, len(sents) - k + 1):
            c = " ".join(sents[i:i + k])
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out or ["(빈 문서)"]


def ngrams(t: str, n: int) -> Set[str]:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + n] for i in range(len(s) - n + 1)} or {s}


def topk(hyp: str, chunks: Sequence[str], k: int) -> List[str]:
    hg = ngrams(hyp, 2)
    scored = []
    for i, c in enumerate(chunks):
        g = ngrams(c, 2)
        scored.append((len(hg & g) / max(len(hg | g), 1), i))
    scored.sort(key=lambda x: -x[0])
    return [chunks[i] for _, i in scored[:k]]


def hangul_len(t: str) -> int:
    return len(re.findall(r"[가-힣]", t))


NUM_UNIT = re.compile(
    r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번|세)")


def numeric_hallucination(src: str, out: str) -> List[str]:
    s = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(src)}
    o = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(out)}
    return sorted(o - s)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[i]


# ===========================================================================
# NLI (선택)
# ===========================================================================

class NLI:
    def __init__(self, gpu: int, batch: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.bs = batch
        self.dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[NLI] {NLI_MODEL} → {self.dev}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL).to(self.dev).eval()
        id2 = {int(k): str(v).lower()
               for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))

    def rate(self, hyps: List[str], chunks: List[str], k: int,
             tau: float) -> float:
        torch = self.torch
        if not hyps or not chunks:
            return float("nan")
        prem, hy, owner = [], [], []
        for i, h in enumerate(hyps):
            for c in topk(h, chunks, k):
                prem.append(c)
                hy.append(h)
                owner.append(i)
        sc: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hy[i:i + self.bs],
                           truncation=True, padding=True, max_length=256,
                           return_tensors="pt")
            enc = {kk: v.to(self.dev) for kk, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            sc.extend(p[:, self.i_ent].tolist())
        best: Dict[int, float] = {}
        for o, s in zip(owner, sc):
            best[o] = max(best.get(o, -1.0), s)
        ok = sum(1 for i in range(len(hyps)) if best.get(i, 0.0) >= tau)
        return 100 * ok / len(hyps)


# ===========================================================================
# 1. 수집
# ===========================================================================

def collect(outdir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(outdir, "*__*.json")))
    if not files:
        sys.exit(f"[FATAL] {outdir}/*__*.json 없음")
    rows = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        for i, s in enumerate(rec["sections"]):
            src, out = s["src"], s["out"]
            sh = hangul_len(src)
            rows.append({
                "doc": rec["doc"], "mode": rec["mode"], "idx": i,
                "title": s.get("title", ""),
                "src": src, "out": out,
                "src_chars": len(src), "out_chars": len(out),
                "src_hangul": sh,
                "ratio": len(out) / max(len(src), 1),
                "meta_removed": s.get("meta_removed", 0),
                "halluc": numeric_hallucination(src, out),
                "ground": None,
            })
    print(f"[수집] 파일 {len(files)}건 → 섹션 {len(rows)}건")
    return rows


def bucket_of(src_hangul: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= src_hangul < hi:
            return i
    return len(BUCKETS) - 1


# ===========================================================================
# 2. 분포
# ===========================================================================

def show_distribution(rows: List[dict]):
    print("\n" + "=" * 96)
    print("2. 길이비 분포 — 원문 길이 구간별")
    print("=" * 96)
    print("  (길이비 = 변환문 글자수 / 원문 글자수. 1.0 이면 길이 동일)")
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        print(f"\n  [{mode}]  섹션 {len(sel)}건")
        print(f"    {'원문 구간':<12}{'n':>5}{'중앙값':>9}{'평균':>9}"
              f"{'75%':>9}{'90%':>9}{'95%':>9}{'최대':>11}")
        for bi, label in enumerate(BUCKET_LABELS):
            b = [r["ratio"] for r in sel if bucket_of(r["src_hangul"]) == bi]
            if not b:
                continue
            print(f"    {label:<12}{len(b):>5}{statistics.median(b):>9.2f}"
                  f"{statistics.fmean(b):>9.2f}{pct(b, 75):>9.2f}"
                  f"{pct(b, 90):>9.2f}{pct(b, 95):>9.2f}{max(b):>11.1f}")
        allr = [r["ratio"] for r in sel]
        print(f"    {'전체':<12}{len(allr):>5}{statistics.median(allr):>9.2f}"
              f"{statistics.fmean(allr):>9.2f}{pct(allr, 75):>9.2f}"
              f"{pct(allr, 90):>9.2f}{pct(allr, 95):>9.2f}{max(allr):>11.1f}")

    print("\n  → 중앙값과 최대값의 거리가 멀수록 소수 이상치가 분포를 끌고 있다는 뜻")


# ===========================================================================
# 3. 품질 지표와의 관계
# ===========================================================================

def show_correlation(rows: List[dict], has_nli: bool):
    print("\n" + "=" * 96)
    print("3. 팽창률이 품질 저하와 연관되는가")
    print("=" * 96)
    sec = [r for r in rows if r["mode"] == "section"]
    if len(sec) < 10:
        sec = rows

    hal = [1.0 if r["halluc"] else 0.0 for r in sec]
    rat = [r["ratio"] for r in sec]
    print(f"  표본 {len(sec)}건 (섹션 모드 기준)")
    print(f"  Spearman(길이비, 수치환각 발생)  = {spearman(rat, hal):+.3f}")
    if has_nli:
        g = [(r["ratio"], r["ground"]) for r in sec if r["ground"] is not None]
        if len(g) >= 10:
            print(f"  Spearman(길이비, 근거율)        = "
                  f"{spearman([x[0] for x in g], [x[1] for x in g]):+.3f}")
            print("    (음수여야 가설과 일치: 많이 부풀수록 근거율이 낮다)")

    print("\n  길이비 구간별 실제 품질")
    edges = [(0, 1.5), (1.5, 3), (3, 5), (5, 10), (10, 10 ** 9)]
    labels = ["~1.5배", "1.5-3배", "3-5배", "5-10배", "10배+"]
    print(f"    {'구간':<10}{'n':>5}{'환각발생':>10}{'환각수':>8}"
          + (f"{'평균 근거율':>13}" if has_nli else ""))
    for (lo, hi), lab in zip(edges, labels):
        b = [r for r in sec if lo <= r["ratio"] < hi]
        if not b:
            continue
        nh = sum(1 for r in b if r["halluc"])
        tot = sum(len(r["halluc"]) for r in b)
        line = f"    {lab:<10}{len(b):>5}{nh / len(b):>9.0%}{tot:>8}"
        if has_nli:
            gs = [r["ground"] for r in b if r["ground"] is not None]
            line += f"{statistics.fmean(gs):>13.1f}" if gs else f"{'—':>13}"
        print(line)


# ===========================================================================
# 4. 이상치 기준 후보
# ===========================================================================

def show_thresholds(rows: List[dict]) -> Dict[str, float]:
    print("\n" + "=" * 96)
    print("4. 이상치 기준 후보 — 아직 확정하지 않는다")
    print("=" * 96)
    sec = [r for r in rows if r["mode"] == "section"] or rows
    out: Dict[str, float] = {}
    print(f"    {'원문 구간':<12}{'n':>5}{'IQR 상한':>12}{'95백분위':>12}"
          f"{'초과 건수(IQR)':>16}")
    for bi, label in enumerate(BUCKET_LABELS):
        b = [r["ratio"] for r in sec if bucket_of(r["src_hangul"]) == bi]
        if len(b) < 4:
            continue
        q1, q3 = pct(b, 25), pct(b, 75)
        iqr_hi = q3 + 1.5 * (q3 - q1)
        p95 = pct(b, 95)
        n_over = sum(1 for x in b if x > iqr_hi)
        out[label] = iqr_hi
        print(f"    {label:<12}{len(b):>5}{iqr_hi:>12.2f}{p95:>12.2f}"
              f"{n_over:>16}")
    print("\n  IQR 상한 = 3사분위 + 1.5×사분위범위. 표준적인 이상치 기준이라")
    print("  '왜 그 값이냐'에 답하기 쉽다. 다만 구간마다 값이 다르므로")
    print("  단일 임계값을 쓸지 구간별로 쓸지는 5번 검토 결과를 보고 정한다.")
    return out


# ===========================================================================
# 5. 맹검 검토 시트
# ===========================================================================

def make_review(rows: List[dict], thresholds: Dict[str, float],
                n: int, seed: int, out_prefix: str):
    print("\n" + "=" * 96)
    print("5. 맹검 검토 시트")
    print("=" * 96)
    sec = [r for r in rows if r["mode"] == "section"] or rows
    rng = random.Random(seed)

    outliers, normals = [], []
    for r in sec:
        lab = BUCKET_LABELS[bucket_of(r["src_hangul"])]
        thr = thresholds.get(lab)
        if thr is not None and r["ratio"] > thr:
            outliers.append(r)
        elif 1.0 <= r["ratio"] <= 2.5:
            normals.append(r)

    n_out = min(len(outliers), int(n * 0.7))
    n_ctl = min(len(normals), n - n_out)
    outliers.sort(key=lambda r: -r["ratio"])
    picked = ([dict(r, group="이상치") for r in outliers[:n_out]]
              + [dict(r, group="대조") for r in rng.sample(normals, n_ctl)])
    rng.shuffle(picked)

    sheet = f"{out_prefix}_review.tsv"
    answer = f"{out_prefix}_review_answer.json"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        f.write("번호\t문서\t섹션\t원문\t변환문\t"
                "판정(정상/의심)\t유형(부연설명/서식재구성/내용창작)\t비고\n")
        for i, r in enumerate(picked, 1):
            src = r["src"].replace("\t", " ").replace("\n", " ")
            out = r["out"].replace("\t", " ").replace("\n", " ")
            f.write(f"{i}\t{r['doc']}\t{r['title'][:40]}\t"
                    f"{src[:600]}\t{out[:900]}\t\t\t\n")
    with open(answer, "w", encoding="utf-8") as f:
        json.dump([{"번호": i, "doc": r["doc"], "title": r["title"],
                    "group": r["group"], "ratio": round(r["ratio"], 2),
                    "src_hangul": r["src_hangul"],
                    "halluc": r["halluc"], "ground": r["ground"]}
                   for i, r in enumerate(picked, 1)],
                  f, ensure_ascii=False, indent=2)

    print(f"  이상치 {n_out}건 + 대조 {n_ctl}건 = {len(picked)}건, 순서 무작위")
    print(f"  [SAVE] {sheet}")
    print(f"  [SAVE] {answer}   ← 판정 전에는 열지 말 것")
    print("\n  판정 방법")
    print("    정상  원문에 있는 내용을 풀어 설명했거나 서식을 문장으로 바꾼 것")
    print("    의심  원문에 없는 사실·수치·조건이 생긴 것")
    print("  판정을 마친 뒤 answer 파일과 대조하면")
    print("  '팽창률로 내용창작을 가려낼 수 있는가'에 답이 나온다.")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--review-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--out", default="expansion")
    a = ap.parse_args()

    print("=" * 96)
    print("1. 수집")
    print("=" * 96)
    rows = collect(a.outdir)

    has_nli = not a.no_nli
    if has_nli:
        nli = NLI(a.gpu, a.batch)
        print("[NLI] 섹션별 근거율 계산 중...", flush=True)
        for i, r in enumerate(rows, 1):
            s_sents = Splitter.split(r["src"])
            o_sents = Splitter.split(r["out"])
            if not s_sents or not o_sents:
                continue
            r["ground"] = round(
                nli.rate(o_sents, build_chunks(s_sents), a.k, a.tau), 1)
            if i % 25 == 0:
                print(f"    {i}/{len(rows)}", flush=True)

    show_distribution(rows)
    show_correlation(rows, has_nli)
    thresholds = show_thresholds(rows)
    make_review(rows, thresholds, a.review_n, a.seed, a.out)

    with open(f"{a.out}_sections.json", "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("src", "out")}
                   for r in rows], f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_sections.json")
    print("=" * 96)


if __name__ == "__main__":
    main()
