#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blank_guard.py — ② 무결성: 빈칸·빈섹션 가드 (GPU 불필요)
=========================================================
왜 필요한가:
    doc5 6번 섹션은 원문 6자인데 변환문이 7,127자였다. 1,188배다.
    doc1 은 원문 "약 정도 소요됩니다"(빈칸)를 "약 1시간"으로 채웠다.
    두 사건의 뿌리는 같다 — 원문에 내용이 없는 자리를 모델이 지어낸다.

    지금까지 수치'보존'만 쟀고(99%로 양호) 수치'환각'은 재지 않았다.
    보존율이 높아도 없던 값이 생기면 동의서로서는 치명적이다.

이 모듈이 하는 일:
    1) 진단   원문 13건의 빈칸·빈섹션을 유형별로 집계한다
    2) 측정   이미 생성된 outputs/*.json 으로 수치환각을 잰다 (GPU 불필요)
    3) 가드   변환 전 통과 판정 + 변환 후 환각 판정 함수를 제공한다
    4) 시뮬   가드를 적용했다면 무엇이 달라졌을지 계산한다

가드 규칙:
    · 본문 한글이 MIN_HANGUL 자 미만인 섹션은 모델을 태우지 않고 원문 그대로 통과
      (doc5 6번 섹션 6자 → 7,127자 폭주가 원천 차단된다)
    · 원문의 빈칸은 [미기재] 로 치환해 모델에게 명시한다
    · 변환 후 원문에 없던 단위 숫자가 생기면 환각으로 표시한다

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python blank_guard.py                 # 진단 + 측정 + 시뮬
    python blank_guard.py --min-hangul 30
    python blank_guard.py --show doc5     # 특정 문서 상세
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

DOCDIR = os.path.expanduser("~/이윤우/docs")
OUTDIR = os.path.expanduser("~/이윤우/outputs")

MIN_HANGUL = 25          # 이 미만이면 변환하지 않고 원문 통과
BLANK_TOKEN = "[미기재]"


# ===========================================================================
# 탐지 규칙 (test_blank.py 로 12개 사례 검증 완료)
# ===========================================================================

BLANK = {
    "빈괄호": re.compile(r"[（(\[〔]\s*[)）\]〕]"),
    "밑줄점선": re.compile(r"[_＿]{2,}|\.{4,}|·{4,}|─{2,}"),
    "수치누락": re.compile(
        r"약\s+(?=정도|가량|이내|이상|이하|동안|쯤)"
        r"|(?<![\d])\s(?=(주|일|개월|시간|분|년|%|cc|ml|mg|회)\s*(정도|가량|이내|동안))"),
    "명사누락": re.compile(r"\s(을|를|이|가|의|은|는|에|와|과)\s"),
    "빈필드": re.compile(r"[:：]\s*$"),
}

NUM_UNIT = re.compile(
    r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번|세)")

SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")
DROP = re.compile(
    r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
    r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
    r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")


def clean(t) -> str:
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


def extract_sections(raw: str) -> List[Tuple[str, str]]:
    lines = [re.sub(r"\s+", " ", l).strip() for l in raw.split("\n")]
    secs: List[Tuple[str, str]] = []
    cur: Optional[str] = None
    buf: List[str] = []
    for l in lines:
        if not l:
            continue
        m = SEC.match(l)
        if m:
            if cur and buf:
                secs.append((cur, " ".join(buf)))
            cur, buf = f"{m.group(1)}. {m.group(2)}", []
            continue
        if cur is None or DROP.search(l):
            continue
        if len(re.findall(r"[가-힣]", l)) < 5:
            continue
        buf.append(l)
    if cur and buf:
        secs.append((cur, " ".join(buf)))
    return secs


# ===========================================================================
# 가드 함수 — 파이프라인에 그대로 옮겨 쓸 수 있다
# ===========================================================================

def hangul_len(t: str) -> int:
    return len(re.findall(r"[가-힣]", t))


def find_blanks(text: str) -> Dict[str, List[str]]:
    out = {}
    for name, pat in BLANK.items():
        hits = [m.group(0) for m in pat.finditer(text)]
        if hits:
            out[name] = hits
    return out


def should_convert(text: str, min_hangul: int = MIN_HANGUL) -> Tuple[bool, str]:
    """(변환할 것인가, 사유). False 면 원문을 그대로 통과시킨다."""
    h = hangul_len(text)
    if h < min_hangul:
        return (False, f"본문 한글 {h}자 < {min_hangul}자")
    return (True, "")


def mark_blanks(text: str) -> Tuple[str, int]:
    """빈칸을 [미기재] 로 치환해 모델에게 명시한다."""
    n = 0
    t = text
    for name in ("빈괄호", "밑줄점선"):
        t, c = BLANK[name].subn(BLANK_TOKEN, t)
        n += c
    # '약 정도' 처럼 수치가 빠진 자리
    t, c = re.subn(r"약\s+(?=정도|가량|이내|이상|이하|동안|쯤)",
                   f"약 {BLANK_TOKEN} ", t)
    n += c
    return (t, n)


def numeric_hallucination(src: str, out: str) -> List[str]:
    """변환문에만 있는 단위 숫자 = 원문에 없던 값."""
    s = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(src)}
    o = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(out)}
    return sorted(o - s)


# ===========================================================================

def stage_diagnose(docdir: str, min_hangul: int) -> dict:
    print("=" * 92)
    print("1. 원문 진단 — 빈칸과 빈 섹션이 얼마나 있는가")
    print("=" * 92)
    paths = [p for p in sorted(glob.glob(os.path.join(docdir, "*.txt")))
             if not os.path.basename(p).startswith("syn")]
    if not paths:
        sys.exit(f"[FATAL] {docdir}/*.txt 없음")

    total = Counter()
    per_doc = {}
    short_secs: List[Tuple[str, str, int]] = []
    print(f"  {'문서':<7}{'섹션':>5}{'짧은섹션':>9}{'빈괄호':>7}{'밑줄':>6}"
          f"{'수치누락':>9}{'명사누락':>9}{'빈필드':>7}")
    for p in paths:
        doc = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        secs = extract_sections(raw)
        hits = find_blanks(clean(raw))
        cnt = {k: len(v) for k, v in hits.items()}
        n_short = 0
        for title, body in secs:
            ok, _ = should_convert(body, min_hangul)
            if not ok:
                n_short += 1
                short_secs.append((doc, title, hangul_len(body)))
        per_doc[doc] = {"n_sec": len(secs), "n_short": n_short, **cnt}
        for k, v in cnt.items():
            total[k] += v
        total["짧은섹션"] += n_short
        total["섹션"] += len(secs)
        print(f"  {doc:<7}{len(secs):>5}{n_short:>9}"
              f"{cnt.get('빈괄호', 0):>7}{cnt.get('밑줄점선', 0):>6}"
              f"{cnt.get('수치누락', 0):>9}{cnt.get('명사누락', 0):>9}"
              f"{cnt.get('빈필드', 0):>7}")
    print(f"  {'합계':<7}{total['섹션']:>5}{total['짧은섹션']:>9}"
          f"{total['빈괄호']:>7}{total['밑줄점선']:>6}"
          f"{total['수치누락']:>9}{total['명사누락']:>9}{total['빈필드']:>7}")

    print(f"\n  변환 대상에서 제외될 짧은 섹션 {len(short_secs)}건 "
          f"(한글 {min_hangul}자 미만)")
    for doc, title, h in sorted(short_secs, key=lambda x: x[2])[:15]:
        print(f"    {doc:<7}{title[:38]:<40} 한글 {h}자")
    return {"per_doc": per_doc, "total": dict(total),
            "short_sections": short_secs}


def stage_measure(outdir: str) -> List[dict]:
    print("\n" + "=" * 92)
    print("2. 수치환각 측정 — 원문에 없던 숫자가 변환문에 생겼는가")
    print("=" * 92)
    files = sorted(glob.glob(os.path.join(outdir, "*__*.json")))
    if not files:
        print("  outputs/*.json 없음 — 건너뜁니다")
        return []
    rows = []
    print(f"  {'문서':<7}{'방식':<9}{'섹션':>5}{'원문수치':>9}{'환각':>6}"
          f"  지어낸 값")
    for p in files:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        halluc_all: List[str] = []
        per_sec = []
        for s in rec["sections"]:
            h = numeric_hallucination(s["src"], s["out"])
            if h:
                per_sec.append((s["title"], h))
            halluc_all.extend(h)
        src = " ".join(s["src"] for s in rec["sections"])
        n_src = len({f"{m.group(1)}{m.group(2)}"
                     for m in NUM_UNIT.finditer(src)})
        uniq = sorted(set(halluc_all))
        rows.append({"doc": rec["doc"], "mode": rec["mode"],
                     "n_src_nums": n_src, "n_halluc": len(uniq),
                     "halluc": uniq, "per_sec": per_sec})
        print(f"  {rec['doc']:<7}{rec['mode']:<9}{len(rec['sections']):>5}"
              f"{n_src:>9}{len(uniq):>6}  {', '.join(uniq[:8])}")

    print("\n  방식별 합계")
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        tot = sum(r["n_halluc"] for r in sel)
        docs_with = sum(1 for r in sel if r["n_halluc"] > 0)
        print(f"    {mode:<10} 환각 수치 {tot:>3}개 / {len(sel)}건 중 "
              f"{docs_with}건에서 발생")
    return rows


def stage_simulate(diag: dict, outdir: str, min_hangul: int):
    print("\n" + "=" * 92)
    print("3. 가드 효과 — 짧은 섹션을 통과시켰다면")
    print("=" * 92)
    files = sorted(glob.glob(os.path.join(outdir, "*__section.json")))
    if not files:
        print("  섹션 모드 결과 없음 — 건너뜁니다")
        return
    saved_chars = 0
    saved_halluc = 0
    affected = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        for s in rec["sections"]:
            ok, why = should_convert(s["src"], min_hangul)
            if ok:
                continue
            grew = len(s["out"]) - len(s["src"])
            h = numeric_hallucination(s["src"], s["out"])
            saved_chars += max(grew, 0)
            saved_halluc += len(h)
            if grew > 200 or h:
                affected.append((rec["doc"], s["title"], len(s["src"]),
                                 len(s["out"]), why, h))
    print(f"  가드 적용 시 막았을 생성량 {saved_chars:,}자 / 수치환각 {saved_halluc}개\n")
    if affected:
        print(f"  {'문서':<7}{'섹션':<34}{'원문':>6}{'변환':>7}  사유")
        for doc, title, si, so, why, h in sorted(
                affected, key=lambda x: -(x[3] - x[2]))[:12]:
            mark = f"  환각 {h}" if h else ""
            print(f"  {doc:<7}{title[:32]:<34}{si:>6}{so:>7}  {why}{mark}")
    else:
        print("  가드가 막았을 사례가 없습니다.")


def show(doc: str, docdir: str, outdir: str, min_hangul: int):
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        sys.exit(f"{p} 없음")
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    print("=" * 92)
    print(f"{doc} 섹션별 상태")
    print("=" * 92)
    print(f"  {'섹션':<38}{'한글':>6}{'판정':>8}  빈칸")
    for title, body in secs:
        ok, why = should_convert(body, min_hangul)
        blanks = find_blanks(body)
        marked, n = mark_blanks(body)
        print(f"  {title[:36]:<38}{hangul_len(body):>6}"
              f"{'변환' if ok else '통과':>8}  "
              f"{ {k: len(v) for k, v in blanks.items()} or '없음'}")
        if n:
            print(f"      치환 후: {marked[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--min-hangul", type=int, default=MIN_HANGUL)
    ap.add_argument("--show", default=None)
    ap.add_argument("--out", default="blank_guard")
    a = ap.parse_args()

    if a.show:
        show(a.show, a.docdir, a.outdir, a.min_hangul)
        return

    diag = stage_diagnose(a.docdir, a.min_hangul)
    rows = stage_measure(a.outdir)
    stage_simulate(diag, a.outdir, a.min_hangul)

    with open(f"{a.out}_report.json", "w", encoding="utf-8") as f:
        json.dump({"diagnose": {k: v for k, v in diag.items()
                                if k != "short_sections"},
                   "short_sections": [list(x) for x in diag["short_sections"]],
                   "hallucination": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_report.json")


if __name__ == "__main__":
    main()
