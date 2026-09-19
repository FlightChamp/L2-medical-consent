#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_vocab_grades.py — 2017 표준 어휘 목록 → 등급 사전
=========================================================
출처:
    국립국어원, 「2017년 국제 통용 한국어 표준 교육과정 적용 연구(4단계)」
    김중섭 외 13명(2017). 어휘·문법 등급 목록.
    (고승연(2025)이 이독성 공식 산출에 사용한 것과 동일한 목록)

원본 구조 ('어휘' 시트):
    전체 번호 | 등급별 번호 | 등급 | 어휘 | 품사 | 길잡이말 | ... | 등급
    등급은 1급~6급. 총 10,635 항목.
      1급   735    2급 1,100    3급 1,655
      4급 2,200    5급 2,365    6급 2,580

표제어 정규화:
    가격02          → 가격          동형어 번호 제거
    가까이01/가까이02 → 가까이        슬래시 이형태 분리
    -가02           → 가            접사 하이픈 제거
    은1             → 은

    정규화 후 고유 표제어 10,013개.

산출물:
    vocab_grades.json
      {"source": ..., "by_word": {"가격": 1, ...}, "counts": {...}}
      값은 해당 표제어의 **최저 등급**이다.
      (같은 표제어가 여러 등급에 나오면 쉬운 쪽으로 잡아 난이도를
       과대평가하지 않는다)

사용법:
    python build_vocab_grades.py --xlsx <경로>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Set

DEFAULT_OUT = os.path.expanduser("~/이윤우/vocab_grades.json")
GRADE_RE = re.compile(r"^([1-6])급$")


def lemma_variants(w: str) -> Set[str]:
    """표제어 표기를 실제 어휘 형태로 정규화한다."""
    out: Set[str] = set()
    for part in str(w).split("/"):
        p = part.strip()
        p = re.sub(r"\d+$", "", p)      # 동형어 번호
        p = p.strip("-").strip()        # 접사 하이픈
        if p and re.search(r"[가-힣]", p):
            out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    if not os.path.exists(a.xlsx):
        sys.exit(f"[FATAL] {a.xlsx} 없음")
    try:
        from openpyxl import load_workbook
    except ImportError:
        sys.exit("[FATAL] openpyxl 필요: pip install openpyxl --break-system-packages")

    wb = load_workbook(a.xlsx, read_only=True)
    if "어휘" not in wb.sheetnames:
        sys.exit(f"[FATAL] '어휘' 시트 없음. 시트 목록: {wb.sheetnames}")
    ws = wb["어휘"]

    by_word: Dict[str, int] = {}
    raw_grade = Counter()
    n_rows = n_skip = 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or len(r) < 4:
            continue
        g, w = r[2], r[3]
        if not g or not w:
            continue
        m = GRADE_RE.match(str(g).strip())
        if not m:
            n_skip += 1
            continue
        lv = int(m.group(1))
        n_rows += 1
        raw_grade[lv] += 1
        for v in lemma_variants(w):
            # 같은 표제어가 여러 등급에 있으면 더 쉬운(낮은) 등급을 택한다
            if v not in by_word or lv < by_word[v]:
                by_word[v] = lv

    if not by_word:
        sys.exit("[FATAL] 어휘를 추출하지 못했습니다")

    lv_count = Counter(by_word.values())
    payload = {
        "source": ("국립국어원 2017년 국제 통용 한국어 표준 교육과정 적용 연구(4단계), "
                   "김중섭 외 13명(2017) 어휘 등급 목록"),
        "note": ("값은 표제어의 최저 등급. 고승연(2025) 이독성 공식에서 "
                 "'중급 이상'은 3급~6급을 뜻한다."),
        "xlsx": os.path.basename(a.xlsx),
        "n_rows": n_rows,
        "n_lemmas": len(by_word),
        "counts_raw": {f"{k}급": v for k, v in sorted(raw_grade.items())},
        "counts_lemma": {f"{k}급": v for k, v in sorted(lv_count.items())},
        "by_word": by_word,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print("=" * 76)
    print("2017 표준 어휘 목록 → 등급 사전")
    print("=" * 76)
    print(f"  원본 항목        {n_rows:,}")
    if n_skip:
        print(f"  등급 형식 불일치  {n_skip:,} (건너뜀)")
    print(f"  고유 표제어      {len(by_word):,}")
    print(f"\n  {'등급':<8}{'원본':>10}{'표제어':>10}")
    for k in sorted(raw_grade):
        print(f"  {k}급{'':<5}{raw_grade[k]:>10,}{lv_count.get(k,0):>10,}")
    초 = sum(v for k, v in lv_count.items() if k <= 2)
    중 = sum(v for k, v in lv_count.items() if k >= 3)
    print(f"\n  초급 (1~2급)     {초:>10,}")
    print(f"  중급 이상 (3~6급) {중:>10,}   ← 고승연(2025) 공식의 분자")
    print(f"\n  예시")
    for w in ["가게", "가격", "수술", "합병증", "갑상선", "동의"]:
        g = by_word.get(w)
        print(f"    {w:<8}{('%d급' % g) if g else '목록 없음'}")
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
