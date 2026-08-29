#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_khidi.py — 업로드 자료 구조 파악: 대역(참조 번역) 구조인지 판별"""
import os, glob, re
try:
    import pymupdf as fitz
except ImportError:
    import fitz

DIRS = [os.path.expanduser("~/shared/data")]
LANG = {
    "영어":     re.compile(r"[A-Za-z]{4,}"),
    "중국어":   re.compile(r"[\u4e00-\u9fff]"),
    "일본어":   re.compile(r"[\u3040-\u30ff]"),
    "베트남어": re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"),
    "러시아어": re.compile(r"[\u0400-\u04ff]"),
    "아랍어":   re.compile(r"[\u0600-\u06ff]"),
    "태국어":   re.compile(r"[\u0e00-\u0e7f]"),
    "몽골어":   re.compile(r"[\u1800-\u18af]"),
}
KO = re.compile(r"[가-힣]")

files = []
for d in DIRS:
    for ext in ("*.pdf","*.txt","*.csv","*.xlsx","*.json","*.hwp","*.docx"):
        files += glob.glob(os.path.join(d, "**", ext), recursive=True)
files = sorted(set(files))
if not files:
    raise SystemExit("[중단] ~/shared/data 에 파일이 없습니다")

print(f"[발견] {len(files)}개 파일\n")
for p in files:
    name = os.path.relpath(p, os.path.expanduser("~"))
    kb = os.path.getsize(p)/1024
    ext = os.path.splitext(p)[1].lower()
    print("="*78)
    print(f"{name}\n  ({kb/1024:.1f}MB)")
    if ext != ".pdf":
        print(f"  형식 {ext} — 별도 처리 필요")
        continue
    try:
        doc = fitz.open(p)
        pages = doc.page_count
        # 앞·중간·뒤에서 표본 추출
        idx = sorted(set([0,1,2,3, pages//3, pages//2, pages*2//3, pages-2, pages-1]))
        idx = [i for i in idx if 0 <= i < pages]
        txt = "\n".join(doc[i].get_text() for i in idx)
        doc.close()
    except Exception as e:
        print(f"  열기 실패: {e}"); continue
    ko = len(KO.findall(txt))
    print(f"  {pages}쪽 / 표본 {len(idx)}쪽에서 한글 {ko}자")
    if ko < 30:
        print("  ⚠ 텍스트 레이어 거의 없음 → 스캔본 가능성, OCR 필요")
        continue
    hits = {k: len(r.findall(txt)) for k, r in LANG.items()}
    hits = {k: v for k, v in hits.items() if v > 20}
    print(f"  동시 등장 언어: {hits if hits else '한국어 단독'}")
    both = 0; samples = []
    for l in txt.split("\n"):
        l2 = " ".join(l.split())
        if KO.search(l2) and any(r.search(l2) for r in LANG.values()):
            both += 1
            if len(l2) > 10 and len(samples) < 6: samples.append(l2)
    print(f"  한 줄에 한국어+외국어 동시 포함: {both}줄")
    if both >= 20:
        print("  ✅ 대역 구조 강함 — 문장 쌍 자동 추출 가능")
    elif hits:
        print("  △ 언어별 구역 분리로 추정 — 페이지 단위 정렬 필요")
    for s in samples:
        print(f"    · {s[:130]}")
