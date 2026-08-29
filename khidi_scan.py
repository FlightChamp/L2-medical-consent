#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""khidi_scan.py — KHIDI PDF 전 페이지 텍스트량 분포 확인"""
import os, glob, re
try:
    import pymupdf as fitz
except ImportError:
    import fitz
KO = re.compile(r"[가-힣]")
for p in sorted(glob.glob(os.path.expanduser("~/shared/data/khidi/*.pdf"))):
    doc = fitz.open(p)
    counts = []
    for i in range(doc.page_count):
        try: counts.append(len(doc[i].get_text().strip()))
        except Exception: counts.append(0)
    ko_total = 0; textpages = [i for i,c in enumerate(counts) if c > 50]
    for i in textpages[:40]:
        ko_total += len(KO.findall(doc[i].get_text()))
    print("="*74)
    print(os.path.basename(p))
    print(f"  총 {doc.page_count}쪽 / 텍스트 있는 쪽 {len(textpages)}쪽 "
          f"({len(textpages)/doc.page_count*100:.0f}%)")
    print(f"  텍스트 쪽 앞 40쪽의 한글 {ko_total}자")
    if textpages:
        print(f"  텍스트 있는 쪽 번호(앞 20개): {[i+1 for i in textpages[:20]]}")
        s = " ".join(doc[textpages[0]].get_text().split())[:200]
        print(f"  샘플(p.{textpages[0]+1}): {s}")
    # 이미지 개수로 스캔본 판정
    img = sum(len(doc[i].get_images()) for i in range(min(20, doc.page_count)))
    print(f"  앞 20쪽 이미지 객체 {img}개 → {'스캔본 확정' if img >= 15 and len(textpages) < doc.page_count*0.2 else '혼합 또는 텍스트본'}")
    doc.close()
