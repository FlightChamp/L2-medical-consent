#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_docs.py — doc6 정체 확인 / doc1 섹션 구조 / 전체 문서 교차오염 스캔"""
import os, re, glob, json
try:
    import pymupdf as fitz
except ImportError:
    import fitz

DOCS = os.path.expanduser("~/이윤우/docs")
OUT  = os.path.join(DOCS, "_check")
os.makedirs(OUT, exist_ok=True)

# 장기/수술 계열 키워드 — 문서 간 교차오염 탐지용
ORGANS = {
    "갑상선": ["갑상선", "갑상샘", "성대마비", "저칼슘혈증"],
    "정형외과": ["골절", "인공관절", "관절치환", "금속고정", "핀고정"],
    "복부": ["담낭", "충수", "맹장", "탈장", "위절제", "대장"],
    "비뇨": ["전립선", "요관", "신장", "방광"],
    "산부인과": ["자궁", "난소", "제왕절개"],
    "안과": ["백내장", "수정체", "망막"],
    "심혈관": ["관상동맥", "스텐트", "심장"],
}

print("=" * 78)
print("[A] doc6.pdf 실제 내용 (HTML5로 판별된 파일)")
print("=" * 78)
p6 = os.path.join(DOCS, "doc6.pdf")
if os.path.exists(p6):
    raw = open(p6, "rb").read()
    print(f"파일 크기: {len(raw)} bytes / 첫 4바이트: {raw[:4]!r}")
    print("-" * 78)
    try:
        print(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        print("디코딩 실패:", e)
else:
    print("doc6.pdf 없음")

print()
print("=" * 78)
print("[B] doc1.pdf 1페이지 블록 구조 (읽기 순서 / y좌표)")
print("=" * 78)
d1 = fitz.open(os.path.join(DOCS, "doc1.pdf"))
pg = d1[0]
blocks = sorted(pg.get_text("blocks"), key=lambda b: (round(b[1]), b[0]))
lines_out = []
for x0, y0, x1, y1, txt, bno, btype in blocks:
    t = " ".join(txt.split())
    if not t:
        continue
    mark = "  <<<< 갑상선 문장" if "갑상선" in t else ""
    line = f"[y={y0:6.1f} x={x0:6.1f}] {t[:150]}{mark}"
    print(line); lines_out.append(line)
with open(os.path.join(OUT, "doc1_p1_blocks.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out))
d1.close()

print()
print("=" * 78)
print("[C] 전체 문서 교차오염 스캔 — 문서별 등장 장기 계열")
print("=" * 78)
rows = []
for path in sorted(glob.glob(os.path.join(DOCS, "*.pdf"))):
    base = os.path.basename(path)
    try:
        doc = fitz.open(path)
        if not doc.is_pdf:
            print(f"{base:<12} (PDF 아님 — 건너뜀)"); doc.close(); continue
        full = "\n".join(p.get_text() for p in doc)
        head = " ".join(full[:120].split())
        hits = {k: sum(full.count(w) for w in ws) for k, ws in ORGANS.items()}
        hits = {k: v for k, v in hits.items() if v > 0}
        flag = "  ⚠ 2계열 이상" if len(hits) >= 2 else ""
        print(f"{base:<12} {hits}{flag}")
        print(f"{'':<12} 서두: {head[:100]}")
        rows.append({"file": base, "head": head, "organ_hits": hits,
                     "mixed": len(hits) >= 2})
        doc.close()
    except Exception as e:
        print(f"{base:<12} 실패: {e}")

with open(os.path.join(OUT, "crosscheck.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\n[저장]", os.path.join(OUT, "doc1_p1_blocks.txt"))
print("[저장]", os.path.join(OUT, "crosscheck.json"))
