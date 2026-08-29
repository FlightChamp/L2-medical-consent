#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rescan.py — doc1 오염 범위 전문 추출 + 보일러플레이트 제거한 교차오염 재스캔"""
import os, glob, json, re
try:
    import pymupdf as fitz
except ImportError:
    import fitz

DOCS = os.path.expanduser("~/이윤우/docs")
OUT  = os.path.join(DOCS, "_check")
os.makedirs(OUT, exist_ok=True)

# 모든 동의서 공통 서식 → 도메인 판정에서 제외
BOILER = ["심장질환", "신장질환", "호흡기질환", "뇌혈관질환", "고.저혈압", "당뇨병",
          "출혈소인", "기도이상", "복용약물", "특이체질", "알레르", "흡연여부",
          "임신여부", "과거병력", "마약사고"]

# 바로 그 수술에만 등장하는 특이 용어만 남김
ORGANS = {
    "갑상선":   ["갑상선", "갑상샘", "성대마비", "저칼슘혈증"],
    "정형외과": ["골절", "인공관절", "관절치환", "십자인대", "견봉", "금속물 고정", "반월상"],
    "척추":     ["척추", "신경 차단", "추간판", "디스크"],
    "복부":     ["담낭", "충수", "맹장", "탈장", "위절제", "대장절제"],
    "대장항문": ["치핵", "항문", "농양", "치루"],
    "비뇨":     ["전립선", "요관", "방광", "신장절제"],
    "산부인과": ["자궁", "난소", "제왕절개"],
    "유방":     ["유방"],
    "호흡기":   ["기흉", "흉관", "폐엽"],
    "안과":     ["백내장", "수정체", "망막"],
    "심혈관":   ["관상동맥", "스텐트", "심근경색"],
}

def domains(text):
    hits = {}
    for k, ws in ORGANS.items():
        n = sum(text.count(w) for w in ws)
        if n: hits[k] = n
    return hits

print("=" * 78)
print("[A] doc1.pdf — '2. 수술의 목적 및 효과' 섹션 전문 (자르지 않음)")
print("=" * 78)
d1 = fitz.open(os.path.join(DOCS, "doc1.pdf"))
pg = d1[0]
for b in sorted(pg.get_text("blocks"), key=lambda x: (round(x[1]), x[0])):
    y0, txt = b[1], b[4]
    if 480 <= y0 <= 700 and txt.strip():
        print(f"--- 블록 y={y0:.1f} / 길이 {len(txt.strip())}자 ---")
        print(txt.strip())
        print()
# 해당 구간 이미지로 저장
clip = fitz.Rect(pg.rect.x0, 470, pg.rect.x1, 710)
pg.get_pixmap(dpi=300, clip=clip).save(os.path.join(OUT, "doc1_p1_section2.png"))
print(f"[저장] _check/doc1_p1_section2.png  (해당 구간 확대 이미지)")

# 갑상선 문장이 정확히 몇 줄에 걸쳐 있는지
rects = pg.search_for("갑상선")
if rects:
    y_start = rects[0].y0
    lines = []
    for b in pg.get_text("blocks"):
        if b[1] >= y_start - 2 and b[1] < 696:
            lines.append(b[4].strip())
    joined = " ".join(" ".join(lines).split())
    print(f"\n[오염 추정 구간 총 길이] {len(joined)}자")
    print(f"[전문] {joined}")
d1.close()

print()
print("=" * 78)
print("[B] 교차오염 재스캔 — 공통 서식 제거 후")
print("=" * 78)
rows = []
for path in sorted(glob.glob(os.path.join(DOCS, "*.pdf"))):
    base = os.path.basename(path)
    doc = fitz.open(path)
    if not doc.is_pdf:
        doc.close(); continue
    full = "\n".join(p.get_text() for p in doc)
    doc.close()
    clean = full
    for w in BOILER:
        clean = clean.replace(w, "")
    title = " ".join(full.strip().split("\n")[0].split())[:40]
    t_dom = set(domains(title).keys())
    b_dom = domains(clean)
    extra = {k: v for k, v in b_dom.items() if k not in t_dom} if t_dom else {}
    status = "오염 의심" if extra else "정상"
    print(f"{base:<12} [{status}] 제목='{title}'")
    print(f"{'':<12} 제목도메인={sorted(t_dom) or '판정불가'}  본문={b_dom}")
    if extra:
        for k in extra:
            for w in ORGANS[k]:
                for m in re.finditer(re.escape(w), clean):
                    s = " ".join(clean[max(0, m.start()-40):m.start()+60].split())
                    print(f"{'':<12}   ⚠ [{k}] …{s}…")
                    break
    rows.append({"file": base, "title": title, "title_domain": sorted(t_dom),
                 "body_domain": b_dom, "extra": extra, "status": status})

with open(os.path.join(OUT, "rescan.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\n[저장]", os.path.join(OUT, "rescan.json"))
