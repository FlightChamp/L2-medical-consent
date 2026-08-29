#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preflight.py — real_run.py의 섹션 추출 로직을 모델 없이 재현해 입력 품질 점검"""
import os, re, glob, json
try:
    import pymupdf as fitz
except ImportError:
    import fitz

DOCS = os.path.expanduser("~/이윤우/docs")
OUT  = os.path.join(DOCS, "_preflight")
os.makedirs(OUT, exist_ok=True)

SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")

# real_run.py 현재 버전
DROP_OLD = re.compile(
 r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
 r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
 r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")

# 제안 수정본: 무/유/미상 서식과 반복 체크박스 패턴 추가
DROP_NEW = re.compile(
 r"유\s*무|무\s*유|미\s*상|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|"
 r"진료과|주치의|시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
 r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태|"
 r"(무|유|미상)(\s+(무|유|미상)){2,}")

def extract_sections(raw, DROP):
    lines = [re.sub(r"\s+", " ", l).strip() for l in raw.split("\n")]
    secs, cur, buf = [], None, []
    for l in lines:
        if not l: continue
        m = SEC.match(l)
        if m:
            if cur and buf: secs.append((cur, buf))
            cur, buf = f"{m.group(1)}. {m.group(2)}", []
            continue
        if cur is None: continue
        if DROP.search(l): continue
        if len(re.findall(r"[가-힣]", l)) < 5: continue
        buf.append(l)
    if cur and buf: secs.append((cur, buf))
    return secs

# 체크리스트 잔재로 의심되는 줄
NOISE = re.compile(r"(무|유|미상)\s+(무|유|미상)|흡연\s*여부|출혈\s*소인|심근경색|기침,\s*가래")

rows = []
print(f"{'파일':<10}{'섹션':>5}{'줄(현재)':>9}{'줄(수정)':>9}{'노이즈(현재)':>13}{'노이즈(수정)':>13}")
print("-" * 62)
for path in sorted(glob.glob(os.path.join(DOCS, "*.pdf"))):
    base = os.path.basename(path); stem = base[:-4]
    doc = fitz.open(path)
    if not doc.is_pdf:
        doc.close(); continue
    raw = "\n".join(p.get_text() for p in doc); doc.close()

    old = extract_sections(raw, DROP_OLD)
    new = extract_sections(raw, DROP_NEW)
    n_old = sum(len(b) for _, b in old)
    n_new = sum(len(b) for _, b in new)
    noise_old = sum(1 for _, b in old for l in b if NOISE.search(l))
    noise_new = sum(1 for _, b in new for l in b if NOISE.search(l))
    print(f"{stem:<10}{len(new):>5}{n_old:>9}{n_new:>9}{noise_old:>13}{noise_new:>13}")

    with open(os.path.join(OUT, f"{stem}_input.txt"), "w", encoding="utf-8") as f:
        for title, buf in new:
            f.write(f"\n### {title}\n")
            for l in buf: f.write(l + "\n")
    if noise_old:
        with open(os.path.join(OUT, f"{stem}_noise.txt"), "w", encoding="utf-8") as f:
            for title, buf in old:
                for l in buf:
                    if NOISE.search(l): f.write(f"[{title}] {l}\n")
    rows.append({"file": base, "sections": len(new), "lines_old": n_old,
                 "lines_new": n_new, "noise_old": noise_old, "noise_new": noise_new})

# doc1 갑상선 문장이 입력에 살아 있는지 확인
p = os.path.join(OUT, "doc1_input.txt")
if os.path.exists(p):
    txt = open(p, encoding="utf-8").read()
    print("\n[doc1 오염 문장 포함 여부]", "포함됨 ✅" if "갑상선" in txt else "누락됨 ❌")
    for l in txt.split("\n"):
        if "갑상선" in l: print("   →", l)

with open(os.path.join(OUT, "preflight.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\n[저장]", OUT)
