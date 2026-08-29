#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_syn.py — compare2.py의 합성 동의서를 txt로 추출하고 real_run 호환성 진단"""
import os, re

HOME = os.path.expanduser("~/이윤우")
DOCS = os.path.join(HOME, "docs")
src  = open(os.path.join(HOME, "compare2.py"), encoding="utf-8").read()

blocks = re.findall(r'"text"\s*:\s*"""(.*?)"""', src, re.S)
print(f"[추출] 합성 문서 {len(blocks)}건\n")

SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")
DROP = re.compile(
 r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
 r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
 r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")

def extract_sections(raw):
    lines = [re.sub(r"\s+"," ",l).strip() for l in raw.split("\n")]
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

for i, b in enumerate(blocks, 1):
    body  = b.strip()
    title = " ".join(body.split("\n")[0].split())
    name  = f"syn{i}"
    path  = os.path.join(DOCS, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")

    secs = extract_sections(body)
    SRC  = "\n".join(f"{h}\n" + "\n".join(bb) for h, bb in secs)
    ok   = len(SRC) >= 200
    print(f"── {name}.txt  «{title}»")
    print(f"   원문 {len(body)}자 / 섹션 {len(secs)}개 / 정제 후 {len(SRC)}자")
    print(f"   섹션: {', '.join(h for h,_ in secs) if secs else '(번호 섹션 없음)'}")
    print(f"   real_run.py 호환: {'✅ 처리 가능' if ok else '❌ 200자 미만 → 건너뜀'}")
    print()

print(f"[저장] {DOCS}/syn1.txt ~ syn{len(blocks)}.txt")
