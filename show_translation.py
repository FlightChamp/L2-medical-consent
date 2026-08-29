#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_translation.py — 번역문에서 오염 문장 발췌 + 완전성 점검"""
import os, re, json
HOME = os.path.expanduser("~/이윤우")
d = json.load(open(os.path.join(HOME, "translate_demo.json"), encoding="utf-8"))
rep = {}
for f in ["reportA.json","reportB.json","reportC.json"]:
    p=os.path.join(HOME,f)
    if os.path.exists(p): rep.update(json.load(open(p,encoding="utf-8")))
SRC = rep["doc1"]["원문"]

MARK = {"English":["thyroid"], "Chinese":["甲状腺"], "Vietnamese":["tuyến giáp","giáp"]}
# 원문 섹션 헤더가 번역문에 몇 개나 남았는지로 완전성 점검
SEC_KO = re.findall(r"^\s*(\d+)\s*[.．]\s*(.+)$", SRC, re.M)

print("="*78); print("[1] 오염 문장이 번역된 모습"); print("="*78)
print("원문: 갑상선은 목 앞부분에 위치한 나비모양의 기관으로 정상인에서는 눈으로 보이지도 않고 손으로 만져지")
print("      (골절 수술 동의서에 잘못 인쇄된 문장, 문장 끝이 잘려 있음)\n")
for lang, marks in MARK.items():
    out = d[lang]["출력"]; low = out.lower()
    i = -1
    for m in marks:
        i = low.find(m.lower())
        if i >= 0: break
    print(f"── {lang}")
    if i < 0:
        print("   (미검출)\n"); continue
    seg = " ".join(out[max(0,i-160):i+260].split())
    print(f"   …{seg}…\n")

print("="*78); print("[2] 번역 완전성 점검"); print("="*78)
print(f"원문 {len(SRC)}자 / 번호 섹션 {len(SEC_KO)}개\n")
print(f"{'언어':<12}{'출력자수':>9}{'원문대비':>9}{'섹션번호':>9}{'문장끝':>8}")
for lang, r in d.items():
    out = r["출력"]
    secs = len(re.findall(r"(?m)^\s*#{0,6}\s*\d+[.)]\s", out))
    tail = "정상" if re.search(r"[.。!?]\s*$", out.strip()) else "중단"
    print(f"{lang:<12}{len(out):>9}{len(out)/len(SRC)*100:>8.0f}%{secs:>9}{tail:>8}")
print("\n※ 중국어는 한자 특성상 글자 수가 원래 적음. 섹션 번호 개수로 누락 여부를 판단할 것")
print("※ 섹션 번호가 원문(10개)보다 크게 적으면 뒷부분이 빠진 것")
