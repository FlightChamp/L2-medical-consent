#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_mistrans.py — 미검출 용어의 원문·번역문 대조 발췌"""
import os, re, json
HOME = os.path.expanduser("~/이윤우")
RES = json.load(open(os.path.join(HOME,"translate_eval.json"), encoding="utf-8"))
LANG = {"en":"영어","zh":"중국어","ja":"일본어"}

rep = {}
for f in ["reportA.json","reportB.json","reportC.json"]:
    p=os.path.join(HOME,f)
    if os.path.exists(p): rep.update(json.load(open(p,encoding="utf-8")))

def ko_sent(src, term):
    """원문에서 용어가 든 문장 하나"""
    for s in re.split(r"(?<=[.!?])\s+|\n+", src):
        if term in s: return " ".join(s.split())
    return ""

def near(text, kws, span=220):
    """번역문에서 문맥 키워드 주변 발췌"""
    low = text.lower()
    for k in kws:
        i = low.find(k.lower())
        if i >= 0:
            return " ".join(text[max(0,i-90):i+span].split())
    return ""

# 영어 문맥 앵커: 용어와 같은 문장에 흔히 오는 단어
ANCHOR = {"갑상선":["butterfly","neck","thyro","esophag","tuyến","甲状","食道","首"],
          "합병증":["complicat","并发","合併"], "후유증":["afteref","sequel","后遗","後遺"],
          "수혈":["transfus","输血","輸血"], "골절":["fractur","骨折"],
          "기흉":["pneumothorax","气胸"], "치핵":["hemorrhoid","痔"],
          "인공관절":["artificial","prosthe","人工关节"], "마취":["anesth","麻醉","麻酔"]}

n = 0
for doc, d in sorted(RES.items(), key=lambda x:(len(x[0]),x[0])):
    SRC = rep[doc]["원문"]
    for code, v in d["언어"].items():
        if not v["미검출"]: continue
        for term, corr in v["미검출"]:
            n += 1
            print("="*78)
            print(f"[{doc}] {LANG[code]}  ·  용어 '{term}' → 기대 대응어 '{corr}' 미검출")
            ks = ko_sent(SRC, term)
            print(f"  원문   : {ks[:150]}")
            anc = ANCHOR.get(term, []) + [corr]
            ex = near(v["출력"], anc)
            print(f"  번역문 : {ex[:260] if ex else '(해당 부분을 찾지 못함 — 문장 누락 가능)'}")
            print(f"  판정 힌트: 번역문에 유사어가 있으면 '표현 차이', 다른 장기명이면 '오역',")
            print(f"             아무것도 없으면 '문장 누락'")
print(f"\n총 {n}건")
