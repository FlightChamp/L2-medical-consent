#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final_mistrans.py — 남은 미검출 10건의 번역문 해당 부분 발췌"""
import os, re, json
HOME = os.path.expanduser("~/이윤우")
RES = json.load(open(os.path.join(HOME,"translate_eval.json"), encoding="utf-8"))
LANG = {"en":"영어","zh":"중국어","ja":"일본어"}

CASES = [("doc1","ja","갑상선",["butterfly","蝶","首","前","甲状","食道","気管"]),
         ("doc3","en","조직검사",["confirm","diagnos","5","7 days","tissue"]),
         ("doc3","ja","담낭",["胆","石","摘出","切除"]),
         ("doc5","en","방광",["lower abdomen","urin","belly","abdomen"]),
         ("doc5","zh","방광",["小腹","下腹","排尿","小便"]),
         ("doc14","zh","방광",["小腹","下腹","排尿","小便"]),
         ("doc8","en","절제",["remov","excis","cut","needle","core"]),
         ("doc11","en","봉합",["stitch","clos","repair","reconstruct"]),
         ("doc12","en","후유증",["complication","side effect","after"]),
         ("doc13","ja","치핵",["痔","手術名","結紮","切除"])]

def near(text, kws, span=230):
    low=text.lower()
    for k in kws:
        i=low.find(k.lower())
        if i>=0: return " ".join(text[max(0,i-110):i+span].split())
    return ""

for doc, code, term, anchors in CASES:
    v = RES.get(doc,{}).get("언어",{}).get(code)
    print("="*78)
    print(f"[{doc}] {LANG[code]} · '{term}'")
    if not v: print("  (데이터 없음)"); continue
    ex = near(v["출력"], anchors)
    print(f"  {ex[:330] if ex else '(앵커 미발견 — 아래 전체에서 직접 확인)'}")
    if not ex:
        print(f"  번역문 앞 300자: {' '.join(v['출력'][:300].split())}")
