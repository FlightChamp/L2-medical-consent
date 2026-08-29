#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_termdict.py — 말뭉치에서 의학 용어 대역 사전 추출"""
import os, json, re, collections

BASE = os.path.expanduser("~/shared/data")
SRC = {
    "en": ("corpus_enko/ko2en_medical_2_validation.json", "영어"),
    "ja": ("corpus_jako/ko2ja_medical_2_validation.json", "일본어"),
    "zh": ("corpus_zhko/ko2zh_medical_2_validation.json", "중국어"),
}
# 오늘 오역이 확인된 용어 + 동의서 핵심 용어
FOCUS = ["갑상선","식도","기관지","담낭","탈장","치핵","기흉","유방","척추","전립선",
         "자궁","방광","신장","관절","골절","인공관절","봉합","절제","마취","합병증",
         "후유증","수혈","종양","생검","조직검사","염증","감염","출혈","통증","재발"]

def load(p):
    obj = json.load(open(p, encoding="utf-8"))
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
    return obj if isinstance(obj, list) else []

dic = collections.defaultdict(lambda: collections.defaultdict(list))
counts = {}
for code, (rel, lang) in SRC.items():
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        print(f"[없음] {rel}"); continue
    rows = load(p)
    ko_k, fo_k = "한국어", lang
    if rows and fo_k not in rows[0]:
        print(f"[경고] '{fo_k}' 필드 없음. 실제 필드: {list(rows[0].keys())}"); continue
    n = 0
    for r in rows:
        ko, fo = str(r.get(ko_k,"")), str(r.get(fo_k,""))
        if len(ko) < 5 or len(fo) < 5: continue
        for t in FOCUS:
            if t in ko:
                dic[t][code].append((ko, fo)); n += 1
    counts[code] = (len(rows), n, lang)
    print(f"[{lang}] {len(rows):,}건 중 초점용어 매칭 {n:,}건")

print("\n" + "="*80)
print(f"{'용어':<12}{'영어':>8}{'일본어':>8}{'중국어':>8}   대표 대응 예시")
print("-"*80)
out = {}
for t in FOCUS:
    c = {k: len(dic[t][k]) for k in ("en","ja","zh")}
    ex = ""
    if dic[t]["en"]:
        ko, fo = dic[t]["en"][0]
        i = ko.find(t)
        ex = f"…{ko[max(0,i-12):i+14]}… → {fo[:60]}"
    print(f"{t:<12}{c['en']:>8}{c['ja']:>8}{c['zh']:>8}   {ex[:90]}")
    out[t] = {k: dic[t][k][:20] for k in ("en","ja","zh")}

zero = [t for t in FOCUS if all(len(dic[t][k])==0 for k in ("en","ja","zh"))]
print(f"\n[세 언어 모두 미등장] {zero if zero else '없음'}")

with open(os.path.expanduser("~/이윤우/termdict_raw.json"),"w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n[저장] ~/이윤우/termdict_raw.json (용어별 문장쌍 최대 20건씩)")

# 갑상선 집중 확인
print("\n" + "="*80)
print("[갑상선] 문장쌍 — 오늘 오역 검증의 정답지")
print("="*80)
for code, lang in (("en","영어"),("ja","일본어"),("zh","중국어")):
    pairs = dic["갑상선"][code]
    print(f"\n── {lang}: {len(pairs)}건")
    for ko, fo in pairs[:3]:
        print(f"   KO: {ko[:95]}")
        print(f"   {lang}: {fo[:95]}")
