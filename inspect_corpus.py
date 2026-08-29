#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_corpus.py v2 — 의료 말뭉치 구조 파악 + 동의서 용어 포함률"""
import os, json, glob, re

BASE = os.path.expanduser("~/shared/data")
TARGETS = {
    "corpus_enko": ("영어",   "ko2en_medical_2_validation.json"),
    "corpus_jako": ("일본어", "ko2ja_medical_2_validation.json"),
    "corpus_zhko": ("중국어", "ko2zh_medical_2_validation.json"),
}
# 우리 동의서 13건에 실제로 등장한 용어 중심
MED = ["수술","시술","마취","합병증","후유증","감염","출혈","혈전색전증","진단","치료",
       "골절","관절","인공관절","담낭","탈장","치핵","기흉","유방","척추","갑상선",
       "전립선","자궁","신장","방광","봉합","절제","삽입","이식","수혈","배액관",
       "신경손상","혈관손상","인대","종양","염증","통증","부작용","알레르기",
       "정복","고정술","흉관","결절","생검","조직검사","석고","보조기구","금식",
       "재발","불유합","지연유합","부정유합","성대마비","저칼슘혈증","배뇨곤란"]
KO = re.compile(r"[가-힣]")

def find_list(obj):
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
    return None

def guess_keys(lst, n=300):
    keys = list(lst[0].keys())
    ko_key = fo_key = None
    best_ko = best_fo = -1
    for k in keys:
        vals = [str(r.get(k, "")) for r in lst[:n]]
        long_vals = [v for v in vals if len(v) > 8]
        if not long_vals: continue
        ko_ratio = sum(1 for v in long_vals if len(KO.findall(v)) > 3)/len(long_vals)
        if ko_ratio > best_ko and ko_ratio > 0.7: best_ko, ko_key = ko_ratio, k
        non = 1 - ko_ratio
        if non > best_fo and non > 0.7: best_fo, fo_key = non, k
    return ko_key, fo_key

for d, (lang, fname) in TARGETS.items():
    p = os.path.join(BASE, d, fname)
    print("="*78)
    if not os.path.exists(p):
        cands = glob.glob(os.path.join(BASE, d, "*medical*"))
        print(f"[{lang}] {fname} 없음. 후보: {[os.path.basename(c) for c in cands]}")
        if not cands: continue
        p = cands[0]
    print(f"[{lang}] {os.path.basename(p)} ({os.path.getsize(p)/1024/1024:.1f}MB)")
    try:
        obj = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  파싱 실패: {e}"); continue
    lst = find_list(obj)
    if not lst:
        print(f"  리스트 못 찾음. 최상위 키: {list(obj.keys())[:10] if isinstance(obj,dict) else type(obj)}")
        continue
    print(f"  총 {len(lst):,}건 / 필드: {list(lst[0].keys())}")
    kk, fk = guess_keys(lst)
    print(f"  한국어 필드: {kk}   {lang} 필드: {fk}")
    if not (kk and fk):
        print(f"  샘플 원본: {str(lst[0])[:400]}"); continue
    print(f"  샘플 KO : {str(lst[0][kk])[:110]}")
    print(f"  샘플 {lang}: {str(lst[0][fk])[:110]}")
    hit, terms = [], {}
    for r in lst:
        s = str(r.get(kk, ""))
        f = [m for m in MED if m in s]
        if f:
            hit.append((r, f))
            for m in f: terms[m] = terms.get(m, 0) + 1
    print(f"\n  ★ 동의서 용어 포함 문장: {len(hit):,}/{len(lst):,}건 ({len(hit)/len(lst)*100:.1f}%)")
    top = sorted(terms.items(), key=lambda x: -x[1])[:20]
    print(f"  상위 용어: {', '.join(f'{k}({v})' for k,v in top)}")
    print(f"  미등장 용어: {[m for m in MED if m not in terms]}")
    print("\n  [예시]")
    for r, f in hit[:4]:
        print(f"    · [{','.join(f[:4])}]")
        print(f"      KO : {str(r[kk])[:100]}")
        print(f"      {lang}: {str(r[fk])[:100]}")
