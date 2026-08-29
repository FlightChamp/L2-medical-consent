#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_termdict v2 — 영어 어간 통합, 중국어/일본어 최장 후보만 채택"""
import os, json, re, collections

BASE = os.path.expanduser("~/shared/data")
SRC = {"en": ("corpus_enko/ko2en_medical_2_validation.json", "영어"),
       "ja": ("corpus_jako/ko2ja_medical_2_validation.json", "일본어"),
       "zh": ("corpus_zhko/ko2zh_medical_2_validation.json", "중국어")}
FOCUS = ["갑상선","식도","기관지","담낭","탈장","치핵","기흉","유방","척추","전립선",
         "자궁","방광","신장","관절","골절","인공관절","봉합","절제","마취","합병증",
         "후유증","수혈","종양","생검","조직검사","염증","감염","출혈","통증","재발"]
MIN_N, MIN_RATE = 3, 0.45

def load(p):
    o = json.load(open(p, encoding="utf-8"))
    if isinstance(o, dict):
        for v in o.values():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
    return o if isinstance(o, list) else []

def stem_en(w):
    """간이 어간: 복수형·형용사형 통합"""
    w = w.lower()
    for suf in ("ectomy","ology","itis"):          # 의학 접미사는 보존
        if w.endswith(suf): return w
    for suf in ("ies",):
        if len(w) > 5 and w.endswith(suf): return w[:-3] + "y"
    for suf in ("es","s"):
        if len(w) > 4 and w.endswith(suf) and not w.endswith("ss"): return w[:-len(suf)]
    return w

STOP_EN = set("""this that with from have been they were their which will more than about into
other some such most many when where would could should also being used using korea korean
said says announced according percent year years case cases people patient patients time
after before through during between under over first second新""".split())

def cands(text, code):
    if code == "en":
        return {stem_en(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
                if stem_en(w) not in STOP_EN and len(stem_en(w)) >= 4}
    if code == "zh":
        out = set()
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            for n in (2,3,4):
                out |= {run[i:i+n] for i in range(len(run)-n+1)}
        return out
    out = set(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
    out |= set(re.findall(r"[\u30a0-\u30ff]{3,10}", text))
    return out

def drop_substrings(items):
    """(후보, 비율) 목록에서 다른 후보의 부분 문자열인 것 제거"""
    items = sorted(items, key=lambda x: (-len(x[0]), -x[1]))
    keep = []
    for c, r in items:
        if any(c in k and c != k and abs(r - kr) < 0.15 for k, kr in keep): continue
        keep.append((c, r))
    return sorted(keep, key=lambda x: -x[1])

DATA = {}
for code, (rel, lang) in SRC.items():
    p = os.path.join(BASE, rel)
    if os.path.exists(p): DATA[code] = (load(p), lang)

result, stat = {}, collections.Counter()
print(f"{'용어':<10}{'언어':<8}{'문장':>6}   대응어 (등장률)")
print("-"*78)
for t in FOCUS:
    result[t] = {}
    for code, (rows, lang) in DATA.items():
        pairs = [(str(r.get("한국어","")), str(r.get(lang,"")))
                 for r in rows if t in str(r.get("한국어",""))]
        pairs = [(k,f) for k,f in pairs if len(f) > 4]
        n = len(pairs)
        if n < MIN_N:
            print(f"{t:<10}{lang:<8}{n:>6}   (표본 부족)"); continue
        cnt = collections.Counter()
        for _, fo in pairs:
            for c in cands(fo, code): cnt[c] += 1
        raw = [(c, v/n) for c, v in cnt.most_common(80) if v/n >= MIN_RATE]
        top = drop_substrings(raw)[:3]
        result[t][code] = top
        if top: stat[code] += 1
        s = ", ".join(f"{c}({r:.2f})" for c, r in top) if top else "(후보 없음)"
        print(f"{t:<10}{lang:<8}{n:>6}   {s[:62]}")
    print()

json.dump(result, open(os.path.expanduser("~/이윤우/termdict_verified.json"),"w"),
          ensure_ascii=False, indent=1)
ok3 = sum(1 for t in FOCUS if all(result[t].get(c) for c in ("en","ja","zh")))
ok1 = sum(1 for t in FOCUS if any(result[t].get(c) for c in ("en","ja","zh")))
print(f"[요약] 3개 언어 모두 확보 {ok3}/{len(FOCUS)} · 1개 이상 확보 {ok1}/{len(FOCUS)}")
print(f"       언어별 확보: 영어 {stat['en']} / 일본어 {stat['ja']} / 중국어 {stat['zh']}")
print("[저장] ~/이윤우/termdict_verified.json")
