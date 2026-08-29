#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translate_eval.py — 동의서 다국어 번역 + 용어 사전 기반 오역 채점"""
import os, re, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 0
HOME = os.path.expanduser("~/이윤우")
LANGS = {"en": ("English", "영어"), "zh": ("Chinese", "중국어"), "ja": ("Japanese", "일본어")}

# 공기어 노이즈 제거: 이 단어들은 대응어로 채택하지 않음
NOISE = {"en": {"cancer","surgery","disease","treatment","hospital","doctor","health"},
         "zh": {"手术","患者","治疗","医院","进行","的患者","在胆囊"},
         "ja": {"手術","患者","治療","病院","新型","コロナウイルス","血液"}}

rep = {}
for f in ["reportA.json","reportB.json","reportC.json"]:
    p = os.path.join(HOME, f)
    if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))
DICT = json.load(open(os.path.join(HOME,"termdict_verified.json"), encoding="utf-8"))

# 용어 → 언어별 대표 대응어 1개 확정
TERM = {}
for t, per in DICT.items():
    e = {}
    for code in ("en","ja","zh"):
        for c, r in per.get(code, []):
            if c in NOISE[code]: continue
            e[code] = c; break
    if e: TERM[t] = e
print(f"[사전] 채점 가능 용어 {len(TERM)}개")
for t, e in list(TERM.items())[:6]: print(f"   {t}: {e}")

docs = sorted(rep.keys(), key=lambda x:(len(x),x))
print(f"[대상] 동의서 {len(docs)}건 × {len(LANGS)}개 언어 = {len(docs)*len(LANGS)}회 번역\n")

tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()

SYS = "You are a professional medical translator. Translate faithfully without adding or omitting content."
def gen(up, mt=4096):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":up}]
    try:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=mt,do_sample=False)
    return tok.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()

KOCH = re.compile(r"[가-힣]")
RES, ROWS = {}, []
for name in docs:
    SRC = rep[name]["원문"]
    present = [t for t in TERM if t in SRC]
    RES[name] = {"원문자수": len(SRC), "등장용어": present, "언어": {}}
    print(f"{'='*74}\n### {name}  ({len(SRC)}자) / 사전 용어 {len(present)}개: {', '.join(present[:10])}")
    for code, (eng, kor) in LANGS.items():
        if not present: continue
        t0=time.time()
        out = gen(f"Translate the following Korean surgical consent form into {eng}. "
                  f"Translate every sentence faithfully. Do not add explanations.\n\n"+SRC)
        dt = time.time()-t0
        hit, miss = [], []
        for t in present:
            c = TERM[t].get(code)
            if not c: continue
            (hit if c.lower() in out.lower() else miss).append((t, c))
        n = len(hit)+len(miss)
        rate = len(hit)/n*100 if n else 0.0
        ko_left = len(KOCH.findall(out))
        RES[name]["언어"][code] = {"초":round(dt,1),"길이":len(out),"한글잔존":ko_left,
                                   "적중":hit,"미검출":miss,"정확도":rate,"출력":out}
        ROWS.append((name, kor, n, len(hit), rate, ko_left, len(out)))
        print(f"  [{kor}] {dt:5.1f}초 {len(out):>5}자 한글{ko_left:>4} "
              f"| 용어 {len(hit)}/{n} ({rate:.0f}%)"
              + (f" | 미검출: {', '.join(f'{a}→{b}' for a,b in miss[:4])}" if miss else ""), flush=True)

print(f"\n{'='*74}\n[요약] 용어 번역 정확도\n{'='*74}")
print(f"{'문서':<8}{'언어':<8}{'대상':>5}{'적중':>5}{'정확도':>8}{'한글잔존':>9}")
for r in ROWS:
    print(f"{r[0]:<8}{r[1]:<8}{r[2]:>5}{r[3]:>5}{r[4]:>7.0f}%{r[5]:>9}")
for code,(eng,kor) in LANGS.items():
    sub=[r for r in ROWS if r[1]==kor]
    if sub:
        tot=sum(r[2] for r in sub); hit=sum(r[3] for r in sub)
        print(f"\n[{kor}] 전체 {hit}/{tot} = {hit/max(tot,1)*100:.1f}%  "
              f"평균 한글잔존 {sum(r[5] for r in sub)/len(sub):.0f}자")

print(f"\n{'='*74}\n[미검출 상세] 오역·누락 후보\n{'='*74}")
agg={}
for name,d in RES.items():
    for code,v in d["언어"].items():
        for t,c in v["미검출"]:
            agg.setdefault((t,code),[]).append(name)
for (t,code),ds in sorted(agg.items(), key=lambda x:-len(x[1])):
    print(f"  {t} → {TERM[t][code]} ({LANGS[code][1]}) : {len(ds)}건  {ds}")

json.dump(RES, open(os.path.join(HOME,"translate_eval.json"),"w"), ensure_ascii=False, indent=1)
print("\n[저장] ~/이윤우/translate_eval.json")
