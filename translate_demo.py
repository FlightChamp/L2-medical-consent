#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""translate_demo.py — Scope② 첫 검증: 오염된 원문이 번역에서 어떻게 전달되는가
   GPU 1 사용 (GPU 0의 Qwen 실행과 병렬)"""
import os, re, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 1
HOME = os.path.expanduser("~/이윤우")

rep = {}
for f in ["reportA.json", "reportB.json", "reportC.json"]:
    p = os.path.join(HOME, f)
    if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))

LANGS = {
    "English":    ["thyroid", "butterfly"],
    "Chinese":    ["甲状腺", "蝴蝶"],
    "Vietnamese": ["tuyến giáp", "giáp", "bướm"],
}
SYS = "You are a professional medical translator. Translate faithfully without adding or omitting content."

print("[모델 로딩] GPU", GPU, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()

def gen(sp, up, mt=3072):
    msgs=[{"role":"system","content":sp},{"role":"user","content":up}]
    try:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=mt,do_sample=False)
    return tok.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()

RES = {}
SRC = rep["doc1"]["원문"]
print(f"\n원문 doc1 (골절수술동의서, {len(SRC)}자) — 갑상선 오염 문장 포함\n")

for lang, markers in LANGS.items():
    print("="*76); print(f"[{lang}] 번역 중…", flush=True)
    t0=time.time()
    out = gen(SYS, f"Translate the following Korean surgical consent form into {lang}. "
                   f"Translate every sentence faithfully. Do not add explanations.\n\n" + SRC)
    dt = time.time()-t0
    low = out.lower()
    found = [m for m in markers if m.lower() in low]
    ko_left = len(re.findall(r"[가-힣]", out))
    RES[lang] = {"초": round(dt,1), "길이": len(out), "오염전달": found,
                 "한글잔존": ko_left, "출력": out}
    print(f"  {dt:.1f}초 | {len(out)}자 | 한글 잔존 {ko_left}자")
    print(f"  오염 문장 전달 여부: {'🔴 전달됨 ' + str(found) if found else '⚪ 미검출'}")
    # 오염 부분 발췌
    for m in found:
        i = low.find(m.lower())
        print(f"  발췌: …{' '.join(out[max(0,i-90):i+150].split())}…")
    print()

print("="*76); print("[요약]"); print("="*76)
print(f"{'언어':<12}{'소요':>7}{'길이':>7}{'한글잔존':>9}{'오염전달':>10}")
for lang, r in RES.items():
    print(f"{lang:<12}{r['초']:>6.1f}s{r['길이']:>7}{r['한글잔존']:>9}"
          f"{('전달' if r['오염전달'] else '미검출'):>10}")

json.dump(RES, open(os.path.join(HOME,"translate_demo.json"),"w"),
          ensure_ascii=False, indent=2)
print("\n[저장] ~/이윤우/translate_demo.json")
