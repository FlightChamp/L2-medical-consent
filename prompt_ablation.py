#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt_ablation.py — 메타 발화·수치 오류를 프롬프트로 억제할 수 있는가
   대상: doc1(빈칸 창작), doc3(수치 변조), doc11(빈칸 창작)"""
import os, re, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 0
HOME = os.path.expanduser("~/이윤우")
TARGETS = ["doc1", "doc3", "doc11"]

# 이미 실행된 결과에서 정제 원문(SRC)을 그대로 재사용 → 조건 동일
data = {}
for fn in ["reportA.json", "reportB.json", "reportC.json"]:
    p = os.path.join(HOME, fn)
    if os.path.exists(p): data.update(json.load(open(p, encoding="utf-8")))

SYS = "당신은 한국어 의료 어시스턴트입니다. 원문에 없는 내용은 절대 추가하지 마세요."
BASE = ("다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
        "수술 이름, 모든 수치, 발생 조건, 시간·기간 정보는 하나도 빠짐없이 유지해 주세요.")
ADD_META = "\n\n인사말, 머리말, 맺음말, 작업 설명을 붙이지 마세요. 변환된 문서 본문만 출력하세요."
ADD_NUM  = ("\n\n원문에서 값이 비어 있는 칸은 채우지 말고 '____'로 그대로 두세요. "
            "원문에 없는 숫자를 새로 만들지 마세요. 모든 숫자와 단위는 원문에 적힌 그대로 사용하세요.")

PROMPTS = {
    "P0_현재":        BASE,
    "P1_메타억제":    BASE + ADD_META,
    "P2_메타+수치":   BASE + ADD_META + ADD_NUM,
}

META = [(r"필요하시?면","사용자 제안"),(r"드립니다|드릴\s*수\s*있","화자 개입"),
        (r"도움이\s*되(었|시)","화자 개입"),(r"다시\s*(정리|작성)해","자기 작업 언급"),
        (r"원문에\s*있는\s*모든","자기 작업 언급"),(r"물론입니다","인사말"),
        (r"아래는\s","머리말"),(r"^---\s*$","구분선")]
UNIT = r"(시간|분|초|일|주일|주|개월|달|년|세|%|cc|ml|mg|g|cm|mm|회|번|명|개|배)"
NUM  = re.compile(r"(\d+(?:[.,]\d+)?)\s*" + UNIT + r"?")
MDNUM = re.compile(r"(^|\n)\s*#{0,6}\s*\d+\.\s|\n\s*\d+\.\s\*\*")

def nums(t, skip_md=False):
    md=set()
    if skip_md:
        for m in MDNUM.finditer(t):
            for i in range(m.start(), m.end()): md.add(i)
    return [(m.group(1)+(m.group(2) or ""), m.start())
            for m in NUM.finditer(t) if not (skip_md and m.start() in md)]

print("[모델 로딩]", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()

def gen(sp, up, mt=4096):
    msgs=[{"role":"system","content":sp},{"role":"user","content":up}]
    try:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=mt,do_sample=False)
    return tok.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()

RESULT = {}
for name in TARGETS:
    if name not in data:
        print(f"[건너뜀] {name} — 결과 파일에 없음"); continue
    SRC = data[name]["원문"]
    RESULT[name] = {}
    print(f"\n{'#'*72}\n### {name}  (원문 {len(SRC)}자)\n{'#'*72}", flush=True)
    src_tok = {t for t,_ in nums(SRC)}
    for pid, up in PROMPTS.items():
        t0=time.time()
        OUT = gen(SYS, up + "\n\n" + SRC)
        dt = time.time()-t0
        metas = [w for pat,w in META for m in [re.search(pat, OUT, re.M)] if m]
        made  = sorted({t for t,_ in nums(OUT, skip_md=True) if t not in src_tok})
        keep  = sum(1 for t in src_tok if t in OUT)
        rate  = keep/len(src_tok)*100 if src_tok else 100
        RESULT[name][pid] = {"길이": len(OUT), "길이비": len(OUT)/len(SRC)*100,
                             "메타": metas, "창작수치": made, "수치보존": rate,
                             "초": round(dt,1), "출력": OUT}
        print(f"  [{pid}] {dt:5.1f}초 | {len(OUT)}자 ({len(OUT)/len(SRC)*100:.0f}%) "
              f"| 메타 {len(metas)}건 {metas} | 창작수치 {len(made)}건 {made} "
              f"| 원문수치 보존 {rate:.0f}%", flush=True)

print("\n" + "="*78)
print("[요약] 프롬프트별 개선 효과")
print("="*78)
h=f"{'문서':<8}{'프롬프트':<14}{'메타':>6}{'창작수치':>9}{'수치보존':>9}{'길이비':>8}"
print(h); print("-"*len(h))
for name, per in RESULT.items():
    for pid, r in per.items():
        print(f"{name:<8}{pid:<14}{len(r['메타']):>6}{len(r['창작수치']):>9}"
              f"{r['수치보존']:>8.0f}%{r['길이비']:>7.0f}%")
    print("-"*len(h))
agg={}
for name, per in RESULT.items():
    for pid, r in per.items():
        a=agg.setdefault(pid, {"메타":0,"창작":0,"보존":[],"n":0})
        a["메타"]+=len(r["메타"]); a["창작"]+=len(r["창작수치"])
        a["보존"].append(r["수치보존"]); a["n"]+=1
print(f"\n{'프롬프트':<14}{'메타 합계':>10}{'창작 합계':>10}{'평균 수치보존':>14}")
for pid,a in agg.items():
    print(f"{pid:<14}{a['메타']:>10}{a['창작']:>10}{sum(a['보존'])/a['n']:>13.1f}%")

json.dump(RESULT, open(os.path.join(HOME,"prompt_ablation.json"),"w"),
          ensure_ascii=False, indent=2)
print("\n[저장] ~/이윤우/prompt_ablation.json")
