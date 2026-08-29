#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final_check.py — 공백 보정 길이비 재계산 + 베트남어 번역 점검"""
import os, re, json
HOME = os.path.expanduser("~/이윤우")
def load(fs):
    d={}
    for f in fs:
        p=os.path.join(HOME,f)
        if os.path.exists(p): d.update(json.load(open(p,encoding="utf-8")))
    return d
H = load(["reportA.json","reportB.json","reportC.json"])
Q = load(["reportQ1.json","reportQ2.json"])
def clean(s):
    s = re.sub(r"[ \t\u00a0]{3,}", " ", s)      # 3칸 이상 연속 공백 축약
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[\ufffd\u0000-\u0008]", "", s) # 깨진 바이트 제거
    return s.strip()

print("="*72); print("[1] 길이비 — 공백/반복 보정 전후"); print("="*72)
print(f"{'문서':<8}{'hari 원본→보정':>18}{'Qwen 원본→보정':>18}")
print("-"*46)
agg={"h":[],"q":[],"hc":[],"qc":[]}
for n in sorted(set(H)&set(Q), key=lambda x:(len(x),x)):
    ho,qo = H[n]["변환문"], Q[n]["변환문"]
    hs,qs = H[n]["원문"],  Q[n]["원문"]
    a  = len(ho)/len(hs)*100;        b  = len(qo)/len(qs)*100
    ac = len(clean(ho))/len(hs)*100; bc = len(clean(qo))/len(qs)*100
    mark = "  ⚠" if abs(b-bc) > 10 else ""
    print(f"{n:<8}{a:>8.0f}%→{ac:>7.0f}%{b:>9.0f}%→{bc:>7.0f}%{mark}")
    agg["h"].append(a); agg["hc"].append(ac); agg["q"].append(b); agg["qc"].append(bc)
m=lambda k: sum(agg[k])/len(agg[k])
print("-"*46)
print(f"{'평균':<8}{m('h'):>8.0f}%→{m('hc'):>7.0f}%{m('q'):>9.0f}%→{m('qc'):>7.0f}%")
print(f"\n보정 후 길이 차이: {m('qc')-m('hc'):+.0f}%p  (보정 전 {m('q')-m('h'):+.0f}%p)")

print("\n"+"="*72); print("[2] 베트남어 번역 — '2. 목적 및 효과' 부분"); print("="*72)
t = json.load(open(os.path.join(HOME,"translate_demo.json"), encoding="utf-8"))
vi = t["Vietnamese"]["출력"]
# 섹션 2 근처 추출
m2 = re.search(r"(?m)^\s*#{0,6}\s*2[.)]\s.*", vi)
if m2:
    print(" ".join(vi[m2.start():m2.start()+700].split()))
else:
    print("(섹션 2를 찾지 못함 — 앞부분 800자)")
    print(" ".join(vi[:800].split()))
print("\n[갑상선 관련 단어 탐색]")
for w in ["tuyến giáp","giáp","bướm","cổ","thyroid","甲状腺"]:
    print(f"  '{w}': {'있음' if w.lower() in vi.lower() else '없음'}")

print("\n"+"="*72); print("[3] 중국어 번역 끝부분 (중단 여부)"); print("="*72)
zh = t["Chinese"]["출력"]
print(" ".join(zh[-300:].split()))
