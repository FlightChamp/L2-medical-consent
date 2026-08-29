#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_models_real.py — 실제 동의서 13건에서 hari vs Qwen3 비교"""
import os, re, json
HOME = os.path.expanduser("~/이윤우")
def load(files):
    d={}
    for f in files:
        p=os.path.join(HOME,f)
        if os.path.exists(p): d.update(json.load(open(p,encoding="utf-8")))
    return d
H = load(["reportA.json","reportB.json","reportC.json"])
Q = load(["reportQ1.json","reportQ2.json"])
if not Q: raise SystemExit("[중단] Qwen 결과 없음 — 실행 완료 후 다시")

def nsp(s): return re.sub(r"\s+","",s)
def stem(t,ON):
    b=nsp(t)
    for c in range(4):
        if len(b)-c<3: break
        if b[:len(b)-c] in ON: return True
    return False
UNIT=r"(시간|분|초|일|주일|주|개월|년|세|%|cc|ml|mg|g|cm|mm|회|번|명|배)"
NUM=re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:[~∼\-–]\s*(\d+(?:[.,]\d+)?))?\s*"+UNIT)
def toks(t): return {(m.group(1)+("~"+m.group(2) if m.group(2) else "")+m.group(3)) for m in NUM.finditer(t)}

common = sorted(set(H)&set(Q), key=lambda x:(len(x),x))
print(f"공통 문서 {len(common)}개\n")
h=f"{'문서':<8}{'사실 H/Q':>16}{'용어보정 H/Q':>18}{'수치창작 H/Q':>14}{'길이비 H/Q':>14}{'메타 H/Q':>10}"
print(h); print("-"*len(h))
agg={"hf":[],"qf":[],"ht":[],"qt":[],"hn":0,"qn":0,"hl":[],"ql":[],"hm":0,"qm":0}
for n in common:
    a,b=H[n],Q[n]
    at=sum(1 for t in a["용어후보"] if stem(t,nsp(a["변환문"])))/len(a["용어후보"])*100
    bt=sum(1 for t in b["용어후보"] if stem(t,nsp(b["변환문"])))/len(b["용어후보"])*100
    an=len(toks(a["변환문"])-toks(a["원문"])); bn=len(toks(b["변환문"])-toks(b["원문"]))
    al=len(a["변환문"])/len(a["원문"])*100; bl=len(b["변환문"])/len(b["원문"])*100
    am,bm=len(a["메타"]),len(b["메타"])
    print(f"{n:<8}{a['사실보존']:>7.1f}/{b['사실보존']:<8.1f}{at:>8.1f}/{bt:<9.1f}"
          f"{an:>6}/{bn:<7}{al:>6.0f}%/{bl:<6.0f}%{am:>4}/{bm:<5}")
    agg["hf"].append(a["사실보존"]); agg["qf"].append(b["사실보존"])
    agg["ht"].append(at); agg["qt"].append(bt)
    agg["hn"]+=an; agg["qn"]+=bn
    agg["hl"].append(al); agg["ql"].append(bl)
    agg["hm"]+=am; agg["qm"]+=bm
print("-"*len(h))
m=lambda k: sum(agg[k])/len(agg[k])
print(f"{'평균':<8}{m('hf'):>7.1f}/{m('qf'):<8.1f}{m('ht'):>8.1f}/{m('qt'):<9.1f}"
      f"{agg['hn']:>6}/{agg['qn']:<7}{m('hl'):>6.0f}%/{m('ql'):<6.0f}%{agg['hm']:>4}/{agg['qm']:<5}")
print("\nH = snuh/hari-q3-8b (의료 특화)   Q = Qwen/Qwen3-8B (범용)")
win_f = "hari" if m('hf')>m('qf') else ("Qwen3" if m('qf')>m('hf') else "동률")
print(f"\n[사실 보존 우위] {win_f}  (차이 {abs(m('hf')-m('qf')):.1f}%p)")
print(f"[합성 문서 기준] hari 79.1% vs Qwen3 81.9% (구 측정 방식, 300자 문서 3건)")
