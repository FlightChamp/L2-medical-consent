#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_all.py — 전체 결과 지표 재계산 + 발표용 정리 (GPU 불필요)"""
import os, re, json

HOME = os.path.expanduser("~/이윤우")
CAND = ["reportA.json", "reportB.json", "reportC.json"]
OUT_MD = os.path.join(HOME, "presentation_summary.md")

data, loaded = {}, []
for fn in CAND:
    p = os.path.join(HOME, fn)
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8")); data.update(d)
            loaded.append(f"{fn}({len(d)})")
        except Exception as e: print(f"[경고] {fn}: {e}")
if not data: raise SystemExit("[중단] reportA/B/C.json 없음")
print(f"[불러옴] {', '.join(loaded)} → 총 {len(data)}개 문서\n")

def nsp(s): return re.sub(r"\s+", "", s)
def split_sents(t):
    out=[]
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+"," ",p).strip(" -–—:·")
        p = re.sub(r"\[\s*\]|\(\s*\)", "", p).strip()
        if len(p)>=12 and len(re.findall(r"[가-힣]",p))>=6: out.append(p)
    return out
HEADER = re.compile(r"^\*\*.*\*\*$|^#{1,6}\s|^-+$")
def is_header(s):
    s2=s.strip()
    if HEADER.match(s2): return True
    return s2.count("**")>=2 and not re.search(r"[다요]\.$|입니다|합니다|됩니다", s2)
NUM = re.compile(r"(\d[\d,.]*)\s*(시간|분|일|주일|주|개월|년|세|%|cc|ml|mg|g|cm|mm|회|번|명|개)?")
def stem_match(t, ON):
    b=nsp(t)
    for c in range(4):
        if len(b)-c < 3: break
        if b[:len(b)-c] in ON: return True
    return False

rows, all_nums, all_lost, all_meta = [], [], [], []
for name, r in sorted(data.items(), key=lambda x:(len(x[0]), x[0])):
    SRC, OUT = r["원문"], r["변환문"]; SN, ON = nsp(SRC), nsp(OUT)
    T = r["용어후보"]
    t_old = sum(1 for t in T if t in OUT)/len(T)*100
    t_new = sum(1 for t in T if stem_match(t,ON))/len(T)*100
    lost  = [t for t in T if not stem_match(t,ON)]
    osents = split_sents(OUT); bad=set(r["환각"])
    rs = [s for s in osents if not is_header(s)]
    rb = [s for s in rs if any(s.startswith(b[:30]) or b.startswith(s[:30]) for b in bad)]
    g_new = (len(rs)-len(rb))/len(rs)*100 if rs else 0
    sn = {m.group(1)+(m.group(2) or "") for m in NUM.finditer(SRC)}
    added, seen = [], set()
    for m in NUM.finditer(OUT):
        v=m.group(1)+(m.group(2) or "")
        if v in sn or nsp(v) in SN or v in seen: continue
        seen.add(v); added.append((v, " ".join(OUT[max(0,m.start()-30):m.end()+30].split())))
    rows.append({"문서":name,"사실":r["사실보존"],"용어o":t_old,"용어n":t_new,
                 "근거o":r["근거율"],"근거n":g_new,"길이비":len(OUT)/len(SRC)*100,
                 "숫자":len(added),"메타":len(r["메타"])})
    all_nums += [(name,v,c) for v,c in added]
    all_lost += [(name,t) for t in lost]
    all_meta += [(name,m) for m in r["메타"]]

h=f"{'문서':<8}{'사실':>7}{'용어 원본→보정':>17}{'근거 원본→보정':>17}{'길이비':>8}{'숫자환각':>9}{'메타':>6}"
print(h); print("-"*len(h))
for x in rows:
    print(f"{x['문서']:<8}{x['사실']:>6.1f}%{x['용어o']:>9.1f}%→{x['용어n']:>5.1f}%"
          f"{x['근거o']:>9.1f}%→{x['근거n']:>5.1f}%{x['길이비']:>7.0f}%{x['숫자']:>9}{x['메타']:>6}")
print("-"*len(h)); n=len(rows); avg=lambda k: sum(x[k] for x in rows)/n
print(f"{'평균':<8}{avg('사실'):>6.1f}%{avg('용어o'):>9.1f}%→{avg('용어n'):>5.1f}%"
      f"{avg('근거o'):>9.1f}%→{avg('근거n'):>5.1f}%{avg('길이비'):>7.0f}%"
      f"{sum(x['숫자'] for x in rows):>9}{sum(x['메타'] for x in rows):>6}")

trunc=[x['문서'] for x in rows if x['길이비']<100]
print(f"\n[절단 의심] 길이비 100% 미만: {trunc if trunc else '없음'}")

print("\n"+"="*74); print(f"[핵심] 숫자 환각 ({len(all_nums)}건)"); print("="*74)
for d,v,c in all_nums: print(f"  ❌ [{d}] '{v}'\n      …{c}…")
print(f"\n  → {n}개 문서 중 {len({d for d,_,_ in all_nums})}개에서 발생")

print("\n"+"="*74); print(f"[참고] 보정 후 소실 용어 ({len(all_lost)}건)"); print("="*74)
for d,t in all_lost: print(f"  [{d}] {t}")
print("\n"+"="*74); print(f"[참고] 메타 발화 ({len(all_meta)}건)"); print("="*74)
for d,m in all_meta: print(f"  [{d}] {m}")

with open(OUT_MD,"w",encoding="utf-8") as f:
    f.write(f"# 실제 병원 동의서 검증 결과\n\n대상: 경기도의료원 실제 수술 동의서 {n}건\n\n")
    f.write("| 문서 | 사실 보존 | 용어(원본) | 용어(보정) | 근거율(원본) | 근거율(보정) | 길이비 | 숫자환각 |\n|---|---|---|---|---|---|---|---|\n")
    for x in rows:
        f.write(f"| {x['문서']} | {x['사실']:.1f}% | {x['용어o']:.1f}% | {x['용어n']:.1f}% | "
                f"{x['근거o']:.1f}% | {x['근거n']:.1f}% | {x['길이비']:.0f}% | {x['숫자']} |\n")
    f.write(f"| **평균** | **{avg('사실'):.1f}%** | **{avg('용어o'):.1f}%** | **{avg('용어n'):.1f}%** | "
            f"**{avg('근거o'):.1f}%** | **{avg('근거n'):.1f}%** | **{avg('길이비'):.0f}%** | "
            f"**{sum(x['숫자'] for x in rows)}** |\n\n")
    f.write(f"## 숫자 환각 ({len(all_nums)}건 / {len({d for d,_,_ in all_nums})}개 문서)\n\n")
    for d,v,c in all_nums: f.write(f"- **{d}** `{v}` — …{c}…\n")
    f.write(f"\n## 메타 발화 ({len(all_meta)}건)\n\n")
    for d,m in all_meta: f.write(f"- {d}: {m}\n")
print(f"\n[저장] {OUT_MD}")
