#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""numeric_verifier v2 — 단위를 가진 수치만 대상으로 보존/변조/창작 검증
   v1의 실패: 섹션 번호(4., 7))를 수치로 집계 → 단위 필수 조건으로 해결"""
import os, re, json

HOME = os.path.expanduser("~/이윤우")
OUT_MD = os.path.join(HOME, "numeric_report.md")

data = {}
for fn in ["reportA.json", "reportB.json", "reportC.json"]:
    p = os.path.join(HOME, fn)
    if os.path.exists(p): data.update(json.load(open(p, encoding="utf-8")))
if not data: raise SystemExit("[중단] reportA/B/C.json 없음")

UNIT = (r"(시간|분|초|일|주일|주|개월|달|년|세|％|%|퍼센트|cc|ml|mL|mg|g|kg|"
        r"cm|mm|회|번|명|개월간|배|도|kPa)")
# 범위(5~7일, 0.1~1%)도 하나의 수치로 취급. 단위 필수.
NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:[~∼\-–]\s*(\d+(?:[.,]\d+)?))?\s*" + UNIT)

def norm_tok(m):
    a, b, u = m.group(1), m.group(2), m.group(3)
    u = "%" if u in ("％", "퍼센트") else u
    return (f"{a}~{b}{u}" if b else f"{a}{u}")

def spans(text):
    out = []
    for m in NUM.finditer(text):
        out.append({"tok": norm_tok(m), "pos": m.start(),
                    "ctx": " ".join(text[max(0, m.start()-45):m.end()+45].split())})
    return out

def keywords(text, pos, span=35):
    return [w for w in re.findall(r"[가-힣]{3,8}", text[max(0,pos-span):pos+span])]

rows, hits = [], []
for name, r in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    SRC, OUT = r["원문"], r["변환문"]
    s_nums, o_nums = spans(SRC), spans(OUT)
    o_toks = {x["tok"] for x in o_nums}
    s_toks = {x["tok"] for x in s_nums}

    kept, changed, lost = [], [], []
    for s in s_nums:
        if s["tok"] in o_toks: kept.append(s); continue
        cand = None
        for kw in keywords(SRC, s["pos"]):
            for m in re.finditer(re.escape(kw), OUT):
                near = [o for o in o_nums if abs(o["pos"]-m.start()) <= 60]
                if near: cand = (kw, near[0]); break
            if cand: break
        if cand and cand[1]["tok"] not in s_toks:
            changed.append((s, cand[1]))
        else:
            lost.append(s)

    made, seen = [], set()
    for o in o_nums:
        if o["tok"] in s_toks or o["tok"] in seen: continue
        seen.add(o["tok"]); made.append(o)

    tot = len(s_nums)
    rows.append({"문서":name, "수치":tot, "보존":len(kept), "변조":len(changed),
                 "소실":len(lost), "창작":len(made),
                 "보존율": len(kept)/tot*100 if tot else 100.0})
    for s,o in changed: hits.append((name,"변조",f"{s['tok']} → {o['tok']}", s["ctx"]))
    for o in made:      hits.append((name,"창작",o["tok"], o["ctx"]))

h=f"{'문서':<8}{'단위수치':>9}{'보존':>6}{'변조':>6}{'소실':>6}{'창작':>6}{'보존율':>9}"
print(h); print("-"*len(h))
for x in rows:
    flag = "  ⚠" if (x["변조"] or x["창작"]) else ""
    print(f"{x['문서']:<8}{x['수치']:>9}{x['보존']:>6}{x['변조']:>6}"
          f"{x['소실']:>6}{x['창작']:>6}{x['보존율']:>8.1f}%{flag}")
print("-"*len(h))
tot=lambda k: sum(x[k] for x in rows)
print(f"{'합계':<8}{tot('수치'):>9}{tot('보존'):>6}{tot('변조'):>6}{tot('소실'):>6}"
      f"{tot('창작'):>6}{tot('보존')/max(tot('수치'),1)*100:>8.1f}%")

print("\n"+"="*76); print(f"[검출] 변조 {tot('변조')}건 / 창작 {tot('창작')}건"); print("="*76)
for d,k,w,c in hits:
    print(f"  {'🔴' if k=='변조' else '🟠'} [{d}] {k}: {w}")
    print(f"      …{c[:115]}…")
bad = len({d for d,_,_,_ in hits})
print(f"\n  → 13개 문서 중 {bad}개에서 수치 오류")

with open(OUT_MD,"w",encoding="utf-8") as f:
    f.write("# 수치 대조 검증 결과 (단위 보유 수치 대상)\n\n")
    f.write("| 문서 | 단위 수치 | 보존 | 변조 | 소실 | 창작 | 보존율 |\n|---|---|---|---|---|---|---|\n")
    for x in rows:
        f.write(f"| {x['문서']} | {x['수치']} | {x['보존']} | {x['변조']} | {x['소실']} | "
                f"{x['창작']} | {x['보존율']:.1f}% |\n")
    f.write(f"| **합계** | **{tot('수치')}** | **{tot('보존')}** | **{tot('변조')}** | "
            f"**{tot('소실')}** | **{tot('창작')}** | "
            f"**{tot('보존')/max(tot('수치'),1)*100:.1f}%** |\n\n")
    f.write(f"## 검출 사례 ({len(hits)}건 / {bad}개 문서)\n\n")
    for d,k,w,c in hits: f.write(f"- **{d}** {k}: `{w}`\n  - …{c[:130]}…\n")
print(f"\n[저장] {OUT_MD}")
