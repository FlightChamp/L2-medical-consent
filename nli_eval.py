#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nli_eval.py — 실제 동의서 13건의 평이화 결과를 NLI로 채점

방법 (SummaC, Laban et al. 2022, TACL 방식):
  원문을 문장 M개, 변환문을 문장 N개로 나눈다.
  M×N 모든 쌍에 대해 「원문 문장이 변환문 문장을 함의하는가」를 NLI로 판정한다.
  변환문 문장마다 가장 잘 맞는 원문 문장의 entailment 점수를 취하고(최댓값),
  그 평균을 문서의 근거율로 삼는다.

지표 세 가지:
  · 근거율        변환문 문장 중 원문에 근거가 있는 비율 (entailment 최댓값 ≥ 0.5)
  · 모순율        원문과 모순되는 문장의 비율 (contradiction 최댓값 ≥ 0.5)
  · 무근거율      원문에 없는 내용을 지어낸 비율 (근거도 모순도 아님)

판정 모델은 생성 모델(hari)과 완전히 별개다.
"""
import os, re, json, time, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HOME = os.path.expanduser("~/이윤우")
OUT = os.path.join(HOME, "nli_eval_result.json")
THRESH = 0.5
BATCH = 64

ap = argparse.ArgumentParser()
ap.add_argument("--reports", default="reportA.json,reportB.json,reportC.json")
ap.add_argument("--tag", default="hari")
args = ap.parse_args()

def split_sents(t):
    out = []
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+", " ", p).strip(" -–—:·*#")
        p = re.sub(r"\*+", "", p).strip()
        if len(p) >= 12 and len(re.findall(r"[가-힣]", p)) >= 6:
            out.append(p)
    return out

data = {}
for f in args.reports.split(","):
    p = os.path.join(HOME, f.strip())
    if os.path.exists(p): data.update(json.load(open(p, encoding="utf-8")))
if not data: raise SystemExit("[중단] 결과 파일 없음")
print(f"[불러옴] 문서 {len(data)}개")

print("[NLI 모델 로딩]", NLI, flush=True)
tok = AutoTokenizer.from_pretrained(NLI)
mdl = AutoModelForSequenceClassification.from_pretrained(NLI).to("cuda:0").eval()
L2I = {v: k for k, v in mdl.config.id2label.items()}
I_ENT, I_CON = L2I["entailment"], L2I["contradiction"]

@torch.no_grad()
def nli_matrix(prems, hyps):
    """(len(hyps), len(prems)) 크기의 entailment / contradiction 확률 행렬"""
    ent = torch.zeros(len(hyps), len(prems))
    con = torch.zeros(len(hyps), len(prems))
    pairs = [(pi, hi) for hi in range(len(hyps)) for pi in range(len(prems))]
    for s in range(0, len(pairs), BATCH):
        chunk = pairs[s:s + BATCH]
        a = [prems[pi] for pi, _ in chunk]
        b = [hyps[hi] for _, hi in chunk]
        enc = tok(a, b, return_tensors="pt", truncation=True,
                  max_length=256, padding=True).to("cuda:0")
        pr = torch.softmax(mdl(**enc).logits.float(), dim=-1).cpu()
        for k, (pi, hi) in enumerate(chunk):
            ent[hi, pi] = pr[k, I_ENT]; con[hi, pi] = pr[k, I_CON]
    return ent, con

RES, ROWS = {}, []
t0 = time.time()
for name in sorted(data, key=lambda x: (len(x), x)):
    SRC, OUTT = data[name]["원문"], data[name]["변환문"]
    prems, hyps = split_sents(SRC), split_sents(OUTT)
    if not prems or not hyps:
        print(f"  [{name}] 문장 추출 실패 — 건너뜀"); continue
    ent, con = nli_matrix(prems, hyps)
    e_max = ent.max(dim=1).values
    c_max = con.max(dim=1).values
    supported = (e_max >= THRESH)
    contra = (~supported) & (c_max >= THRESH)
    unsup = ~(supported | contra)
    n = len(hyps)
    row = {"문서": name, "원문문장": len(prems), "변환문장": n,
           "근거율": supported.float().mean().item() * 100,
           "모순율": contra.float().mean().item() * 100,
           "무근거율": unsup.float().mean().item() * 100,
           "평균근거점수": e_max.mean().item()}
    ROWS.append(row)
    RES[name] = dict(row, 무근거문장=[hyps[i] for i in range(n) if unsup[i]][:10],
                          모순문장=[hyps[i] for i in range(n) if contra[i]][:10])
    print(f"  [{name}] 원문 {len(prems)}문장 → 변환 {n}문장 | "
          f"근거 {row['근거율']:.1f}% / 모순 {row['모순율']:.1f}% / "
          f"무근거 {row['무근거율']:.1f}%", flush=True)

print(f"\n{'='*78}\n[요약] {args.tag}  ({time.time()-t0:.0f}초)\n{'='*78}")
print(f"{'문서':<8}{'원문':>6}{'변환':>6}{'근거율':>9}{'모순율':>9}{'무근거율':>10}{'평균점수':>10}")
for r in ROWS:
    print(f"{r['문서']:<8}{r['원문문장']:>6}{r['변환문장']:>6}"
          f"{r['근거율']:>8.1f}%{r['모순율']:>8.1f}%{r['무근거율']:>9.1f}%{r['평균근거점수']:>10.3f}")
n = len(ROWS); avg = lambda k: sum(r[k] for r in ROWS) / n
print("-" * 78)
print(f"{'평균':<8}{'':>6}{'':>6}{avg('근거율'):>8.1f}%{avg('모순율'):>8.1f}%"
      f"{avg('무근거율'):>9.1f}%{avg('평균근거점수'):>10.3f}")

print(f"\n{'='*78}\n[무근거 문장 — 원문에 근거가 없는 내용]\n{'='*78}")
for name, v in RES.items():
    if v["무근거문장"]:
        print(f"\n── {name} ({v['무근거율']:.0f}%)")
        for s in v["무근거문장"][:5]: print(f"   · {s[:100]}")

print(f"\n{'='*78}\n[모순 문장 — 원문과 반대되는 내용]\n{'='*78}")
any_c = False
for name, v in RES.items():
    if v["모순문장"]:
        any_c = True
        print(f"\n── {name} ({v['모순율']:.0f}%)")
        for s in v["모순문장"][:5]: print(f"   · {s[:100]}")
if not any_c: print("  없음")

json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[저장] {OUT}")
