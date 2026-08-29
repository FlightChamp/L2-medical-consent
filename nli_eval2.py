#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nli_eval2.py — NLI 채점 v2 (메타 발화 분리 + 원문 청크 묶음)

v1의 두 가지 한계를 단계적으로 보정하고, 각 단계 결과를 함께 출력한다.

  단계 A (v1)   원문 1문장 vs 변환 1문장
  단계 B        변환문에서 메타 발화(챗봇 인사말·작업 설명·제목)를 제거 후 재계산
                → 메타 발화는 문서 내용이 아니므로 충실도 분모에서 빼고 건수만 따로 보고
  단계 C        원문을 1~3문장 묶음(청크)으로도 전제에 넣어 재계산
                → 모델이 원문 여러 문장을 한 문장으로 합친 경우를 인식하기 위함
                  (SummaC 논문도 문서를 청크로 나누는 변형을 다룸)

세 단계 수치를 나란히 보여주므로, 지표를 어떻게 다듬었고
그때마다 값이 얼마나 움직였는지 그대로 확인할 수 있다.
"""
import os, re, json, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI  = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HOME = os.path.expanduser("~/이윤우")
OUT  = os.path.join(HOME, "nli_eval2_result.json")
TH, BATCH, MAXCHUNK = 0.5, 96, 3

# 메타 발화 판별 패턴 — 문서 내용이 아니라 모델이 덧붙인 말
META = [
    r"^아래는", r"물론입니다", r"필요하시?면", r"드릴\s*수\s*있",
    r"원문에?\s*있는\s*모든", r"원문의?\s*모든\s*내용", r"다시\s*(작성|정리)한",
    r"추가\s*내용은?\s*없", r"이해할\s*수\s*있도록\s*(간단|명확)",
    r"^수술\s*동의서\s*\(", r"^\[?.*설명판\)?$", r"도움이\s*되",
]
META_RE = [re.compile(p) for p in META]
def is_meta(s):
    return any(r.search(s) for r in META_RE)

def split_sents(t):
    out = []
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+", " ", p).strip(" -–—:·*#")
        p = re.sub(r"\*+", "", p).strip()
        if len(p) >= 12 and len(re.findall(r"[가-힣]", p)) >= 6:
            out.append(p)
    return out

def chunks(sents, k=MAXCHUNK):
    """1문장 ~ k문장 연속 묶음 전부"""
    out = list(sents)
    for n in range(2, k + 1):
        for i in range(len(sents) - n + 1):
            out.append(" ".join(sents[i:i + n]))
    return out

data = {}
for f in ("reportA.json", "reportB.json", "reportC.json"):
    p = os.path.join(HOME, f)
    if os.path.exists(p): data.update(json.load(open(p, encoding="utf-8")))
print(f"[불러옴] 문서 {len(data)}개")

print("[NLI 모델 로딩]", flush=True)
tok = AutoTokenizer.from_pretrained(NLI)
mdl = AutoModelForSequenceClassification.from_pretrained(NLI).to("cuda:0").eval()
L2I = {v: k for k, v in mdl.config.id2label.items()}
I_E, I_C = L2I["entailment"], L2I["contradiction"]

@torch.no_grad()
def best_scores(prems, hyps):
    """가설별 (entailment 최댓값, contradiction 최댓값)"""
    e = torch.zeros(len(hyps)); c = torch.zeros(len(hyps))
    pairs = [(pi, hi) for hi in range(len(hyps)) for pi in range(len(prems))]
    for s in range(0, len(pairs), BATCH):
        ch = pairs[s:s + BATCH]
        enc = tok([prems[pi] for pi, _ in ch], [hyps[hi] for _, hi in ch],
                  return_tensors="pt", truncation=True, max_length=256,
                  padding=True).to("cuda:0")
        pr = torch.softmax(mdl(**enc).logits.float(), dim=-1).cpu()
        for k, (pi, hi) in enumerate(ch):
            e[hi] = max(e[hi], pr[k, I_E]); c[hi] = max(c[hi], pr[k, I_C])
    return e, c

def rates(e, c):
    sup = (e >= TH); con = (~sup) & (c >= TH); uns = ~(sup | con)
    n = max(len(e), 1)
    return (sup.float().sum().item()/n*100, con.float().sum().item()/n*100,
            uns.float().sum().item()/n*100, e.mean().item())

RES, ROWS = {}, []
t0 = time.time()
for name in sorted(data, key=lambda x: (len(x), x)):
    SRC, OUTT = data[name]["원문"], data[name]["변환문"]
    prem1 = split_sents(SRC)
    hyps_all = split_sents(OUTT)
    if not prem1 or not hyps_all: continue

    metas = [h for h in hyps_all if is_meta(h)]
    hyps  = [h for h in hyps_all if not is_meta(h)]
    if not hyps: continue

    eA, cA = best_scores(prem1, hyps_all)          # A: 원본
    eB, cB = best_scores(prem1, hyps)              # B: 메타 제거
    premC  = chunks(prem1)
    eC, cC = best_scores(premC, hyps)              # C: + 청크 묶음

    A, B, C = rates(eA, cA), rates(eB, cB), rates(eC, cC)
    sup = (eC >= TH); con = (~sup) & (cC >= TH); uns = ~(sup | con)
    row = {"문서": name, "원문": len(prem1), "변환": len(hyps_all),
           "메타": len(metas), "본문": len(hyps),
           "A근거": A[0], "B근거": B[0], "C근거": C[0],
           "C모순": C[1], "C무근거": C[2], "C점수": C[3]}
    ROWS.append(row)
    RES[name] = dict(row,
        메타문장=metas,
        무근거문장=[hyps[i] for i in range(len(hyps)) if uns[i]],
        모순문장=[hyps[i] for i in range(len(hyps)) if con[i]])
    print(f"  [{name}] 메타 {len(metas)}건 제거 | 근거율 "
          f"A {A[0]:.1f}% → B {B[0]:.1f}% → C {C[0]:.1f}%", flush=True)

n = len(ROWS); avg = lambda k: sum(r[k] for r in ROWS)/n
print(f"\n{'='*82}\n[단계별 근거율]  ({time.time()-t0:.0f}초)\n{'='*82}")
print(f"{'문서':<8}{'원문':>5}{'본문':>5}{'메타':>5}"
      f"{'A 원본':>9}{'B 메타제거':>11}{'C +청크':>10}{'C모순':>8}{'C무근거':>9}")
for r in ROWS:
    print(f"{r['문서']:<8}{r['원문']:>5}{r['본문']:>5}{r['메타']:>5}"
          f"{r['A근거']:>8.1f}%{r['B근거']:>10.1f}%{r['C근거']:>9.1f}%"
          f"{r['C모순']:>7.1f}%{r['C무근거']:>8.1f}%")
print("-"*82)
print(f"{'평균':<8}{'':>5}{'':>5}{sum(r['메타'] for r in ROWS):>5}"
      f"{avg('A근거'):>8.1f}%{avg('B근거'):>10.1f}%{avg('C근거'):>9.1f}%"
      f"{avg('C모순'):>7.1f}%{avg('C무근거'):>8.1f}%")
print(f"\n개선폭:  메타 제거 +{avg('B근거')-avg('A근거'):.1f}%p   "
      f"청크 추가 +{avg('C근거')-avg('B근거'):.1f}%p   "
      f"합계 +{avg('C근거')-avg('A근거'):.1f}%p")

print(f"\n{'='*82}\n[메타 발화 — 문서 내용이 아닌 모델의 덧붙임]\n{'='*82}")
tot_meta = 0
for name, v in RES.items():
    if v["메타문장"]:
        tot_meta += len(v["메타문장"])
        print(f"\n── {name} ({len(v['메타문장'])}건)")
        for s in v["메타문장"][:3]: print(f"   · {s[:95]}")
print(f"\n  전체 {tot_meta}건 / {n}개 문서")

print(f"\n{'='*82}\n[최종 무근거 문장 — 보정 후에도 원문에 근거 없음]\n{'='*82}")
for name, v in RES.items():
    if v["무근거문장"]:
        print(f"\n── {name} ({v['C무근거']:.0f}%)")
        for s in v["무근거문장"][:6]: print(f"   · {s[:95]}")

json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[저장] {OUT}")
