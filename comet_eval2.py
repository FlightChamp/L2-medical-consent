#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comet_eval2.py — CometKiwi 재적용 (문장 정렬 문제 보정)

v1의 문제:
  원문과 번역문의 문장 수가 달라(46 vs 34 등) 앞에서부터 1:1로 짝지으면
  중간부터 전부 어긋난다. 그 결과 점수가 실제보다 크게 낮게 나왔다.
  (사전 시험에서 정상 번역은 0.88이었는데 v1에서는 0.48~0.73)

v2의 방법 — 두 가지를 함께 계산해 비교한다:
  A. 문서 단위   원문 전체와 번역문 전체를 한 쌍으로 넣는다. 정렬 문제가 없다.
  B. 청크 단위   원문과 번역문을 각각 N등분해 같은 순번끼리 비교한다.
                 문장 수가 달라도 문단 수준에서는 대응이 유지된다.

두 값이 비슷하면 신뢰할 수 있고, 크게 다르면 그 사실을 함께 보고한다.
"""
import os, re, json, warnings, logging
warnings.filterwarnings("ignore")
for n in ("pytorch_lightning", "lightning.pytorch", "lightning"):
    logging.getLogger(n).setLevel(logging.ERROR)
from comet import download_model, load_from_checkpoint

HOME = os.path.expanduser("~/이윤우")
OUT  = os.path.join(HOME, "comet_result2.json")
LANG = {"en": "영어", "zh": "중국어", "ja": "일본어", "vi": "베트남어"}
NCHUNK = 6          # 문서를 몇 등분해 비교할지

def clean(t):
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"\n{2,}", "\n", t).strip()

def nchunks(t, n):
    """텍스트를 문자 수 기준 n등분"""
    t = clean(t); L = len(t)
    if L == 0: return []
    step = max(L // n, 1)
    return [t[i:i+step] for i in range(0, L, step)][:n]

rep = {}
for f in ("reportA.json", "reportB.json", "reportC.json"):
    p = os.path.join(HOME, f)
    if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))
TE = json.load(open(os.path.join(HOME, "translate_eval.json"), encoding="utf-8"))
TD = {}
p = os.path.join(HOME, "translate_demo.json")
if os.path.exists(p): TD = json.load(open(p, encoding="utf-8"))

print("[모델 로딩]", flush=True)
model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

JOBS = []
for doc, d in TE.items():
    for code, v in d.get("언어", {}).items():
        JOBS.append((doc, code, rep[doc]["원문"], v["출력"]))
for name, v in TD.items():
    if name == "Vietnamese":
        JOBS.append(("doc1(사례)", "vi", rep["doc1"]["원문"], v["출력"]))
print(f"[대상] {len(JOBS)}건\n")

# A: 문서 단위
dataA = [{"src": clean(s), "mt": clean(m)} for _, _, s, m in JOBS]
print("[A] 문서 단위 채점", flush=True)
outA = model.predict(dataA, batch_size=8, gpus=1, progress_bar=True).scores

# B: 청크 단위
dataB, idxB = [], []
for k, (doc, code, s, m) in enumerate(JOBS):
    cs, cm = nchunks(s, NCHUNK), nchunks(m, NCHUNK)
    n = min(len(cs), len(cm))
    for i in range(n):
        dataB.append({"src": cs[i], "mt": cm[i]}); idxB.append(k)
print(f"\n[B] 청크 단위 채점 ({len(dataB)}쌍)", flush=True)
outB = model.predict(dataB, batch_size=16, gpus=1, progress_bar=True).scores

from collections import defaultdict
bag = defaultdict(list)
for k, sc in zip(idxB, outB): bag[k].append(sc)

ROWS = []
for k, (doc, code, s, m) in enumerate(JOBS):
    b = sum(bag[k]) / len(bag[k]) if bag[k] else 0
    ROWS.append({"문서": doc, "code": code, "언어": LANG[code],
                 "A문서": outA[k], "B청크": b, "차이": abs(outA[k] - b)})

print(f"\n{'='*76}\n[문서별 점수 — 두 방식 비교]\n{'='*76}")
print(f"{'문서':<12}{'언어':<8}{'A 문서단위':>11}{'B 청크단위':>11}{'차이':>9}")
order = {"en": 0, "zh": 1, "ja": 2, "vi": 3}
for r in sorted(ROWS, key=lambda x: (len(x["문서"]), x["문서"], order[x["code"]])):
    flag = "  ⚠" if r["차이"] > 0.1 else ""
    print(f"{r['문서']:<12}{r['언어']:<8}{r['A문서']:>11.4f}{r['B청크']:>11.4f}{r['차이']:>9.4f}{flag}")

print(f"\n{'='*76}\n[언어별 평균]\n{'='*76}")
print(f"{'언어':<10}{'A 문서단위':>12}{'B 청크단위':>12}{'문서수':>8}")
for code in ("en", "zh", "ja", "vi"):
    sub = [r for r in ROWS if r["code"] == code]
    if sub:
        a = sum(r["A문서"] for r in sub)/len(sub)
        b = sum(r["B청크"] for r in sub)/len(sub)
        print(f"{LANG[code]:<10}{a:>12.4f}{b:>12.4f}{len(sub):>8}")

json.dump([{k: v for k, v in r.items()} for r in ROWS],
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[저장] {OUT}")
