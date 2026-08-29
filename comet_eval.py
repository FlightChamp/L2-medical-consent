#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comet_eval.py — 실제 동의서 번역에 CometKiwi 적용

대상: translate_eval.json (동의서 13건 × 영어·중국어·일본어)
      translate_demo.json (doc1 × 베트남어, 사례 분석분)

방법:
  원문과 번역문을 문장 단위로 정렬한 뒤, 각 쌍에 CometKiwi 점수를 매긴다.
  문서 점수는 문장 점수의 평균.
  CometKiwi는 참조 번역 없이 원문-번역문만으로 0~1 품질을 예측한다.
  (Rei et al., WMT 2022 — 품질 추정 공유과제 1위 모델)

문장 정렬:
  원문과 번역문의 문장 수가 다를 수 있으므로, 개수가 맞는 문서만 1:1로 쓰고
  다르면 짧은 쪽 길이에 맞춰 앞에서부터 자른다. 정렬 방식과 사용 문장 수를 함께 보고한다.
"""
import os, re, json, warnings, logging
warnings.filterwarnings("ignore")
for n in ("pytorch_lightning", "lightning.pytorch", "lightning"):
    logging.getLogger(n).setLevel(logging.ERROR)
from comet import download_model, load_from_checkpoint

HOME = os.path.expanduser("~/이윤우")
OUT  = os.path.join(HOME, "comet_result.json")
LANG = {"en": "영어", "zh": "중국어", "ja": "일본어", "vi": "베트남어"}

def sents(t):
    out = []
    for p in re.split(r"(?<=[.!?。！？])\s+|\n+", t):
        p = re.sub(r"\s+", " ", p).strip(" -–—:·*#")
        p = re.sub(r"\*+", "", p).strip()
        if len(p) >= 8: out.append(p)
    return out

rep = {}
for f in ("reportA.json", "reportB.json", "reportC.json"):
    p = os.path.join(HOME, f)
    if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))

TE = json.load(open(os.path.join(HOME, "translate_eval.json"), encoding="utf-8"))
TD = {}
p = os.path.join(HOME, "translate_demo.json")
if os.path.exists(p): TD = json.load(open(p, encoding="utf-8"))

print("[모델 로딩] Unbabel/wmt22-cometkiwi-da", flush=True)
model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

# 평가할 (문서, 언어, 원문, 번역문) 수집
JOBS = []
for doc, d in TE.items():
    src = rep[doc]["원문"]
    for code, v in d.get("언어", {}).items():
        JOBS.append((doc, code, src, v["출력"]))
for code_name, v in TD.items():
    code = {"English": "en", "Chinese": "zh", "Japanese": "ja", "Vietnamese": "vi"}.get(code_name)
    if code == "vi":
        JOBS.append(("doc1(사례)", "vi", rep["doc1"]["원문"], v["출력"]))

print(f"[대상] {len(JOBS)}건\n")

# 문장 쌍 만들기
DATA, INDEX = [], []
for doc, code, src, mt in JOBS:
    S, M = sents(src), sents(mt)
    n = min(len(S), len(M))
    if n < 3: continue
    for i in range(n):
        DATA.append({"src": S[i], "mt": M[i]})
        INDEX.append((doc, code, len(S), len(M), n))

print(f"[문장 쌍] {len(DATA)}개 채점 시작", flush=True)
out = model.predict(DATA, batch_size=32, gpus=1, progress_bar=True)

# 집계
from collections import defaultdict
agg = defaultdict(list); meta = {}
for (doc, code, ns, nm, n), sc in zip(INDEX, out.scores):
    agg[(doc, code)].append(sc)
    meta[(doc, code)] = (ns, nm, n)

RES, ROWS = {}, []
for (doc, code), scores in agg.items():
    ns, nm, n = meta[(doc, code)]
    avg = sum(scores) / len(scores)
    low = sorted(range(len(scores)), key=lambda i: scores[i])[:3]
    ROWS.append({"문서": doc, "언어": LANG[code], "code": code,
                 "원문문장": ns, "번역문장": nm, "사용": n, "점수": avg,
                 "최저": min(scores)})
    RES[f"{doc}_{code}"] = {"평균": avg, "문장수": n, "원문문장": ns, "번역문장": nm,
                            "최저점수문장": [{"src": DATA[0]["src"], "score": scores[i]} for i in low[:1]]}

print(f"\n{'='*80}\n[문서별 CometKiwi 점수]\n{'='*80}")
print(f"{'문서':<12}{'언어':<8}{'원문':>6}{'번역':>6}{'사용':>6}{'평균점수':>10}{'최저':>9}")
order = {"en": 0, "zh": 1, "ja": 2, "vi": 3}
for r in sorted(ROWS, key=lambda x: (len(x["문서"]), x["문서"], order[x["code"]])):
    print(f"{r['문서']:<12}{r['언어']:<8}{r['원문문장']:>6}{r['번역문장']:>6}"
          f"{r['사용']:>6}{r['점수']:>10.4f}{r['최저']:>9.4f}")

print(f"\n{'='*80}\n[언어별 평균]\n{'='*80}")
for code in ("en", "zh", "ja", "vi"):
    sub = [r for r in ROWS if r["code"] == code]
    if sub:
        m = sum(r["점수"] for r in sub) / len(sub)
        print(f"  {LANG[code]:<8} {m:.4f}   (문서 {len(sub)}건)")

json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[저장] {OUT}")
