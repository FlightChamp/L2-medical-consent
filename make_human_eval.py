#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_human_eval.py — 사람 검증용 판정 시트 생성

목적:
  NLI가 매긴 근거율 96.2%가 사람 판단과 얼마나 일치하는지 측정한다.
  자동 지표를 신뢰할 근거가 되고, 일치하지 않는 부분은 지표의 한계로 보고한다.

표본 설계:
  · NLI가 「근거 있음」으로 본 문장 20개  (무작위)
  · NLI가 「무근거」로 본 문장 전부        (12개)
  · NLI가 「모순」으로 본 문장 전부        (소수)
  → 두 부류를 섞고 순서를 무작위로 흩어 NLI 판정을 숨긴다.
    평가자가 어느 쪽인지 모르는 상태에서 판정해야 편향이 없다.

출력:
  human_eval_sheet.csv   평가자 3인이 각자 채워 넣을 시트
  human_eval_answer.json 채점용 정답키 (NLI 판정 + 정답 위치, 평가자에게 배포 금지)
"""
import os, re, json, random, csv

HOME = os.path.expanduser("~/이윤우")
random.seed(20260826)          # 재현 가능하도록 고정
N_SUPPORTED = 20               # 근거 있음 표본 수

rep = {}
for f in ("reportA.json", "reportB.json", "reportC.json"):
    p = os.path.join(HOME, f)
    if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))
nli = json.load(open(os.path.join(HOME, "nli_eval2_result.json"), encoding="utf-8"))

def split_sents(t):
    out = []
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+", " ", p).strip(" -–—:·*#")
        p = re.sub(r"\*+", "", p).strip()
        if len(p) >= 12 and len(re.findall(r"[가-힣]", p)) >= 6: out.append(p)
    return out

META_RE = [re.compile(p) for p in [
    r"^아래는", r"물론입니다", r"필요하시?면", r"드릴\s*수\s*있",
    r"원문에?\s*있는\s*모든", r"원문의?\s*모든\s*내용", r"다시\s*(작성|정리)한",
    r"추가\s*내용은?\s*없", r"이해할\s*수\s*있도록\s*(간단|명확)",
    r"^수술\s*동의서\s*\(", r"^\[?.*설명판\)?$", r"도움이\s*되"]]
is_meta = lambda s: any(r.search(s) for r in META_RE)

items = []
for doc, v in nli.items():
    uns = set(v.get("무근거문장", []))
    con = set(v.get("모순문장", []))
    body = [s for s in split_sents(rep[doc]["변환문"]) if not is_meta(s)]
    for s in body:
        lab = "무근거" if s in uns else ("모순" if s in con else "근거있음")
        items.append({"문서": doc, "문장": s, "NLI": lab})

sup = [x for x in items if x["NLI"] == "근거있음"]
oth = [x for x in items if x["NLI"] != "근거있음"]
random.shuffle(sup)
sample = sup[:N_SUPPORTED] + oth
random.shuffle(sample)

# 각 문장이 속한 문서의 원문도 함께 제공해야 판정 가능
SRC = {doc: rep[doc]["원문"] for doc in {x["문서"] for x in sample}}

sheet = os.path.join(HOME, "human_eval_sheet.csv")
with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["번호", "문서", "변환문 문장", "판정(1/2/3)", "메모"])
    for i, x in enumerate(sample, 1):
        w.writerow([i, x["문서"], x["문장"], "", ""])

key = os.path.join(HOME, "human_eval_answer.json")
json.dump({"표본": sample, "생성시각": "seed 20260826"},
          open(key, "w"), ensure_ascii=False, indent=1)

srcfile = os.path.join(HOME, "human_eval_원문.txt")
with open(srcfile, "w", encoding="utf-8") as f:
    for doc in sorted(SRC, key=lambda x: (len(x), x)):
        f.write(f"\n{'='*70}\n### {doc} 원문\n{'='*70}\n{SRC[doc]}\n")

print(f"[표본] 총 {len(sample)}문장")
print(f"  근거있음 {sum(1 for x in sample if x['NLI']=='근거있음')} / "
      f"무근거 {sum(1 for x in sample if x['NLI']=='무근거')} / "
      f"모순 {sum(1 for x in sample if x['NLI']=='모순')}")
print(f"  대상 문서 {len(SRC)}개")
print(f"\n[생성]")
print(f"  {sheet}          ← 평가자 3인에게 배포")
print(f"  {srcfile}   ← 원문 (판정 근거)")
print(f"  {key}     ← 정답키 (배포 금지)")
