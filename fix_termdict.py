#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_termdict.py — 대응어 다중 등록 + 오류 수정 후 재채점"""
import os, json, re
HOME = os.path.expanduser("~/이윤우")
RES = json.load(open(os.path.join(HOME,"translate_eval.json"), encoding="utf-8"))
LANG = {"en":"영어","zh":"중국어","ja":"일본어"}

# 수동 보정 사전: 용어당 허용 대응어 여러 개 (하나라도 있으면 적중)
TERM = {
 "갑상선":{"en":["thyroid"],"zh":["甲状腺"],"ja":["甲状腺"]},
 "신장":{"en":["kidney","renal","nephro"],"zh":["肾脏","肾"],"ja":["腎臓","腎"]},
 "관절":{"en":["joint","articular","arthro"],"zh":["关节"],"ja":["関節"]},
 "골절":{"en":["fracture"],"zh":["骨折"],"ja":["骨折"]},
 "인공관절":{"en":["artificial joint","prosthe","arthroplasty"],"zh":["人工关节"],"ja":["人工関節"]},
 "합병증":{"en":["complication"],"zh":["并发症"],"ja":["合併症"]},
 "후유증":{"en":["aftereffect","sequela","sequelae"],"zh":["后遗症"],"ja":["後遺症"]},
 "수혈":{"en":["transfusion"],"zh":["输血"],"ja":["輸血"]},
 "감염":{"en":["infection","infect"],"zh":["感染"],"ja":["感染"]},
 "염증":{"en":["inflammation","inflammatory"],"zh":["炎症","发炎"],"ja":["炎症"]},
 "출혈":{"en":["bleeding","hemorrhage","haemorrhage"],"zh":["出血"],"ja":["出血"]},
 "통증":{"en":["pain"],"zh":["疼痛","痛"],"ja":["痛み","疼痛"]},
 "재발":{"en":["recurrence","recur","relapse"],"zh":["复发"],"ja":["再発"]},
 "마취":{"en":["anesthe","anaesthe"],"zh":["麻醉"],"ja":["麻酔"]},
 "봉합":{"en":["sutur","clos"],"zh":["缝合"],"ja":["縫合"]},
 "절제":{"en":["resect","excis","remov","ectomy"],"zh":["切除"],"ja":["切除"]},
 "종양":{"en":["tumor","tumour","neoplas"],"zh":["肿瘤"],"ja":["腫瘍"]},
 "척추":{"en":["spinal","spine","vertebra"],"zh":["脊椎","脊柱","脊髓"],"ja":["脊椎","脊髄"]},
 "전립선":{"en":["prostate","prostatic"],"zh":["前列腺"],"ja":["前立腺"]},
 "방광":{"en":["bladder"],"zh":["膀胱"],"ja":["膀胱"]},
 "자궁":{"en":["uterus","uterine","cervix","cervical"],"zh":["子宫"],"ja":["子宮"]},
 "유방":{"en":["breast","mammar"],"zh":["乳房","乳腺"],"ja":["乳房","乳腺"]},
 "담낭":{"en":["gallbladder","cholecyst"],"zh":["胆囊"],"ja":["胆囊","胆のう"]},
 "탈장":{"en":["hernia"],"zh":["疝","疝气"],"ja":["ヘルニア","脱腸"]},
 "치핵":{"en":["hemorrhoid","haemorrhoid"],"zh":["痔疮","痔核"],"ja":["痔核"]},
 "기흉":{"en":["pneumothorax"],"zh":["气胸"],"ja":["気胸"]},
 "기관지":{"en":["bronch"],"zh":["支气管"],"ja":["気管支"]},
 "식도":{"en":["esophag","oesophag"],"zh":["食道","食管"],"ja":["食道"]},
 "생검":{"en":["biopsy"],"zh":["活检"],"ja":["生検"]},
 "조직검사":{"en":["biopsy","histolog","patholog"],"zh":["活检","组织检查"],"ja":["生検","組織検査"]},
}
rep = {}
for f in ["reportA.json","reportB.json","reportC.json"]:
    p=os.path.join(HOME,f)
    if os.path.exists(p): rep.update(json.load(open(p,encoding="utf-8")))

rows, miss = [], []
for doc, d in sorted(RES.items(), key=lambda x:(len(x[0]),x[0])):
    SRC = rep[doc]["원문"]
    present = [t for t in TERM if t in SRC]
    for code, v in d["언어"].items():
        out = v["출력"]; low = out.lower()
        hit=n=0
        for t in present:
            cands = TERM[t].get(code, [])
            if not cands: continue
            n += 1
            if any(c.lower() in low for c in cands): hit += 1
            else: miss.append((doc, code, t, cands))
        if n: rows.append((doc, LANG[code], n, hit, hit/n*100))

print(f"{'문서':<8}{'언어':<8}{'대상':>5}{'적중':>5}{'정확도':>9}")
for r in rows: print(f"{r[0]:<8}{r[1]:<8}{r[2]:>5}{r[3]:>5}{r[4]:>8.0f}%")
print()
for code,kor in LANG.items():
    sub=[r for r in rows if r[1]==kor]
    if sub:
        tot=sum(r[2] for r in sub); h=sum(r[3] for r in sub)
        print(f"[{kor}] {h}/{tot} = {h/tot*100:.1f}%")
print(f"\n[미검출] 총 {len(miss)}건")
for doc,code,t,c in miss:
    print(f"  {doc} {LANG[code]}: {t} → {c}")
