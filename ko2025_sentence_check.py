#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ko2025_sentence_check.py — 원문 문장 분할이 이독성 개선폭을 얼마나 좌우하는지 확인 (조회 전용)

원문 문장 길이를 세 가지 방식으로 재고, HARI 도 같은 방식으로 재서 비교한다.
  ① 공백 연결   기존 방식. 줄을 공백으로 이어 마침표에서만 끊김 → 서식 항목이 뭉쳐 과대
  ② 줄바꿈 보존  PDF 줄바꿈마다 끊김 → 문장 중간의 줄바꿈까지 끊어 과소일 수 있음
  ③ Kiwi 문장 분리  형태소 분석기의 한국어 문장 분리기 (마침표 없이 끝나는 문장도 인식)
지수는 논문 방식 어휘 비율(형태소·중복 제외, policy A)로 계산한다.
"""
import glob, os, statistics as st, sys
HOME = os.environ.get("L2_HOME", os.path.expanduser("~/이윤우"))
sys.path.insert(0, HOME); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_ladder import Splitter
import ko2025_paper_method as P

pv = P.PaperVocab()
kiwi = pv.kiwi
def wps_kiwi(t):
    ss = [s.text.strip() for s in kiwi.split_into_sents(t) if len(s.text.strip()) >= 10]
    return sum(len(s.split()) for s in ss) / len(ss) if ss else None
def wps_split(t):
    return P.wps(t)[0]

docs = sorted(os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(HOME, "docs", "*.txt"))
              if not os.path.basename(p).startswith("syn"))
R = {k: [] for k in ("K_space", "K_line", "K_kiwi", "H_split", "H_kiwi", "rK", "rH")}
for d in docs:
    K, KL = P.load_K(d), P.load_K_lines(d)
    H = P.load_json(os.path.join(HOME, "outputs_prompt", d + "__P0.json"), "out")
    if not (K and KL and H):
        continue
    R["K_space"].append(wps_split(K)); R["K_line"].append(wps_split(KL))
    R["K_kiwi"].append(wps_kiwi(K.replace("\n", " ")))
    R["H_split"].append(wps_split(H)); R["H_kiwi"].append(wps_kiwi(H))
    R["rK"].append(pv.ratio(K)["ratio_A"]); R["rH"].append(pv.ratio(H)["ratio_A"])
m = {k: st.fmean([x for x in v if x is not None]) for k, v in R.items()}
I = lambda r, w: 61.994 - 0.261 * r - 1.045 * w
print("=" * 86)
print(f"원문 문장 분할 방식별 비교 — {len(R['rK'])}문서 평균, 지수는 논문 방식 어휘비율(A)")
print("=" * 86)
print(f"  {'방식':<26}{'원문 어절/문장':>14}{'원문 지수':>10}{'HARI 어절/문장':>15}{'HARI 지수':>10}{'개선폭':>8}")
for name, kk, hk in (("① 공백 연결 (기존)", "K_space", "H_split"),
                     ("② 줄바꿈 보존", "K_line", "H_split"),
                     ("③ Kiwi 문장 분리 (양쪽 동일)", "K_kiwi", "H_kiwi")):
    ik, ih = I(m["rK"], m[kk]), I(m["rH"], m[hk])
    print(f"  {name:<26}{m[kk]:>14.2f}{ik:>10.2f}{m[hk]:>15.2f}{ih:>10.2f}{ih - ik:>+8.2f}")
print("\n  원문 어절/문장 문서별 (①/②/③)")
for d, a, b, c in zip(docs, R["K_space"], R["K_line"], R["K_kiwi"]):
    print(f"    {d:<6} {a:6.2f} / {b:6.2f} / {c:6.2f}")
