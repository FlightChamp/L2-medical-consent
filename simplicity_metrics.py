#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simplicity_metrics.py — 평이성 지표
====================================
설계 원칙:
    · 모든 지표에 방향(↑ 좋음 / ↓ 좋음 / 해석주의)과 분류를 명시한다.
    · 공식이 확보되지 않은 지표는 임의 구현하지 않고 None 을 돌려준다.
    · Safety 지표와 합쳐 단일 총점을 만들지 않는다.

지표 분류:
    [표준]        학계에서 정의가 확립된 것
    [논문공식]    특정 논문의 수식을 그대로 구현한 것
    [프로젝트정의] 본 과제에서 정의한 것. 계산식과 함께 제시해야 의미가 성립

공식 확보 현황 (2026-09-15 기준):
    확보   조용구(2016) 국어 이독성 공식 — 지수 확정, 어휘 목록 미확보
              GL = 4.874 + 0.591·A − 9.201·B^3
                A = 평균 문장길이
                B = 5,000단어 목록에 포함되는 단어의 비율
              (1) 지수 3 은 세제곱이 맞다. KISS 영문 초록에서 B^3 으로 확인됨.
              (2) 5,000단어 목록의 출처·내용이 원문에서 미확인.
           → (2) 가 풀릴 때까지 CHO2016_ENABLED = False 로 둔다.
              CHO2016_CUBED 는 True 로 확정.

    최우선  고승연(2025) 「제2언어 학습자를 위한 한국어 읽기 텍스트 이독성 공식
           개발 및 수준 환산」 새국어교육 143, pp.245-272, 한국국어교육학회
           DOI 10.15734/koed..143.202506.245
           L2 학습자용 이독성 공식과 숙달도 수준 환산을 직접 개발한 연구이므로
           본 과제에 가장 적합하다. PDF 확보 후 공식·변수 정의·필요 어휘/문법
           자원·수준 환산표를 확인한 뒤 구현 여부를 정한다.
           초록만 보고 공식을 추정하지 않는다.

    보류   KReaD — 대교 특허. 어휘 등급 목록 28,332단어 비공개
    보류   이보미(2020) 한국어 읽기 텍스트 난이도 공식 — PDF 미확보
    보류   Suh et al.(2013) 텍스트 복잡도 — PDF 미확보

    해당없음  Natmal — (주)낱말의 상용 '문장검사프로그램'. 지표가 아니라 도구다.
              논문에서는 어절 수 계산에만 사용되었다.

즉시 구현 가능한 지표만으로 Phase 1 을 시작하고, 논문이 확보되는 대로 추가한다.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from prompt_ladder import Splitter, jaccard3          # noqa: E402

# ===========================================================================
# 공식 미확보 지표 스위치 — 확보 전에는 켜지 않는다
# ===========================================================================
CHO2016_ENABLED = False          # 조용구(2016). 5,000단어 목록 확보 시 True
CHO2016_CUBED = True             # 확정. KISS 영문 초록에서 B^3 확인
EASY_WORD_LIST_PATH = None       # 쉬운 단어 목록 파일 경로

# ===========================================================================
# 지표 정의 — 이름: (방향, 분류, 설명)
#   dir: "lower"  값이 낮을수록 쉬움
#        "higher" 값이 높을수록 쉬움
#        "info"   방향을 단정할 수 없음. 해석에 주의
# ===========================================================================
# ===========================================================================
# 고승연(2025) L2 이독성 공식
# ===========================================================================
# 고승연(2025), 「제2언어 학습자를 위한 한국어 읽기 텍스트 이독성 공식 개발 및
# 수준 환산」, 새국어교육 143, pp.245-272. DOI 10.15734/koed..143.202506.245
#
#   이독성 지수 = 61.994 - 0.261 x 중급이상어휘비율 - 1.045 x 평균어절수
#
# 방향: 값이 **높을수록 쉬움** (종속변수 = 학습자 성적)
# 모형: R = .993, R2 = .986, 수정 R2 = .982, F(2,12) = 253.87, p < .001
#
# 수준 환산표 (논문 [표 8])
#   >= 45        쉬움    대부분의 중급 학습자가 이해 가능
#   35 - 44.99   보통    중급 이상 학습자 권장
#   < 35         어려움  고급 학습자 또는 지도자용
#
# 어휘 목록: 국립국어원 2017년 국제 통용 한국어 표준 교육과정 적용 연구(4단계),
#            김중섭 외 13명(2017). build_vocab_grades.py 로 사전 생성.
#
# 미등재 어휘 처리는 논문에 명시되지 않았다. 두 정책을 모두 계산해 병기한다.
#   policy A  미등재를 중급 이상으로 계산 (기본)
#   policy B  미등재를 분모에서 제외

KO2025 = {
    "const": 61.994,
    "coef_mid_vocab": -0.261,
    "coef_words_per_sent": -1.045,
    "direction": "higher_easier",
    "source": ("고승연(2025), 새국어교육 143, pp.245-272, "
               "DOI 10.15734/koed..143.202506.245"),
    "model_fit": "R=.993, R2=.986, adj R2=.982, F(2,12)=253.87, p<.001",
    "levels": [(45.0, "쉬움"), (35.0, "보통")],
    "unresolved": ("논문에 미등재 어휘 처리 방식이 명시되지 않음. "
                   "policy A(미등재=중급이상) / B(분모 제외) 병기"),
}

_L2_HOME = os.path.expanduser("~/이윤우")
VOCAB_PATH = os.environ.get("VOCAB_GRADES",
                            os.path.join(_L2_HOME, "vocab_grades.json"))
if not os.path.exists(VOCAB_PATH):
    _local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vocab_grades.json")
    if os.path.exists(_local):
        VOCAB_PATH = _local

_VOCAB = None
_VOCAB_TRIED = False
_MIN_STEM = 2      # 어절에서 조사·어미를 떼어낼 때 남길 최소 길이


def _load_vocab():
    global _VOCAB, _VOCAB_TRIED
    if _VOCAB_TRIED:
        return _VOCAB
    _VOCAB_TRIED = True
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, encoding="utf-8") as f:
                _VOCAB = json.load(f).get("by_word")
        except Exception:
            _VOCAB = None
    return _VOCAB


def _grade_of(eojeol, vocab):
    """어절의 어휘 등급. 형태소 분석기 없이 최장 일치로 표제어를 찾는다."""
    w = re.sub(r"[^가-힣A-Za-z]", "", eojeol)
    if not w:
        return None
    if w in vocab:
        return vocab[w]
    for cut in range(len(w) - 1, _MIN_STEM - 1, -1):
        if w[:cut] in vocab:
            return vocab[w[:cut]]
    return None


def mid_vocab_ratio(text, policy="A"):
    """중급 이상(3-6급) 어휘 비율 %. (비율, 분모, 미등재수) 또는 None."""
    vocab = _load_vocab()
    if not vocab:
        return None
    eojeols = [w for w in str(text).split() if re.search(r"[가-힣]", w)]
    if not eojeols:
        return None
    mid = low = unlisted = 0
    for e in eojeols:
        g = _grade_of(e, vocab)
        if g is None:
            unlisted += 1
        elif g >= 3:
            mid += 1
        else:
            low += 1
    if policy == "A":
        num, den = mid + unlisted, mid + low + unlisted
    else:
        num, den = mid, mid + low
    if den == 0:
        return None
    return (round(100.0 * num / den, 2), den, unlisted)


def ko2025_index(text, policy="A"):
    """고승연(2025) 이독성 지수. 높을수록 쉽다. None 이면 어휘 사전 없음."""
    mv = mid_vocab_ratio(text, policy)
    if mv is None:
        return None
    ss = Splitter.split(text)
    if not ss:
        return None
    wps = sum(len(x.split()) for x in ss) / len(ss)
    return round(KO2025["const"]
                 + KO2025["coef_mid_vocab"] * mv[0]
                 + KO2025["coef_words_per_sent"] * wps, 2)


def ko2025_level(idx):
    """수준 환산표 적용 (논문 [표 8])."""
    if idx is None:
        return None
    for lo, name in KO2025["levels"]:
        if idx >= lo:
            return name
    return "어려움"


SPEC: Dict[str, tuple] = {
    # 고승연(2025) L2 이독성 — 값이 높을수록 쉬움
    "ko2025_A": ("higher", "published", "L2 이독성 지수 (미등재=중급이상)"),
    "ko2025_B": ("higher", "published", "L2 이독성 지수 (미등재 제외)"),
    "mid_vocab_A": ("lower", "published", "중급이상 어휘 비율 % (미등재=중급이상)"),
    "mid_vocab_B": ("lower", "published", "중급이상 어휘 비율 % (미등재 제외)"),
    "n_sentences":      ("info",   "프로젝트정의", "문장 수"),
    "chars_per_sent":   ("lower",  "프로젝트정의", "문장당 평균 글자 수"),
    "words_per_sent":   ("lower",  "프로젝트정의", "문장당 평균 어절 수"),
    "long_sent_ratio":  ("lower",  "프로젝트정의", "25어절 초과 문장 비율 %"),
    "sent_split_ratio": ("higher", "프로젝트정의", "출력 문장 수 / 원문 문장 수"),
    "length_change":    ("info",   "프로젝트정의", "글자 수 증감 %"),
    "hanja_ratio":      ("lower",  "프로젝트정의", "한자어 접미사 비율 (어절 100개당)"),
    "medterm_ratio":    ("info",   "프로젝트정의", "전문용어 출현 비율 (어절 100개당)"),
    "term_explain":     ("higher", "프로젝트정의", "용어(설명) 형태로 풀이된 전문용어 수"),
    "term_explain_rate":("higher", "프로젝트정의", "출현 전문용어 중 설명이 붙은 비율 %"),
    "copy_similarity":  ("info",   "프로젝트정의", "원문과의 문자 3-gram Jaccard. 1.0 = 미변환"),
    "difficult_word":   ("lower",  "논문공식",     "어려운 어휘 비율 — 어휘 목록 미확보"),
    "cho2016_grade":    ("lower",  "논문공식",     "조용구(2016) 학년 수준 — 공식 미확정"),
}

# ===========================================================================
# 어휘 자료 — 기존 스크립트와 동일한 정의를 쓴다
# ===========================================================================
MED_TERMS = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양", "감염",
    "출혈", "혈전", "색전증", "마취", "수혈", "봉합", "절제", "이식", "배액관",
    "합병증", "후유증", "부작용", "재발", "불유합", "성대마비", "저칼슘혈증",
]
HANJA_SUFFIX = re.compile(
    r"[가-힣]{1,4}(증|염|술|양성|성|적|화|법|부위|부|경|관|제|액|압|통|"
    r"기능|장애|손상|절제|봉합|주입|투여)(?=[\s,.)]|$)")

# "전문용어(쉬운 설명)" 형태 — 괄호 안에 한글 설명이 붙은 경우
TERM_EXPLAIN = re.compile(r"([가-힣]{2,10})\s*\(([^)]{4,40})\)")
# 괄호 안이 한글 위주여야 설명으로 본다 (영문 약어·수치는 제외)
_KO_IN_PAREN = re.compile(r"[가-힣]")


def _easy_words() -> Optional[set]:
    if not EASY_WORD_LIST_PATH or not os.path.exists(EASY_WORD_LIST_PATH):
        return None
    with open(EASY_WORD_LIST_PATH, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


# ===========================================================================

def simplicity(text: str, src: Optional[str] = None) -> Dict[str, Optional[float]]:
    """평이성 지표. src 를 주면 원문 대비 지표도 함께 낸다."""
    sents = Splitter.split(text)
    out: Dict[str, Optional[float]] = {k: None for k in SPEC}
    if not sents:
        return out

    words = [len(s.split()) for s in sents]
    n_words = sum(words) or 1
    joined = " ".join(sents)

    out["n_sentences"] = len(sents)
    out["chars_per_sent"] = round(sum(len(s) for s in sents) / len(sents), 2)
    out["words_per_sent"] = round(n_words / len(sents), 2)
    out["long_sent_ratio"] = round(
        100 * sum(1 for w in words if w > 25) / len(sents), 2)
    out["hanja_ratio"] = round(
        100 * len(HANJA_SUFFIX.findall(joined)) / n_words, 2)

    n_med = sum(joined.count(t) for t in MED_TERMS)
    out["medterm_ratio"] = round(100 * n_med / n_words, 2)

    # 용어(설명) 형태로 풀이된 전문용어
    explained = set()
    for m in TERM_EXPLAIN.finditer(joined):
        term, desc = m.group(1), m.group(2)
        if term in MED_TERMS or any(t in term for t in MED_TERMS):
            if len(_KO_IN_PAREN.findall(desc)) >= 3:      # 괄호 안이 한글 설명
                explained.add(term)
    out["term_explain"] = len(explained)
    present = {t for t in MED_TERMS if t in joined}
    out["term_explain_rate"] = (
        round(100 * len(explained) / len(present), 1) if present else None)

    if src is not None:
        src_sents = Splitter.split(src)
        out["length_change"] = round(
            100 * (len(text) - len(src)) / max(len(src), 1), 2)
        out["sent_split_ratio"] = (
            round(len(sents) / len(src_sents), 3) if src_sents else None)
        out["copy_similarity"] = round(jaccard3(src, text), 3)

    # ── 공식 미확보 지표 ────────────────────────────────────────────
    ew = _easy_words()
    if ew:
        toks = joined.split()
        easy = sum(1 for w in toks if w in ew)
        out["difficult_word"] = round(100 * (1 - easy / max(len(toks), 1)), 2)
        if CHO2016_ENABLED and CHO2016_CUBED is not None:
            ratio = easy / max(len(toks), 1)
            r = ratio ** 3 if CHO2016_CUBED else ratio
            out["cho2016_grade"] = round(
                4.874 + 0.591 * out["words_per_sent"] - 9.201 * r, 2)
    # 목록이 없으면 두 지표 모두 None 으로 남는다 (임의 구현하지 않음)

    # 고승연(2025) L2 이독성 — 미등재 어휘 정책 A/B 를 모두 계산한다
    for _pol in ("A", "B"):
        _mv = mid_vocab_ratio(text, _pol)
        out["mid_vocab_" + _pol] = _mv[0] if _mv else None
        out["ko2025_" + _pol] = ko2025_index(text, _pol)

    return out


def unresolved() -> List[str]:
    """아직 구현할 수 없는 지표와 사유."""
    msgs = []
    if _easy_words() is None:
        msgs.append("difficult_word — 쉬운 단어 목록 미확보")
    if not CHO2016_ENABLED:
        msgs.append("cho2016_grade — GL = 4.874 + 0.591A - 9.201B^3 확정. "
                    "B 의 5,000단어 목록 미확보로 보류")
    msgs.append("KReaD — 대교 특허, 어휘 등급 목록 28,332단어 비공개")
    msgs.append("이보미(2020) L2 난이도 공식 — PDF 미확보")
    msgs.append("Suh et al.(2013) 텍스트 복잡도 — PDF 미확보")
    if _load_vocab() is None:
        msgs.append("고승연(2025) L2 이독성 — 공식 확보 완료. "
                    "어휘 사전 없음(" + VOCAB_PATH + "). "
                    "build_vocab_grades.py 실행 필요")
    else:
        msgs.append("고승연(2025) L2 이독성 — 구현 완료. 단 미등재 어휘 "
                    "처리 방식이 논문에 없어 policy A/B 를 병기함")
    msgs.append("Natmal — 지표가 아니라 (주)낱말의 상용 프로그램. 대상 아님")
    return msgs


def arrow(name: str) -> str:
    d = SPEC.get(name, ("info",))[0]
    return {"lower": "↓", "higher": "↑", "info": "—"}.get(d, "—")


if __name__ == "__main__":
    demo_src = ("수술에 의한 뼈의 연속성 개선을 통한 통증 감소 및 기능의 개선. "
                "감염, 출혈, 혈전색전증, 불유합 등의 가능성이 있습니다.")
    demo_out = ("수술로 뼈를 다시 이어 붙여 통증을 줄이고 기능을 회복시킵니다. "
                "감염이 생길 수 있습니다. 출혈이 있을 수 있습니다. "
                "불유합(뼈가 제대로 붙지 않는 것)이 생길 수 있습니다.")
    print("=" * 78)
    print("지표 데모")
    print("=" * 78)
    r = simplicity(demo_out, demo_src)
    for k, v in r.items():
        d, cls, desc = SPEC[k]
        print(f"  {arrow(k)} {k:<20}{str(v):>10}   [{cls}] {desc}")
    print("\n미해결 항목")
    for m in unresolved():
        print(f"  · {m}")
