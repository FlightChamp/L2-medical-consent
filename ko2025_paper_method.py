#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ko2025_paper_method.py — 고승연(2025) 논문 방식으로 이독성 재계산 (민감도 분석)
==============================================================================
기존 지표(simplicity_metrics.ko2025_index)는 바꾸지 않는다. 같은 출력에 대해
논문 원문의 계산 방식에 더 가까운 값을 나란히 계산해, 결론이 바뀌는지 본다.

논문(4.2.1절, 표 3)에서 확인한 규칙과 이 스크립트의 대응
    어휘 단위      '중복 어휘를 제외한 형태소'        → Kiwi 형태소 분석, 서로 다른 어휘(type)만 셈
    목록 밖 어휘    '기타'로 분류해 어려운 어휘에 포함  → policy A (논문 규칙). B 는 참고용
    문장 길이      총 어절 수 ÷ 문장 수              → 기존과 같은 정의
    문장 경계      논문에 명시 없음                  → 기존 Splitter 그대로 (우리 규칙)

논문에 없는 것 → 이 스크립트가 정한 가정 (보고 시 명시)
    · 어휘로 세는 형태소 = 내용 형태소(명사·대명사·수사·동사·형용사·보조용언·관형사·부사·어근·감탄사).
      조사·어미·접사·기호·숫자·외국어는 '어휘 목록'의 대상이 아니라고 보고 제외
    · 동사·형용사는 어간 + '다' 로 표제어를 만든다 (있 → 있다)

함께 계산하는 것
    · 원문(줄바꿈 보존): 원문 추출 시 줄을 공백으로 잇지 않고 줄바꿈을 살려
      문장당 어절 수를 다시 잰다 (원문만 서식 항목이 한 문장으로 뭉치는 비대칭 확인)
    · 논문 [표 6] 회귀계수(62.147 − 0.262 − 1.004)로 계산한 값 (공식과 계수가 다르므로)

사용법
    cd ~/이윤우 && source .venv/bin/activate && pip install -q kiwipiepy
    python ko2025_paper_method.py
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics as st
import sys

HOME = os.environ.get("L2_HOME", os.path.expanduser("~/이윤우"))
for _p in (HOME, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt_ladder import extract_sections, Splitter, SEC, DROP   # noqa: E402
import simplicity_metrics as SM                                    # noqa: E402

FORMULA = (61.994, -0.261, -1.045)      # 논문 4.2.4절 공식 (기존과 동일)
TABLE6 = (62.147, -0.262, -1.004)       # 논문 [표 6] 비표준화 회귀계수
CONTENT = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX",
           "MM", "MAG", "MAJ", "XR", "IC"}
PRED = {"VV", "VA", "VX"}


def level(x):
    return SM.ko2025_level(x)


# ===========================================================================
# 텍스트 불러오기
# ===========================================================================

def load_K(doc):
    p = os.path.join(HOME, "docs", doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_K_lines(doc):
    """extract_sections 와 같은 필터를 쓰되, 줄을 공백이 아니라 줄바꿈으로 잇는다."""
    p = os.path.join(HOME, "docs", doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines = [re.sub(r"\s+", " ", l).strip() for l in raw.split("\n")]
    secs, cur, buf = [], None, []
    for l in lines:
        if not l:
            continue
        m = SEC.match(l)
        if m:
            if cur and buf:
                secs.append("\n".join(buf))
            cur, buf = m.group(0), []
            continue
        if cur is None or DROP.search(l):
            continue
        if len(re.findall(r"[가-힣]", l)) < 5:
            continue
        buf.append(l)
    if cur and buf:
        secs.append("\n".join(buf))
    return "\n".join(secs) if secs else None


def load_json(path, key):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get(key)


# ===========================================================================
# 논문 방식 어휘 비율
# ===========================================================================

class PaperVocab:
    def __init__(self):
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            sys.exit("[FATAL] kiwipiepy 가 없습니다:  pip install -q kiwipiepy")
        self.kiwi = Kiwi()
        self.vocab = SM._load_vocab()
        if not self.vocab:
            sys.exit("[FATAL] vocab_grades.json 을 읽지 못했습니다.")

    def lemmas(self, text):
        out = []
        for t in self.kiwi.tokenize(text):
            tag = t.tag.split("-")[0]
            if tag not in CONTENT:
                continue
            form = t.form
            if not re.search(r"[가-힣]", form):
                continue
            lem = form + "다" if tag in PRED and not form.endswith("다") else form
            out.append((lem, tag))
        return out

    def grade(self, lem):
        return self.vocab.get(lem)

    def ratio(self, text):
        types = sorted({l for l, _ in self.lemmas(text)})
        g = {l: self.grade(l) for l in types}
        low = sum(1 for v in g.values() if v is not None and v < 3)
        mid = sum(1 for v in g.values() if v is not None and v >= 3)
        unl = sum(1 for v in g.values() if v is None)
        n = low + mid + unl
        return {"n_types": n, "low": low, "mid": mid, "unlisted": unl,
                "ratio_A": 100 * (mid + unl) / max(n, 1),
                "ratio_B": 100 * mid / max(low + mid, 1),
                "grades": g}


def wps(text):
    ss = Splitter.split(text)
    return (sum(len(s.split()) for s in ss) / len(ss)) if ss else None, len(ss)


def idx(coef, ratio, w):
    return coef[0] + coef[1] * ratio + coef[2] * w


# ===========================================================================

def main():
    pv = PaperVocab()
    docs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(HOME, "docs", "*.txt"))
                  if not os.path.basename(p).startswith("syn"))
    routes = {
        "원문": lambda d: load_K(d),
        "원문(줄바꿈보존)": lambda d: load_K_lines(d),
        "HARI": lambda d: load_json(os.path.join(HOME, "outputs_prompt", d + "__P0.json"), "out"),
        "괄호형": lambda d: load_json(os.path.join(HOME, "outputs_glossary_v11", "E", d + ".json"), "E_output"),
        "문장분리형": lambda d: load_json(os.path.join(HOME, "outputs_glossary_v11_sent", "E", d + ".json"), "E_output"),
        "여기서": lambda d: load_json(os.path.join(HOME, "outputs_glossary_v11_sent_yeogi", "E", d + ".json"), "E_output"),
    }

    rows = []
    unl_counter = {r: collections.Counter() for r in routes}
    for d in docs:
        for r, fn in routes.items():
            t = fn(d)
            if not t:
                continue
            w, ns = wps(t)
            pr = pv.ratio(t)
            cur = SM.ko2025_index(t, "A")
            for l, g in pr["grades"].items():
                if g is None:
                    unl_counter[r][l] += 1
            rows.append({
                "doc": d, "route": r, "n_sents": ns, "wps": w,
                "ours_ko2025_A": cur,
                "paper_n_types": pr["n_types"], "paper_unlisted": pr["unlisted"],
                "paper_ratio_A": pr["ratio_A"], "paper_ratio_B": pr["ratio_B"],
                "paper_ko2025_A": idx(FORMULA, pr["ratio_A"], w),
                "paper_ko2025_B": idx(FORMULA, pr["ratio_B"], w),
                "paper_ko2025_A_table6": idx(TABLE6, pr["ratio_A"], w),
            })
        print(".", end="", flush=True)
    print()

    def M(r, k):
        v = [x[k] for x in rows if x["route"] == r and x.get(k) is not None]
        return st.fmean(v) if v else None
    def f(v, w=9, d=2):
        return "—".rjust(w) if v is None else f"{v:{w}.{d}f}"

    print("=" * 108)
    print("고승연(2025) 논문 방식 재계산 — 13문서 평균 (값이 높을수록 쉬움)")
    print("=" * 108)
    print("  " + "경로".ljust(16) + "어절/문장".rjust(9) + "기존지수".rjust(10)
          + "어휘type".rjust(9) + "기타%".rjust(8) + "논문비율A".rjust(10)
          + "논문지수A".rjust(10) + "수준".rjust(6) + "논문지수B".rjust(10)
          + "표6계수A".rjust(10))
    for r in routes:
        if M(r, "wps") is None:
            continue
        n = M(r, "paper_n_types") or 1
        print("  " + r.ljust(16) + f(M(r, "wps")) + f(M(r, "ours_ko2025_A"), 10)
              + f(M(r, "paper_n_types"), 9, 0)
              + f(100 * (M(r, "paper_unlisted") or 0) / n, 8, 1)
              + f(M(r, "paper_ratio_A"), 10) + f(M(r, "paper_ko2025_A"), 10)
              + str(level(M(r, "paper_ko2025_A"))).rjust(6)
              + f(M(r, "paper_ko2025_B"), 10) + f(M(r, "paper_ko2025_A_table6"), 10))

    print("\n  비교 (논문 방식 지수 A 기준)")
    pairs = [("원문", "HARI"), ("원문(줄바꿈보존)", "HARI"), ("HARI", "괄호형"),
             ("HARI", "문장분리형"), ("HARI", "여기서"), ("괄호형", "문장분리형")]
    for a_, b_ in pairs:
        if M(a_, "paper_ko2025_A") is None or M(b_, "paper_ko2025_A") is None:
            continue
        d_new = M(b_, "paper_ko2025_A") - M(a_, "paper_ko2025_A")
        d_old = M(b_, "ours_ko2025_A") - M(a_, "ours_ko2025_A")
        better = sum(1 for d in docs
                     if (lambda xa, xb: xa and xb and xb["paper_ko2025_A"] > xa["paper_ko2025_A"])(
                         next((x for x in rows if x["doc"] == d and x["route"] == a_), None),
                         next((x for x in rows if x["doc"] == d and x["route"] == b_), None)))
        print(f"    {a_:>14} → {b_:<10}  논문방식 {d_new:+6.2f}  (기존 {d_old:+6.2f})   "
              f"뒤쪽이 더 쉬운 문서 {better}/{len(docs)}")

    print("\n  HARI 의 '기타'(목록 밖) 어휘 상위 20 — 논문 규칙상 어려운 어휘로 셈")
    print("    " + ", ".join(f"{l}({n})" for l, n in unl_counter["HARI"].most_common(20)))

    # ── doc1 예시: HARI 첫 2문장의 형태소 판정 ─────────────────────────
    t = routes["HARI"]("doc1")
    if t:
        print("\n  예시 — doc1 HARI 첫 2문장의 어휘(내용 형태소) 판정")
        for s in Splitter.split(t)[:2]:
            print("    문장: " + s)
            items = []
            for l, tag in pv.lemmas(s):
                g = pv.grade(l)
                items.append(f"{l}/{tag}:{'기타' if g is None else str(g) + '급'}")
            print("      " + ", ".join(items))

    with open(os.path.join(HOME, "ko2025_paper_method.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "assumptions": {
                "vocab_unit": "unique content morphemes (Kiwi); POS " + ",".join(sorted(CONTENT)),
                "predicate_lemma": "stem + 다",
                "unlisted": "counted as difficult (paper rule) = policy A; B for reference",
                "sentence_boundary": "existing Splitter (not specified in paper)",
                "formula": FORMULA, "table6": TABLE6,
            },
            "summary": {r: {k: M(r, k) for k in ("wps", "ours_ko2025_A", "paper_n_types",
                                                "paper_ratio_A", "paper_ko2025_A",
                                                "paper_ko2025_B", "paper_ko2025_A_table6")}
                        for r in routes},
            "rows": rows,
        }, fh, ensure_ascii=False, indent=2)
    print("\n[SAVE] ko2025_paper_method.json (수치만, 본문 없음)")


if __name__ == "__main__":
    main()
