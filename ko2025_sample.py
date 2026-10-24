#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ko2025_sample.py — 이독성 지수 계산 샘플 공개 및 성분 분해
============================================================
교수님 요청 두 가지에 답하기 위한 조회 전용 스크립트다.
아무 파일도 수정하지 않는다.

  (1) 이독성 지수 계산의 샘플 데이터를 자세히 보여줄 것
  (2) 용어 설명 삽입으로 이독성이 악화된 원인을
      '어휘 항'과 '문장 길이 항'으로 분리해 정량화할 것

고승연(2025) 공식
    이독성 지수 = 61.994 − 0.261 × 중급이상어휘비율 − 1.045 × 평균어절수
    값이 높을수록 쉬움.  >=45 쉬움 / 35~44.99 보통 / <35 어려움

성분 분해 원리
    지수는 두 항의 선형 결합이므로 변화량을 정확히 나눌 수 있다.

        Δ지수 = (−0.261 × Δ어휘비율) + (−1.045 × Δ평균어절수)
                ╰─ 어휘 항 기여 ─╯      ╰─ 문장길이 항 기여 ─╯

    두 기여의 합이 Δ지수와 일치하는지 검산해 계산이 맞는지 확인한다.

사용법
    cd ~/이윤우 && source .venv/bin/activate
    python ko2025_sample.py                    # 전체 (문서별 + 성분분해 + 샘플)
    python ko2025_sample.py --doc doc1         # 특정 문서만 자세히
    python ko2025_sample.py --sample-sents 5   # 문장 샘플 개수
    python ko2025_sample.py --policy B         # policy B 로 계산
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from prompt_ladder import extract_sections, Splitter
except ImportError as e:
    sys.exit("[FATAL] prompt_ladder.py 필요: " + str(e))

try:
    from simplicity_metrics import (
        mid_vocab_ratio, ko2025_index, ko2025_level, KO2025, _vocab_grades,
    )
except ImportError:
    try:
        from simplicity_metrics import mid_vocab_ratio, ko2025_index, ko2025_level, KO2025
        _vocab_grades = None
    except ImportError as e:
        sys.exit("[FATAL] simplicity_metrics 에 Ko2025 함수가 없습니다: " + str(e))

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
EDIR_V105 = os.path.join(HOME, "outputs_glossary_v105", "E")
EDIR_V11 = os.path.join(HOME, "outputs_glossary_v11", "E")

C_MV = KO2025.get("coef_mid_vocab", -0.261)
C_WPS = KO2025.get("coef_words_per_sent", -1.045)
C_INT = KO2025.get("const", KO2025.get("intercept", 61.994))


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def wps(text: str) -> float:
    """평균 어절 수. Splitter 가 나눈 문장마다 공백 분리 토큰 수의 평균."""
    sents = Splitter.split(text)
    if not sents:
        return 0.0
    return st.fmean(len(s.split()) for s in sents)


def n_sents(text: str) -> int:
    return len(Splitter.split(text))


def _mv(text: str, policy: str) -> Tuple[float, int, int]:
    """mid_vocab_ratio 는 (비율, 전체 어절 수, 중급이상 어절 수) 를 돌려준다.
       구현에 따라 실수 하나만 돌려줄 수도 있어 둘 다 받는다."""
    r = mid_vocab_ratio(text, policy)
    if isinstance(r, (tuple, list)):
        ratio = float(r[0])
        total = int(r[1]) if len(r) > 1 else 0
        mid = int(r[2]) if len(r) > 2 else 0
        return ratio, total, mid
    return float(r), 0, 0


def analyse(text: str, policy: str = "A") -> Dict[str, float]:
    mv, n_tok, n_mid = _mv(text, policy)
    w = wps(text)
    idx = C_INT + C_MV * mv + C_WPS * w
    return {"mid_vocab": mv, "wps": w, "index": idx,
            "level": ko2025_level(idx), "n_sents": n_sents(text),
            "n_chars": len(text), "n_tokens": n_tok, "n_mid": n_mid}


def decompose(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """a → b 변화를 두 항으로 분해."""
    d_mv = b["mid_vocab"] - a["mid_vocab"]
    d_w = b["wps"] - a["wps"]
    c_mv = C_MV * d_mv
    c_w = C_WPS * d_w
    d_idx = b["index"] - a["index"]
    tot = abs(c_mv) + abs(c_w)
    return {
        "d_mid_vocab": d_mv, "d_wps": d_w,
        "contrib_vocab": c_mv, "contrib_length": c_w,
        "d_index": d_idx, "checksum": c_mv + c_w,
        "share_vocab": 100 * abs(c_mv) / tot if tot else 0.0,
        "share_length": 100 * abs(c_w) / tot if tot else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--edir-v105", default=EDIR_V105)
    ap.add_argument("--edir-v11", default=EDIR_V11)
    ap.add_argument("--policy", default="A", choices=["A", "B"])
    ap.add_argument("--doc", default=None, help="특정 문서만 자세히")
    ap.add_argument("--sample-sents", type=int, default=4)
    ap.add_argument("--out", default="ko2025_sample.json")
    a = ap.parse_args()

    print("=" * 100)
    print("이독성 지수 계산 샘플 및 성분 분해")
    print("=" * 100)
    print("  공식   이독성 지수 = %.3f %+.3f × 중급이상어휘비율 %+.3f × 평균어절수"
          % (C_INT, C_MV, C_WPS))
    print("  환산   45 이상 쉬움 / 35~44.99 보통 / 35 미만 어려움")
    print("  정책   policy " + a.policy
          + ("  (목록 미등재 어휘를 중급 이상으로 계산)" if a.policy == "A"
             else "  (목록 미등재 어휘를 분모에서 제외)"))
    if _vocab_grades:
        try:
            g = _vocab_grades()
            n_mid = sum(1 for v in g.values() if v >= 3)
            print("  어휘   표제어 %d개 중 중급이상(3~6급) %d개" % (len(g), n_mid))
        except Exception:
            pass

    # ── 데이터 적재 ────────────────────────────────────────────────────
    rows: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(a.edir_v11, "*.json"))):
        with open(p, encoding="utf-8") as f:
            d11 = json.load(f)
        doc = d11["doc"]
        K = load_K(doc, a.docdir)
        if not K:
            continue
        p105 = os.path.join(a.edir_v105, doc + ".json")
        E105 = None
        if os.path.exists(p105):
            with open(p105, encoding="utf-8") as f:
                E105 = json.load(f)["E_output"]
        rows.append({
            "doc": doc,
            "K": K, "A": d11["A_output"],
            "E105": E105, "E11": d11["E_output"],
            "n_ins_11": d11["n_insertions"],
            "insertions": d11["insertions"],
        })
    if not rows:
        sys.exit("[FATAL] outputs_glossary_v11/E 에 결과가 없습니다.")

    # ── 1. 문서별 계산 과정 ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("1. 문서별 계산 — 두 변수와 지수")
    print("=" * 100)
    print("  " + "문서".ljust(7) + "구분".ljust(9)
          + "문장수".rjust(7) + "어절/문장".rjust(10)
          + "중급이상%".rjust(11) + "중급/전체".rjust(12)
          + "지수".rjust(9) + "수준".rjust(8))
    stats = {"K": [], "A": [], "E105": [], "E11": []}
    per_doc = []
    for r in rows:
        rec = {"doc": r["doc"], "n_insertions": r["n_ins_11"]}
        for key, label in [("K", "원문"), ("A", "A 평이화"),
                           ("E105", "E 7개"), ("E11", "E 11개")]:
            if r[key] is None:
                continue
            m = analyse(r[key], a.policy)
            stats[key].append(m)
            rec[key] = m
            frac = ("%d/%d" % (m["n_mid"], m["n_tokens"])) if m["n_tokens"] else "—"
            print("  " + r["doc"].ljust(7) + label.ljust(9)
                  + str(m["n_sents"]).rjust(7)
                  + ("%10.2f" % m["wps"])
                  + ("%11.2f" % m["mid_vocab"])
                  + frac.rjust(12)
                  + ("%9.2f" % m["index"])
                  + m["level"].rjust(8))
        per_doc.append(rec)
        print()

    def avg(key: str) -> Dict[str, float]:
        v = stats[key]
        return {k: st.fmean(x[k] for x in v)
                for k in ("mid_vocab", "wps", "index")} if v else {}

    mK, mA, m105, m11 = avg("K"), avg("A"), avg("E105"), avg("E11")

    print("=" * 100)
    print("2. 13문서 평균")
    print("=" * 100)
    print("  " + "구분".ljust(22) + "어절/문장".rjust(11)
          + "중급이상%".rjust(12) + "지수".rjust(10) + "수준".rjust(9))
    for m, label in [(mK, "원문 그대로 (Identity)"), (mA, "A — HARI 평이화"),
                     (m105, "E — 용어 설명 7개"), (m11, "E — 용어 설명 11개")]:
        if not m:
            continue
        print("  " + label.ljust(22)
              + ("%11.2f" % m["wps"]) + ("%12.2f" % m["mid_vocab"])
              + ("%10.2f" % m["index"]) + ko2025_level(m["index"]).rjust(9))

    # ── 3. 성분 분해 ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("3. 성분 분해 — 악화의 원인이 어휘인가 문장 길이인가")
    print("=" * 100)
    print("  Δ지수 = (%.3f × Δ어휘비율) + (%.3f × Δ평균어절수)" % (C_MV, C_WPS))
    print("          ╰─ 어휘 항 ─╯        ╰─ 문장길이 항 ─╯\n")

    pairs = [("A", "E105", "A → E 7개"), ("A", "E11", "A → E 11개"),
             ("E105", "E11", "E 7개 → E 11개"), ("K", "A", "원문 → A (참고)")]
    dec_out = {}
    print("  " + "비교".ljust(18) + "Δ어휘".rjust(9) + "기여".rjust(9)
          + "몫".rjust(8) + "  │" + "Δ어절".rjust(9) + "기여".rjust(9)
          + "몫".rjust(8) + "  │" + "Δ지수".rjust(9) + "검산".rjust(9))
    for ka, kb, label in pairs:
        A_, B_ = avg(ka), avg(kb)
        if not A_ or not B_:
            continue
        d = decompose(A_, B_)
        dec_out[label] = d
        print("  " + label.ljust(18)
              + ("%+9.2f" % d["d_mid_vocab"]) + ("%+9.3f" % d["contrib_vocab"])
              + ("%7.1f%%" % d["share_vocab"]) + "  │"
              + ("%+9.2f" % d["d_wps"]) + ("%+9.3f" % d["contrib_length"])
              + ("%7.1f%%" % d["share_length"]) + "  │"
              + ("%+9.2f" % d["d_index"]) + ("%+9.3f" % d["checksum"]))

    key = "A → E 11개"
    if key in dec_out:
        d = dec_out[key]
        print("""
  읽는 법 — %s
    어휘 항이 지수를 %+.3f 만큼 움직였고 문장길이 항이 %+.3f 만큼 움직였습니다.
    악화분에서 문장길이 항의 몫이 %.1f%%, 어휘 항의 몫이 %.1f%% 입니다.
    두 기여의 합 %+.3f 이 Δ지수 %+.3f 과 일치하므로 계산이 맞습니다.""" % (
            key, d["contrib_vocab"], d["contrib_length"],
            d["share_length"], d["share_vocab"], d["checksum"], d["d_index"]))

    # ── 4. 문장 단위 샘플 ─────────────────────────────────────────────
    target = a.doc or rows[0]["doc"]
    r = next((x for x in rows if x["doc"] == target), rows[0])
    print("\n" + "=" * 100)
    print("4. 문장 단위 계산 샘플 — " + r["doc"])
    print("=" * 100)
    print("  어절 수는 공백으로 나눈 토큰 개수입니다.")
    print("  Splitter 는 10자 미만 조각과 번호만 있는 줄을 문장에서 제외합니다.\n")

    for key, label in [("A", "A — HARI 평이화"), ("E11", "E — 용어 설명 11개")]:
        sents = Splitter.split(r[key])
        print("  [" + label + "]  총 " + str(len(sents)) + "문장")
        for i, s in enumerate(sents[:a.sample_sents], 1):
            toks = s.split()
            print("    %d) %d어절  %s" % (i, len(toks), s[:76]))
            print("       어절: " + " / ".join(toks[:9])
                  + (" ..." if len(toks) > 9 else ""))
        m = analyse(r[key], a.policy)
        extra = (" (%d/%d어절)" % (m["n_mid"], m["n_tokens"])) if m["n_tokens"] else ""
        print("    → 평균 %.2f어절, 중급이상 %.2f%%%s, 지수 %.2f (%s)"
              % (m["wps"], m["mid_vocab"], extra, m["index"], m["level"]))
        print("      검산  %.3f %+.3f×%.2f %+.3f×%.2f = %.2f\n"
              % (C_INT, C_MV, m["mid_vocab"], C_WPS, m["wps"], m["index"]))

    # ── 5. 삽입된 설명문의 어절 수 ────────────────────────────────────
    print("=" * 100)
    print("5. 삽입된 설명문이 문장을 얼마나 늘리는가 — " + r["doc"])
    print("=" * 100)
    print("  " + "용어".ljust(12) + "설명 어절".rjust(10) + "  설명문")
    tot_tok = 0
    for ins in r["insertions"]:
        t = ins["explanation"].split()
        tot_tok += len(t)
        print("  " + ins["term"].ljust(12) + str(len(t)).rjust(10)
              + "  " + ins["explanation"][:58])
    if r["insertions"]:
        print("\n  삽입 %d건, 추가된 어절 합계 %d개, 건당 평균 %.1f어절"
              % (len(r["insertions"]), tot_tok, tot_tok / len(r["insertions"])))
        mA_ = analyse(r["A"], a.policy)
        print("  이 문서의 문장 수가 %d개이므로 단순 계산으로 문장당 %+.2f어절"
              % (mA_["n_sents"], tot_tok / max(mA_["n_sents"], 1)))
        mE_ = analyse(r["E11"], a.policy)
        print("  실측 변화는 %.2f → %.2f (%+.2f) 입니다."
              % (mA_["wps"], mE_["wps"], mE_["wps"] - mA_["wps"]))

    # ── 저장 ───────────────────────────────────────────────────────────
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "formula": {"intercept": C_INT, "coef_mid_vocab": C_MV,
                        "coef_words_per_sent": C_WPS},
            "policy": a.policy,
            "levels": {">=45": "쉬움", "35~44.99": "보통", "<35": "어려움"},
            "averages": {"K": mK, "A": mA, "E_v105": m105, "E_v11": m11},
            "decomposition": dec_out,
            "per_doc": per_doc,
        }, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] " + a.out)

    print("""
결과 보는 법
  · 3절의 '몫' 이 악화 원인의 비중입니다. 문장길이 항의 몫이 크면
    어휘 때문이 아니라 문장이 길어져서 지수가 내려간 것입니다.
  · '검산' 열이 Δ지수와 같아야 정상입니다. 지수가 두 항의 선형
    결합이므로 정확히 일치해야 합니다.
  · 5절은 설명문 자체가 몇 어절인지 보여줍니다. 설명이 길면
    문장길이 항이 그만큼 악화됩니다.""")
    print("=" * 100)


if __name__ == "__main__":
    main()
