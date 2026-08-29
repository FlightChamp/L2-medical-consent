#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readability.py — ③ 변환 단계의 평이화 정도 측정
================================================
왜 필요한가:
    ④ 검증은 "변환문이 원문에 근거하는가"만 본다.
    "쉬워졌는가"는 아무도 재지 않는다.
    극단적으로, 원문을 그대로 복사한 변환기가 ④에서 만점을 받는다.
    이 프로젝트의 목적이 L2 화자의 이해 지원인 이상, 이 구멍은 치명적이다.

설계 방침:
    한국어에는 Flesch 같은 확립된 가독성 공식이 없다.
    따라서 '이 공식을 썼다'고 말할 외부 권위가 없다. 대신 두 가지로 정당화한다.

    (1) 개별 지표를 각각 한 문장으로 설명 가능하게 만든다.
        블랙박스 점수 하나를 내놓지 않는다.
    (2) 사람 판정과의 상관으로 검증한다.
        어떤 지표가 실제 사람 판단을 예측하는지 데이터로 고른다.

    아울러 원문 복사 탐지를 넣는다. 평이화 지표의 존재 이유가 이것이다.

지표 (모두 낮을수록 읽기 쉬움):
    sent_len    평균 어절 수                문장이 길수록 어렵다
    long_ratio  25어절 초과 문장 비율       긴 문장이 섞여 있으면 그 문장이 병목
    clause      문장당 연결어미 수          복문일수록 어렵다
    nominal     명사형 어미 빈도            '~함/됨/임'은 문어체 압축 표현
    passive     피동 표현 빈도              행위 주체가 흐려진다
    term        의료 전문용어 비율          L2 화자에게 가장 큰 장벽
    hanja       한자어 접미사 비율          한자어는 고유어보다 어렵다
    paren       괄호·기호 밀도              서식 잔재가 읽기를 방해한다
    formal      문어체 표현 빈도            '~에의', '~함에 있어' 등

    복사 탐지: 원문-변환문 문자 3-gram Jaccard. 0.90 이상이면 사실상 복사.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python readability.py --inspect                 # 변환문 파일 구조 확인
    python readability.py
    python readability.py --pairs mypairs.tsv --human human_eval_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ===========================================================================
# 전처리
# ===========================================================================

class Norm:
    CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, t) -> str:
        t = unicodedata.normalize("NFKC", str(t))
        t = cls.CTRL.sub(" ", t)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\n{3,}", "\n\n", cls.WS.sub(" ", t)).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls.BOUND.split(t):
            if not p:
                continue
            s = p.strip()
            if s and not cls.NUM_PREFIX.match(s):
                out.append(s)
        return out


# ===========================================================================
# 지표
# ===========================================================================

MED_TERMS = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양", "감염",
    "출혈", "혈전", "색전증", "마취", "수혈", "봉합", "절제", "이식", "배액관",
    "합병증", "후유증", "부작용", "재발", "불유합", "성대마비", "저칼슘혈증",
    "기흉", "탈장", "치핵", "복막염", "패혈증", "섬망", "욕창", "구축",
]

# 한자어 접미사. 고유어에는 거의 나타나지 않아 한자어 추정 지표로 쓴다.
HANJA_SUFFIX = re.compile(
    r"[가-힣]{1,4}(증|염|술|종양|양성|성|적|화|법|부위|부|경|관|제|액|압|통|"
    r"기능|장애|손상|절제|봉합|주입|투여)(?=[\s,.)]|$)")

CONNECTIVE = re.compile(
    r"(며|면서|지만|으나|나|고|거나|든지|아서|어서|니까|므로|므로써|"
    r"도록|더라도|는데|은데|ㄴ데|려면|자면|다면|경우|때문에|위하여|위해)")

NOMINAL = re.compile(r"[가-힣]+(함|됨|임|음|기)(?=[\s,.)]|$)")

PASSIVE = re.compile(r"[가-힣]+(되[어었습니다는]|받[았습니다는]|당[했합]|"
                     r"지[어었습니다는]|혀[졌집])")

FORMAL = re.compile(r"(에의|로서의|으로서의|함에|음에|하는 바|에 있어|"
                    r"에 한하여|에 준하여|하여야|되어야|바랍니다)")

PAREN = re.compile(r"[()\[\]{}（）〔〕:：·／/※▶►▲■□○●◦-]")


@dataclass
class Features:
    n_sent: int = 0
    n_word: int = 0
    sent_len: float = 0.0
    long_ratio: float = 0.0
    clause: float = 0.0
    nominal: float = 0.0
    passive: float = 0.0
    term: float = 0.0
    hanja: float = 0.0
    paren: float = 0.0
    formal: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in
                ("sent_len", "long_ratio", "clause", "nominal", "passive",
                 "term", "hanja", "paren", "formal")}


METRIC_LABELS = {
    "sent_len": "평균 어절 수",
    "long_ratio": "긴 문장 비율",
    "clause": "문장당 연결어미",
    "nominal": "명사형 어미",
    "passive": "피동 표현",
    "term": "전문용어 비율",
    "hanja": "한자어 접미사",
    "paren": "괄호·기호 밀도",
    "formal": "문어체 표현",
}


class Analyzer:
    LONG = 25   # 어절

    @classmethod
    def text(cls, t: str) -> Features:
        sents = Splitter.split(Norm.clean(t))
        return cls.from_sentences(sents)

    @classmethod
    def from_sentences(cls, sents: Sequence[str]) -> Features:
        f = Features()
        if not sents:
            return f
        words_per = [len(s.split()) for s in sents]
        total_words = sum(words_per) or 1
        f.n_sent = len(sents)
        f.n_word = total_words
        f.sent_len = total_words / len(sents)
        f.long_ratio = sum(1 for w in words_per if w > cls.LONG) / len(sents)

        joined = " ".join(sents)
        f.clause = len(CONNECTIVE.findall(joined)) / len(sents)
        f.nominal = 100 * len(NOMINAL.findall(joined)) / total_words
        f.passive = 100 * len(PASSIVE.findall(joined)) / total_words
        f.hanja = 100 * len(HANJA_SUFFIX.findall(joined)) / total_words
        f.formal = 100 * len(FORMAL.findall(joined)) / total_words
        f.paren = 100 * len(PAREN.findall(joined)) / max(len(joined), 1)
        f.term = 100 * sum(joined.count(t) for t in MED_TERMS) / total_words
        return f


def trigrams(t: str) -> set:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + 3] for i in range(len(s) - 2)} or {s}


def jaccard(a: str, b: str) -> float:
    ga, gb = trigrams(a), trigrams(b)
    return len(ga & gb) / max(len(ga | gb), 1)


# ===========================================================================
# 통계
# ===========================================================================

def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


# ===========================================================================
# 변환문 로딩
# ===========================================================================

class PairLoader:
    """(문서명, 원문, 변환문) 목록을 만든다."""

    SRC_HINTS = ("원문", "source", "src", "original", "input", "before")
    OUT_HINTS = ("변환", "output", "out", "simplified", "result", "after",
                 "converted", "hari", "pred")

    @staticmethod
    def from_tsv(path: str) -> List[Tuple[str, str, str]]:
        out = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f, delimiter="\t")
            cols = rd.fieldnames or []
            src = next((c for c in cols if any(h in c.lower()
                                               for h in PairLoader.SRC_HINTS)), None)
            dst = next((c for c in cols if any(h in c.lower()
                                               for h in PairLoader.OUT_HINTS)), None)
            if not src or not dst:
                sys.exit(f"[FATAL] {path} 에서 원문/변환문 열을 못 찾음: {cols}")
            for i, r in enumerate(rd):
                a, b = Norm.clean(r.get(src, "")), Norm.clean(r.get(dst, ""))
                if len(a) > 30 and len(b) > 30:
                    out.append((r.get("doc") or f"row{i}", a, b))
        return out

    @classmethod
    def inspect(cls, patterns: Sequence[str]):
        print("[INSPECT] 변환문이 들어있을 만한 JSON 구조를 훑습니다\n")
        for pat in patterns:
            for path in sorted(glob.glob(pat)):
                try:
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                except Exception:
                    continue
                print(f"  {path}")
                cls._describe(d, indent="    ", depth=0)
                print()

    @classmethod
    def _describe(cls, node, indent: str, depth: int):
        if depth > 3:
            return
        if isinstance(node, dict):
            for k, v in list(node.items())[:12]:
                if isinstance(v, str):
                    kind = "긴 한글" if len(v) > 80 else "문자열"
                    print(f"{indent}{k}: {kind} ({len(v)}자) {v[:50]}...")
                elif isinstance(v, (list, dict)):
                    print(f"{indent}{k}: {type(v).__name__} "
                          f"({len(v)}개)")
                    cls._describe(v, indent + "  ", depth + 1)
                else:
                    print(f"{indent}{k}: {type(v).__name__} = {v}")
        elif isinstance(node, list) and node:
            cls._describe(node[0], indent + "  ", depth + 1)

    @classmethod
    def auto(cls, patterns: Sequence[str]) -> List[Tuple[str, str, str]]:
        """JSON 안에서 원문/변환문으로 보이는 키 쌍을 찾아낸다."""
        pairs: List[Tuple[str, str, str]] = []
        for pat in patterns:
            for path in sorted(glob.glob(pat)):
                try:
                    with open(path, encoding="utf-8") as f:
                        d = json.load(f)
                except Exception:
                    continue
                recs = cls._records(d)
                for i, r in enumerate(recs):
                    if not isinstance(r, dict):
                        continue
                    s = cls._pick(r, cls.SRC_HINTS)
                    o = cls._pick(r, cls.OUT_HINTS)
                    if s and o and s != o:
                        name = str(r.get("doc") or r.get("name")
                                   or r.get("file") or f"{os.path.basename(path)}#{i}")
                        pairs.append((name, Norm.clean(s), Norm.clean(o)))
                if pairs:
                    print(f"  {path} 에서 {len(pairs)}쌍 확보")
                    return pairs
        return pairs

    @staticmethod
    def _records(d):
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return [d]
        return []

    @staticmethod
    def _pick(rec: dict, hints: Sequence[str]) -> Optional[str]:
        best = None
        for k, v in rec.items():
            if not isinstance(v, str) or len(v) < 30:
                continue
            if any(h in k.lower() for h in hints):
                if best is None or len(v) > len(best):
                    best = v
        return best


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--pairs", default=None, help="원문/변환문 TSV")
    ap.add_argument("--json", nargs="+",
                    default=["realdoc_report.json", "*report*.json"],
                    help="변환문이 들어있을 JSON 후보")
    ap.add_argument("--human", default="human_eval_sheet.csv")
    ap.add_argument("--copy-threshold", type=float, default=0.90)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default="readability")
    a = ap.parse_args()

    if a.inspect:
        PairLoader.inspect(a.json)
        return

    # -- 1. 원문만으로 기준선 --------------------------------------------
    print("=" * 76)
    print("1. 원문 기준선 — 동의서가 얼마나 어려운가")
    print("=" * 76)
    doc_feats = {}
    for p in sorted(glob.glob(os.path.join(a.docs, "*.txt"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith("syn"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            doc_feats[name] = Analyzer.text(f.read())
    if not doc_feats:
        sys.exit(f"[FATAL] {a.docs}/*.txt 없음")

    keys = list(METRIC_LABELS)
    print(f"  {'문서':<8}{'문장':>5}" + "".join(f"{METRIC_LABELS[k][:6]:>10}"
                                              for k in keys))
    for name, f in doc_feats.items():
        d = f.as_dict()
        print(f"  {name:<8}{f.n_sent:>5}" + "".join(f"{d[k]:>10.2f}" for k in keys))
    avg = {k: sum(f.as_dict()[k] for f in doc_feats.values()) / len(doc_feats)
           for k in keys}
    print(f"  {'평균':<8}{'':>5}" + "".join(f"{avg[k]:>10.2f}" for k in keys))

    # -- 2. 변환문과 비교 --------------------------------------------------
    print("\n" + "=" * 76)
    print("2. 원문 → 변환문 변화")
    print("=" * 76)
    if a.pairs:
        pairs = PairLoader.from_tsv(a.pairs)
    else:
        pairs = PairLoader.auto(a.json)
    if not pairs:
        print("  변환문 쌍을 찾지 못했습니다.")
        print("  --inspect 로 JSON 구조를 확인하거나 --pairs 로 TSV를 지정하세요.")
        print("  (열 이름에 '원문'/'변환' 또는 source/output 이 들어가면 자동 인식)")
    else:
        print(f"  쌍 {len(pairs)}건\n")
        rows = []
        for name, src, out in pairs:
            fs, fo = Analyzer.text(src), Analyzer.text(out)
            sim = jaccard(src, out)
            rows.append((name, fs, fo, sim))

        print(f"  {'문서':<14}{'유사도':>8}" +
              "".join(f"{METRIC_LABELS[k][:6]:>11}" for k in keys))
        for name, fs, fo, sim in rows:
            ds, do = fs.as_dict(), fo.as_dict()
            cells = []
            for k in keys:
                if ds[k] == 0:
                    cells.append(f"{'—':>11}")
                else:
                    cells.append(f"{100*(do[k]-ds[k])/ds[k]:>+10.0f}%")
            flag = " ← 복사 의심" if sim >= a.copy_threshold else ""
            print(f"  {name[:13]:<14}{sim:>8.2f}" + "".join(cells) + flag)

        print("\n  (숫자는 변화율. 음수가 평이화 방향)")
        n_copy = sum(1 for *_, s in rows if s >= a.copy_threshold)
        if n_copy:
            print(f"\n  [!] 원문 유사도 {a.copy_threshold} 이상 {n_copy}건 — "
                  "사실상 복사입니다.")
            print("      ④ 근거율은 만점이 나오지만 평이화는 일어나지 않았습니다.")

        # 요약 지수
        agg = {}
        for k in keys:
            base = [f.as_dict()[k] for _, f, _, _ in rows if f.as_dict()[k] > 0]
            new = [g.as_dict()[k] for _, f, g, _ in rows if f.as_dict()[k] > 0]
            if base:
                agg[k] = 100 * (sum(new) / len(new) - sum(base) / len(base)) \
                    / (sum(base) / len(base))
        print("\n  전체 평균 변화율")
        for k, v in sorted(agg.items(), key=lambda x: x[1]):
            mark = "평이화" if v < -5 else ("역행" if v > 5 else "변화 없음")
            print(f"    {METRIC_LABELS[k]:<14}{v:>+8.1f}%   {mark}")
        core = [agg.get("sent_len", 0), agg.get("term", 0)]
        print(f"\n  핵심 2지표 평균(문장 길이·전문용어) {sum(core)/2:>+.1f}%")
        print("  → 이 두 가지는 L2 가독성 연구에서 가장 일관되게 지지되는 요인이라"
              " 대표 지표로 삼는다.")

    # -- 3. 사람 판정과의 상관 ---------------------------------------------
    print("\n" + "=" * 76)
    print("3. 사람 판정과의 상관 — 어떤 지표가 사람 판단을 예측하는가")
    print("=" * 76)
    if not os.path.exists(a.human):
        print(f"  {a.human} 없음 — 건너뜁니다.")
    else:
        with open(a.human, encoding="utf-8-sig", newline="") as f:
            recs = list(csv.DictReader(f))
        if not recs:
            print("  빈 파일입니다.")
        else:
            cols = list(recs[0])
            tcol = None
            for c in cols:
                vals = [str(r.get(c, "")) for r in recs]
                if sum(1 for v in vals if len(v) > 20) > len(recs) * 0.5:
                    tcol = c
                    break
            rcol = None
            for c in cols:
                vals = [str(r.get(c, "")).strip() for r in recs]
                nums = [v for v in vals if v in ("1", "2", "3", "4", "5")]
                if len(nums) > len(recs) * 0.5:
                    rcol = c
                    break
            print(f"  열 판별: 문장={tcol} / 판정={rcol}")
            if not tcol or not rcol:
                print("  판별 실패 — 열 이름을 확인해 주세요:", cols)
            else:
                sel = [(Norm.clean(r[tcol]), int(r[rcol])) for r in recs
                       if str(r.get(rcol, "")).strip() in ("1", "2", "3", "4", "5")
                       and len(str(r.get(tcol, ""))) > 10]
                ratings = [x[1] for x in sel]
                dist = Counter(ratings)
                print(f"  표본 {len(sel)}건 / 판정 분포 {dict(sorted(dist.items()))}")
                if len(dist) < 2:
                    print("  판정이 한 가지뿐이라 상관을 계산할 수 없습니다.")
                else:
                    feats = [Analyzer.from_sentences([t]) for t, _ in sel]
                    print(f"\n  {'지표':<14}{'Spearman':>10}   해석")
                    out_rows = []
                    for k in keys:
                        xs = [f.as_dict()[k] for f in feats]
                        if len(set(xs)) < 3:
                            continue
                        r = spearman(xs, ratings)
                        out_rows.append((k, r))
                    for k, r in sorted(out_rows, key=lambda x: -abs(x[1])):
                        if abs(r) >= 0.4:
                            note = "사람 판단을 잘 예측"
                        elif abs(r) >= 0.2:
                            note = "약한 관련"
                        else:
                            note = "관련 없음"
                        print(f"  {METRIC_LABELS[k]:<14}{r:>+10.3f}   {note}")
                    print(f"\n  주의: 표본 {len(sel)}건, 평정자 1인입니다.")
                    print("  상관이 높은 지표라도 확정 근거로 쓰지 말고,"
                          " 3인 확대 후 재계산하세요.")

    print("\n" + "=" * 76)
    with open(f"{a.out}_docs.json", "w", encoding="utf-8") as f:
        json.dump({k: v.as_dict() for k, v in doc_feats.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {a.out}_docs.json")


if __name__ == "__main__":
    main()
