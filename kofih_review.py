#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kofih_review.py — 용어 대역 검토 시트 생성 (사람이 확정하는 방식)
=================================================================
왜 자동 추출을 포기하는가:
    비평행 PDF 에서 이중언어 용어쌍을 자동 추출하는 것은 그 자체로 연구 주제다.
    n-gram 통계 방식은 세 번 시도해 세 번 실패했고, 정답을 아는 케이스
    (갑상선 ↔ tuyến giáp)조차 재현하지 못했다.

    반면 실제로 필요한 것은 베트남어 용어 30~40개뿐이다.
    영어·중국어·일본어는 AI Hub 말뭉치 기반 사전이 이미 있고 검증도 끝났다.
    이 규모는 기계가 후보 문맥을 찾아주고 사람이 확정하는 편이 빠르고 정확하다.

이 스크립트가 하는 일:
    1) KOFIH PDF 에서 한국어 블록과 인접 외국어 블록을 뽑는다
    2) 지정한 용어가 들어있는 한국어 블록과 그 짝을 찾는다
    3) 외국어 블록에서 '이 용어의 대역어일 가능성이 있는' 후보에 표시를 단다
       (어디까지나 힌트이며 확정은 사람이 한다)
    4) 검토용 TSV 를 만든다 — 사람이 '확정대역어' 열만 채우면 사전이 완성된다

주의:
    정렬 구조가 실제로 맞는지는 이 스크립트가 보증하지 못한다.
    출력 시트의 앞부분 10~20행을 반드시 눈으로 확인하고 시작할 것.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python kofih_review.py                      # 베트남어, 기본 용어 목록
    python kofih_review.py --langs vi en zh
    python kofih_review.py --terms 갑상선 담낭 골절
    python kofih_review.py --max-ex 5           # 용어당 예문 수
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

KOFIH_DIR = "/home/hufs/shared/data/kofih_sel"
LANG_NAMES = {"en": "영어", "zh": "중국어", "vi": "베트남어"}

# 동의서 13건에 실제로 등장하는 용어 중 임상적으로 중요한 것들.
# NER 이 아니라 기존 실험에서 확인된 용어를 손으로 정리한 목록이다.
DEFAULT_TERMS = [
    # 해부 부위
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절",
    "인대", "혈관", "신경", "폐", "심장", "간", "위", "대장", "무릎", "고관절",
    # 질환·병태
    "골절", "탈장", "기흉", "치핵", "종양", "염증", "궤양", "혈전", "색전증",
    "불유합", "성대마비", "저칼슘혈증", "감염", "출혈", "재발",
    # 처치·시술
    "마취", "수혈", "봉합", "절제", "이식", "배액관", "흉관", "내시경",
    # 결과·위험
    "합병증", "후유증", "통증", "부작용", "사망",
]


# ===========================================================================
# 전처리 · 언어 판정 (kofih_dict2.py 와 동일 기준)
# ===========================================================================

class Norm:
    CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, t: str) -> str:
        t = cls.CTRL.sub(" ", t)
        t = unicodedata.normalize("NFC", t)
        return cls.WS.sub(" ", t).strip()


VN_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
               "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def detect(t: str, target: str) -> str:
    ko = han = lat = 0
    for ch in t:
        if "가" <= ch <= "힣":
            ko += 1
        elif "\u4e00" <= ch <= "\u9fff":
            han += 1
        elif (ch.isalpha() and ord(ch) < 0x250) or ch in VN_MARKS:
            lat += 1
    total = ko + han + lat
    if total < 3:
        return "other"
    if ko / total >= 0.30:
        return "ko"
    if target == "zh" and han / total >= 0.30:
        return "zh"
    if target in ("en", "vi") and lat / total >= 0.50:
        return target
    return "other"


# ===========================================================================
# PDF
# ===========================================================================

class PdfReader:
    def __init__(self):
        try:
            import pymupdf
            self.mod, self.backend = pymupdf, "pymupdf"
        except ImportError:
            try:
                import fitz
                self.mod, self.backend = fitz, "fitz"
            except ImportError:
                sys.exit("[FATAL] pymupdf 가 필요합니다: pip install pymupdf")

    def pages(self, path: str) -> List[List[str]]:
        out = []
        doc = self.mod.open(path)
        for pg in doc:
            txt = pg.get_text("text") or ""
            out.append([Norm.clean(l) for l in txt.split("\n") if l.strip()])
        doc.close()
        return out


# ===========================================================================
# 블록 · 정렬
# ===========================================================================

@dataclass
class Pair:
    vol: str
    page: int
    ko: str
    fo: str
    order: str          # 'ko→fo' 또는 'fo→ko' — 자료의 배치 순서 확인용


def build_pairs(pages: List[List[str]], target: str, vol: str) -> List[Pair]:
    blocks: List[Tuple[str, str, int]] = []
    for pno, lines in enumerate(pages, 1):
        cur, buf = None, []
        for line in lines:
            lg = detect(line, target)
            if lg == "other":
                continue
            if lg != cur and buf:
                blocks.append((cur, " ".join(buf), pno))
                buf = []
            cur = lg
            buf.append(line)
        if buf and cur:
            blocks.append((cur, " ".join(buf), pno))

    pairs: List[Pair] = []
    for i, (lg, txt, pno) in enumerate(blocks):
        if lg != "ko":
            continue
        if i + 1 < len(blocks) and blocks[i + 1][0] == target \
                and blocks[i + 1][2] == pno:
            pairs.append(Pair(vol, pno, txt, blocks[i + 1][1], "ko→fo"))
        elif i - 1 >= 0 and blocks[i - 1][0] == target \
                and blocks[i - 1][2] == pno:
            pairs.append(Pair(vol, pno, txt, blocks[i - 1][1], "fo→ko"))
    return pairs


# ===========================================================================
# 후보 힌트 (확정이 아니라 참고용)
# ===========================================================================

class Hinter:
    """해당 용어가 든 문맥에서만 유독 자주 나오는 표현에 표시를 단다.

    자동 확정용이 아니다. 사람이 외국어 문장을 읽을 때 눈이 갈 지점을
    좁혀주는 용도다.
    """

    def __init__(self, lang: str, pairs: Sequence[Pair]):
        self.lang = lang
        self.bg = Counter()
        for p in pairs:
            self.bg.update(set(self.units(p.fo)))
        self.n = max(len(pairs), 1)

    def units(self, text: str) -> List[str]:
        t = text.lower()
        if self.lang == "zh":
            s = re.sub(r"[^\u4e00-\u9fff]", "", t)
            return [s[i:i + n] for n in (2, 3) for i in range(len(s) - n + 1)]
        return [w for w in re.findall(r"[a-zà-ỹ]+", t) if len(w) >= 4]

    def hint(self, foreign_texts: Sequence[str], top: int = 4) -> str:
        if not foreign_texts:
            return ""
        c = Counter()
        for t in foreign_texts:
            c.update(set(self.units(t)))
        scored = []
        for u, k in c.items():
            p_pos = k / len(foreign_texts)
            p_bg = self.bg.get(u, 0) / self.n
            if p_bg > 0 and p_pos / p_bg >= 3.0 and k >= 2:
                scored.append((p_pos / p_bg, u))
        scored.sort(reverse=True)
        return ", ".join(u for _, u in scored[:top])


# ===========================================================================

def window(text: str, term: str, width: int = 90) -> str:
    i = text.find(term)
    if i < 0:
        return text[:width * 2]
    a = max(0, i - width // 2)
    return ("…" if a > 0 else "") + text[a:a + width * 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kofih", default=KOFIH_DIR)
    ap.add_argument("--langs", nargs="+", default=["vi"])
    ap.add_argument("--terms", nargs="+", default=None)
    ap.add_argument("--max-ex", type=int, default=3, help="용어당 예문 수")
    ap.add_argument("--out", default="term_review")
    a = ap.parse_args()

    terms = a.terms or DEFAULT_TERMS
    reader = PdfReader()
    print(f"[PDF] backend={reader.backend}")
    print(f"[TERMS] 검토 대상 {len(terms)}개 (손으로 정리한 목록, NER 미사용)")

    summary = {}
    for lang in a.langs:
        files = sorted(glob.glob(os.path.join(a.kofih, f"*_{lang}_*.pdf")))
        if not files:
            print(f"[{lang}] PDF 없음 — 건너뜀")
            continue
        pairs: List[Pair] = []
        for path in files:
            vol = os.path.splitext(os.path.basename(path))[0]
            try:
                pairs.extend(build_pairs(reader.pages(path), lang, vol))
            except Exception as e:
                print(f"  {vol}: 읽기 실패 {e}")
        order = Counter(p.order for p in pairs)
        print(f"\n[{LANG_NAMES[lang]}] 파일 {len(files)}개 / 대역쌍 {len(pairs)}건 "
              f"(배치 ko→fo {order['ko→fo']} · fo→ko {order['fo→ko']})")

        hinter = Hinter(lang, pairs)
        rows = []
        found = 0
        for term in terms:
            hits = [p for p in pairs if term in p.ko]
            if not hits:
                rows.append([term, "", "", "", "미등장", ""])
                continue
            found += 1
            hint = hinter.hint([p.fo for p in hits])
            for p in hits[:a.max_ex]:
                rows.append([term, p.vol, str(p.page),
                             window(p.ko, term), p.fo[:300], hint])
        summary[lang] = {"pairs": len(pairs), "terms": len(terms), "found": found}
        print(f"  KOFIH 등장 용어 {found}/{len(terms)}개")

        path = f"{a.out}_{lang}.tsv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("용어\t권\t쪽\t한국어문맥\t외국어문장\t후보힌트\t확정대역어\t비고\n")
            for r in rows:
                f.write("\t".join(x.replace("\t", " ") for x in r) + "\t\t\n")
        print(f"  [SAVE] {path}  ({len(rows)}행)")

    print("\n" + "=" * 74)
    print("다음 할 일")
    print("=" * 74)
    print("  1. TSV 를 엑셀/스프레드시트로 열어 앞 10~20행을 먼저 확인하세요.")
    print("     한국어 문맥과 외국어 문장이 같은 내용이 아니면 정렬이 깨진 것이고,")
    print("     그 경우 이 자료로는 사전을 만들 수 없습니다. 거기서 멈추세요.")
    print("  2. 정렬이 맞으면 '확정대역어' 열을 채웁니다. 후보힌트는 참고만 하세요.")
    print("  3. 판단이 안 서는 행은 비워두고 '비고'에 이유를 적으세요.")
    print("  4. 채운 시트를 sheet2dict.py 로 사전 JSON 으로 변환합니다.")
    print("=" * 74)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
