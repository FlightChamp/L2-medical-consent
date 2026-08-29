#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kofih_dict2.py — 동의서 기준 의료용어 대역 사전 구축 (KOFIH 다국어 자료)
=======================================================================
기존 접근의 문제:
    KOFIH 쪽에서 용어를 뽑아 동의서와 대조 → 교집합 0개.
    페이지 단위 동시출현으로 후보를 만들어 "가는·가장·것이" 같은
    조사 붙은 일반어가 후보를 뒤덮음.

이 스크립트의 접근:
    (1) 탐색 방향을 뒤집는다
        검증해야 할 대상은 '동의서에 등장하는 용어'뿐이다.
        docs/*.txt 에서 의료용어를 먼저 확정하고, 그 용어만 KOFIH 에서 찾는다.
    (2) 문장 교대 대역 구조를 쓴다
        KOFIH 자료는 한 페이지 안에서 한국어 블록과 외국어 블록이 번갈아 나온다.
        페이지 전체 동시출현이 아니라 '인접한 블록쌍' 안에서만 후보를 찾는다.
    (3) NER 로 후보를 정제한다
        SungJoo/medical-ner-koelectra 로 용어를 뽑아 일반어를 원천 차단한다.
        모델 로딩 실패 시 규칙 기반으로 자동 대체한다.

신뢰도 등급:
    A  한국어 용어 바로 뒤 괄호에 외국어 병기가 직접 확인됨
    B  정렬된 블록쌍에서 배경 대비 충분히 높은 빈도로 공기 (근거 2건 이상)
    C  근거 1건뿐 — 후보로만 남김

산출:
    kofih_dict2.json   전체 결과
    kofih_dict2.tsv    표 형태 (검토용)
    kofih_dict2_pairs.tsv  정렬된 블록쌍 표본 (구조 검증용)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python kofih_dict2.py
    python kofih_dict2.py --langs vi --min-support 1
    python kofih_dict2.py --no-ner            # NER 없이 규칙 기반만
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

KOFIH_DIR = "/home/hufs/shared/data/kofih_sel"
DOCS_DIR = "docs"
NER_MODEL = "SungJoo/medical-ner-koelectra"

LANGS = ("en", "zh", "vi")
LANG_NAMES = {"en": "영어", "zh": "중국어", "vi": "베트남어"}


# ===========================================================================
# 텍스트 정규화
# ===========================================================================

class Norm:
    CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, t: str) -> str:
        # KOFIH PDF 는 한글 사이 공백이 \u0001(SOH) 로 들어온다
        t = cls.CTRL.sub(" ", t)
        t = unicodedata.normalize("NFC", t)   # 베트남어 성조 결합을 보존
        t = cls.WS.sub(" ", t)
        return t.strip()


# ===========================================================================
# 언어 판정
# ===========================================================================

VN_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
               "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


class LangID:
    @staticmethod
    def counts(t: str) -> Dict[str, int]:
        ko = han = lat = vn = 0
        for ch in t:
            if "가" <= ch <= "힣":
                ko += 1
            elif "\u4e00" <= ch <= "\u9fff":
                han += 1
            elif ch.isalpha() and ord(ch) < 0x250:
                lat += 1
                if ch in VN_MARKS:
                    vn += 1
            elif ch in VN_MARKS:
                lat += 1
                vn += 1
        return {"ko": ko, "han": han, "lat": lat, "vn": vn}

    @classmethod
    def detect(cls, t: str, target: str) -> str:
        """'ko' / target / 'other' 중 하나로 판정."""
        c = cls.counts(t)
        total = c["ko"] + c["han"] + c["lat"]
        if total < 3:
            return "other"
        if c["ko"] / total >= 0.30:
            return "ko"
        if target == "zh" and c["han"] / total >= 0.30:
            return "zh"
        if target in ("en", "vi") and c["lat"] / total >= 0.50:
            # 베트남어는 성조 부호 유무로 영어와 구분되지만,
            # 단일 언어 자료이므로 파일명이 알려주는 target 을 신뢰한다.
            return target
        return "other"


# ===========================================================================
# PDF 추출
# ===========================================================================

class PdfReader:
    def __init__(self):
        self.backend = None
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.backend = "pymupdf"
        except ImportError:
            try:
                import pdfplumber
                self.pdfplumber = pdfplumber
                self.backend = "pdfplumber"
            except ImportError:
                sys.exit("[FATAL] PyMuPDF(fitz) 또는 pdfplumber 가 필요합니다.\n"
                         "        pip install pymupdf")
        print(f"[PDF] backend = {self.backend}")

    def pages(self, path: str) -> List[List[str]]:
        """페이지별 줄 목록."""
        out: List[List[str]] = []
        if self.backend == "pymupdf":
            doc = self.fitz.open(path)
            for pg in doc:
                txt = pg.get_text("text") or ""
                out.append([Norm.clean(l) for l in txt.split("\n") if l.strip()])
            doc.close()
        else:
            with self.pdfplumber.open(path) as pdf:
                for pg in pdf.pages:
                    txt = pg.extract_text() or ""
                    out.append([Norm.clean(l) for l in txt.split("\n") if l.strip()])
        return out


# ===========================================================================
# 블록 정렬
# ===========================================================================

@dataclass
class Block:
    lang: str
    text: str
    page: int


@dataclass
class Pair:
    ko: str
    fo: str
    page: int
    vol: str


class Aligner:
    """페이지 안에서 연속된 같은 언어 줄을 블록으로 묶고, 인접 블록을 짝짓는다."""

    @staticmethod
    def blocks(pages: List[List[str]], target: str) -> List[Block]:
        out: List[Block] = []
        for pno, lines in enumerate(pages, 1):
            cur_lang, buf = None, []
            for line in lines:
                lg = LangID.detect(line, target)
                if lg == "other":
                    continue
                if lg != cur_lang and buf:
                    out.append(Block(cur_lang, " ".join(buf), pno))
                    buf = []
                cur_lang = lg
                buf.append(line)
            if buf and cur_lang:
                out.append(Block(cur_lang, " ".join(buf), pno))
        return out

    @staticmethod
    def pairs(blocks: List[Block], target: str, vol: str) -> Tuple[List[Pair], dict]:
        """한국어 블록에 인접한 외국어 블록을 짝짓는다 (뒤 우선, 없으면 앞)."""
        pairs: List[Pair] = []
        n_ko = sum(1 for b in blocks if b.lang == "ko")
        matched = 0
        for i, b in enumerate(blocks):
            if b.lang != "ko":
                continue
            cand = None
            if i + 1 < len(blocks) and blocks[i + 1].lang == target \
                    and blocks[i + 1].page == b.page:
                cand = blocks[i + 1]
            elif i - 1 >= 0 and blocks[i - 1].lang == target \
                    and blocks[i - 1].page == b.page:
                cand = blocks[i - 1]
            if cand:
                matched += 1
                pairs.append(Pair(b.text, cand.text, b.page, vol))
        stats = {
            "blocks": len(blocks),
            "ko_blocks": n_ko,
            "fo_blocks": sum(1 for b in blocks if b.lang == target),
            "paired": matched,
            "pair_rate": matched / n_ko if n_ko else 0.0,
        }
        return pairs, stats


# ===========================================================================
# 한국어 의료용어 추출
# ===========================================================================

JOSA = ("으로써", "으로서", "에서는", "에게는", "이라는", "라는", "으로", "에서",
        "에게", "부터", "까지", "이나", "나", "와", "과", "은", "는", "이",
        "가", "을", "를", "의", "에", "도", "만", "로", "및")

STOP = {"환자", "수술", "치료", "검사", "시행", "경우", "가능", "발생", "필요",
        "설명", "동의", "내용", "방법", "결과", "이상", "이하", "정도", "때문",
        "위해", "대해", "관련", "확인", "기타", "본인", "의사", "병원", "진료"}


class TermExtractor:
    def __init__(self, use_ner: bool):
        self.pipe = None
        self.mode = "rule"
        if use_ner:
            self._try_ner()

    def _try_ner(self):
        try:
            from transformers import (AutoModelForTokenClassification,
                                      AutoTokenizer, pipeline)
            tok = AutoTokenizer.from_pretrained(NER_MODEL)
            mdl = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
            self.pipe = pipeline("token-classification", model=mdl, tokenizer=tok,
                                 aggregation_strategy="simple", device=0)
            self.mode = "ner"
            labels = sorted(set(str(v) for v in mdl.config.id2label.values()))
            print(f"[NER] {NER_MODEL} 로딩 성공  labels={labels}")
        except Exception as e:
            print(f"[NER] 로딩 실패 → 규칙 기반으로 대체합니다 ({type(e).__name__}: {e})")
            self.pipe = None
            self.mode = "rule"

    @staticmethod
    def strip_josa(w: str) -> str:
        for j in sorted(JOSA, key=len, reverse=True):
            if len(w) > len(j) + 1 and w.endswith(j):
                return w[: -len(j)]
        return w

    def _clean(self, cands: Sequence[str]) -> Set[str]:
        out = set()
        for c in cands:
            c = self.strip_josa(re.sub(r"[^가-힣]", "", c))
            if 2 <= len(c) <= 10 and c not in STOP:
                out.add(c)
        return out

    def extract(self, text: str) -> Set[str]:
        if self.pipe is not None:
            found = []
            # 모델 입력 길이 제한을 고려해 잘라서 처리
            for i in range(0, len(text), 800):
                chunk = text[i:i + 800]
                try:
                    for e in self.pipe(chunk):
                        if str(e.get("entity_group", "O")).upper() != "O":
                            found.append(e["word"])
                except Exception:
                    continue
            return self._clean(found)
        # 규칙 기반 대체: 한글 어절 중 길이 2~10, 조사 제거 후 불용어 제외
        return self._clean(re.findall(r"[가-힣]{2,12}", text))


# ===========================================================================
# 대역어 후보 추출
# ===========================================================================

class CandidateMiner:
    PAREN = re.compile(r"([가-힣]{2,12})\s*[（(]\s*([^)）]{2,40})\s*[)）]")

    def __init__(self, lang: str):
        self.lang = lang

    def ngrams(self, text: str) -> Counter:
        t = text.lower()
        if self.lang == "zh":
            s = re.sub(r"[^\u4e00-\u9fff]", "", t)
            return Counter(s[i:i + n] for n in (2, 3, 4)
                           for i in range(len(s) - n + 1))
        words = [w for w in re.findall(r"[a-zà-ỹ]+", t) if len(w) >= 3]
        c = Counter(words)
        c.update(" ".join(words[i:i + 2]) for i in range(len(words) - 1))
        return c

    def paren_glosses(self, ko_text: str) -> List[Tuple[str, str]]:
        out = []
        for m in self.PAREN.finditer(ko_text):
            ko, fo = m.group(1), Norm.clean(m.group(2))
            if self.lang == "zh" and re.search(r"[\u4e00-\u9fff]", fo):
                out.append((ko, fo))
            elif self.lang in ("en", "vi") and re.search(r"[a-zA-Zà-ỹ]", fo):
                out.append((ko, fo))
        return out


# ===========================================================================
# 메인 파이프라인
# ===========================================================================

@dataclass
class Entry:
    term: str
    lang: str
    candidate: str
    grade: str
    support: int
    ratio: float
    evidence: List[str] = field(default_factory=list)


class Builder:
    def __init__(self, args):
        self.a = args
        self.reader = PdfReader()
        self.extractor = TermExtractor(not args.no_ner)
        self.stats: Dict[str, dict] = {}
        self.all_pairs: Dict[str, List[Pair]] = defaultdict(list)
        self.entries: List[Entry] = []

    # -- 1. 동의서 용어 -----------------------------------------------------

    def consent_terms(self) -> Set[str]:
        paths = [p for p in sorted(glob.glob(os.path.join(self.a.docs, "*.txt")))
                 if not os.path.basename(p).startswith("syn")]
        if not paths:
            sys.exit(f"[FATAL] {self.a.docs}/*.txt 없음")
        terms: Counter = Counter()
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                txt = Norm.clean(f.read())
            for t in self.extractor.extract(txt):
                terms[t] += 1
        keep = {t for t, c in terms.items() if c >= self.a.min_docs}
        print(f"\n[TERMS] 동의서 {len(paths)}건에서 용어 {len(terms)}개 추출 "
              f"→ {self.a.min_docs}개 문서 이상 등장 {len(keep)}개 "
              f"(방식: {self.extractor.mode})")
        print("  상위 30: " + ", ".join(t for t, _ in terms.most_common(30)))
        return keep

    # -- 2. KOFIH 정렬 ------------------------------------------------------

    def load_kofih(self):
        print("\n[KOFIH] 자료 정렬")
        print(f"  {'파일':<28}{'쪽':>5}{'KO블록':>8}{'외국블록':>9}{'정렬쌍':>8}{'정렬률':>8}")
        for lang in self.a.langs:
            for path in sorted(glob.glob(os.path.join(self.a.kofih, f"*_{lang}_*.pdf"))):
                vol = os.path.splitext(os.path.basename(path))[0]
                try:
                    pages = self.reader.pages(path)
                except Exception as e:
                    print(f"  {vol:<28} 읽기 실패: {e}")
                    continue
                blocks = Aligner.blocks(pages, lang)
                pairs, st = Aligner.pairs(blocks, lang, vol)
                st["pages"] = len(pages)
                self.stats[vol] = st
                self.all_pairs[lang].extend(pairs)
                print(f"  {vol:<28}{len(pages):>5}{st['ko_blocks']:>8}"
                      f"{st['fo_blocks']:>9}{st['paired']:>8}{st['pair_rate']:>7.0%}")
        for lang in self.a.langs:
            print(f"  → {LANG_NAMES[lang]} 정렬쌍 총 {len(self.all_pairs[lang])}건")

    # -- 3. 대역어 채굴 -----------------------------------------------------

    def mine(self, terms: Set[str]):
        print("\n[MINE] 대역어 후보 추출")
        for lang in self.a.langs:
            pairs = self.all_pairs[lang]
            if not pairs:
                print(f"  {LANG_NAMES[lang]}: 정렬쌍 없음 — 건너뜀")
                continue
            miner = CandidateMiner(lang)

            # 배경 분포
            bg = Counter()
            for p in pairs:
                bg.update(miner.ngrams(p.fo).keys())
            n_bg = len(pairs)

            # A등급: 괄호 병기
            gloss: Dict[str, Counter] = defaultdict(Counter)
            for p in pairs:
                for ko, fo in miner.paren_glosses(p.ko):
                    if ko in terms:
                        gloss[ko][fo.lower()] += 1

            hit = 0
            for term in sorted(terms):
                pos = [p for p in pairs if term in p.ko]
                if not pos:
                    continue
                hit += 1
                if gloss.get(term):
                    fo, c = gloss[term].most_common(1)[0]
                    self.entries.append(Entry(term, lang, fo, "A", c, float("inf"),
                                              [p.vol for p in pos[:3]]))
                    continue
                cnt = Counter()
                for p in pos:
                    cnt.update(miner.ngrams(p.fo).keys())
                scored = []
                for cand, c in cnt.items():
                    if c < self.a.min_support:
                        continue
                    p_pos = c / len(pos)
                    p_bg = bg.get(cand, 0) / n_bg
                    if p_bg <= 0:
                        continue
                    ratio = p_pos / p_bg
                    if ratio >= self.a.min_ratio:
                        scored.append((ratio, c, cand))
                scored.sort(reverse=True)
                for ratio, c, cand in scored[:1]:
                    grade = "B" if c >= 2 else "C"
                    self.entries.append(Entry(term, lang, cand, grade, c,
                                              round(ratio, 2),
                                              [p.vol for p in pos[:3]]))
            print(f"  {LANG_NAMES[lang]}: 동의서 용어 {len(terms)}개 중 "
                  f"KOFIH 등장 {hit}개 → 후보 확보 "
                  f"{sum(1 for e in self.entries if e.lang == lang)}개")

    # -- 4. 보고 / 저장 -----------------------------------------------------

    def report(self):
        print("\n" + "=" * 78)
        print("결과 요약")
        print("=" * 78)
        for lang in self.a.langs:
            sel = [e for e in self.entries if e.lang == lang]
            g = Counter(e.grade for e in sel)
            print(f"  {LANG_NAMES[lang]:<6} 총 {len(sel):>4}  "
                  f"A {g['A']:>3} / B {g['B']:>3} / C {g['C']:>3}")

        print("\n[A·B등급 표본]")
        print(f"  {'용어':<12}{'언어':<6}{'대역어 후보':<28}{'등급':<5}{'근거':>5}{'비율':>8}")
        shown = [e for e in self.entries if e.grade in ("A", "B")]
        shown.sort(key=lambda e: (-e.support, e.term))
        for e in shown[:40]:
            r = "∞" if e.ratio == float("inf") else f"{e.ratio:.1f}"
            print(f"  {e.term:<12}{LANG_NAMES[e.lang]:<6}{e.candidate[:26]:<28}"
                  f"{e.grade:<5}{e.support:>5}{r:>8}")

        # 알려진 정답으로 자체 검증
        print("\n[자체 검증] 이미 확인된 대응관계가 재현되는가")
        checks = [("갑상선", "vi", "tuyến giáp"), ("갑상선", "en", "thyroid"),
                  ("갑상선", "zh", "甲状腺")]
        for term, lang, expect in checks:
            if lang not in self.a.langs:
                continue
            got = [e for e in self.entries if e.term == term and e.lang == lang]
            if not got:
                print(f"  {term}/{LANG_NAMES[lang]}: 후보 없음 (기대 '{expect}')")
            else:
                ok = any(expect.lower() in e.candidate.lower() for e in got)
                mark = "OK" if ok else "불일치"
                print(f"  {term}/{LANG_NAMES[lang]}: {got[0].candidate} "
                      f"[{got[0].grade}] — 기대 '{expect}' → {mark}")
        print("=" * 78)

    def save(self):
        out = {
            "stats": self.stats,
            "entries": [vars(e) for e in self.entries],
        }
        with open("kofih_dict2.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        with open("kofih_dict2.tsv", "w", encoding="utf-8") as f:
            f.write("term\tlang\tcandidate\tgrade\tsupport\tratio\tevidence\n")
            for e in sorted(self.entries, key=lambda x: (x.term, x.lang)):
                f.write(f"{e.term}\t{e.lang}\t{e.candidate}\t{e.grade}\t"
                        f"{e.support}\t{e.ratio}\t{','.join(e.evidence)}\n")
        with open("kofih_dict2_pairs.tsv", "w", encoding="utf-8") as f:
            f.write("lang\tvol\tpage\tko\tforeign\n")
            for lang in self.a.langs:
                for p in self.all_pairs[lang][:300]:
                    f.write(f"{lang}\t{p.vol}\t{p.page}\t{p.ko[:200]}\t{p.fo[:200]}\n")
        print("\n[SAVE] kofih_dict2.json / kofih_dict2.tsv / kofih_dict2_pairs.tsv")
        print("       pairs.tsv 를 열어 한국어-외국어 짝이 실제로 맞는지 먼저 확인하세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kofih", default=KOFIH_DIR)
    ap.add_argument("--docs", default=DOCS_DIR)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--min-docs", type=int, default=2,
                    help="이 개수 이상의 동의서에 등장하는 용어만 대상으로 삼음")
    ap.add_argument("--min-support", type=int, default=2,
                    help="정렬쌍에서 최소 몇 번 나와야 후보로 인정할지")
    ap.add_argument("--min-ratio", type=float, default=3.0,
                    help="배경 출현율 대비 최소 몇 배여야 후보로 인정할지")
    ap.add_argument("--no-ner", action="store_true")
    a = ap.parse_args()

    b = Builder(a)
    terms = b.consent_terms()
    b.load_kofih()
    b.mine(terms)
    b.report()
    b.save()


if __name__ == "__main__":
    main()
