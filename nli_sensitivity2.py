#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nli_sensitivity2.py — NLI 민감도 실험 v2
========================================
v1에서 드러난 세 가지 문제를 정면으로 해결한다.

  (1) 표본 부족      TERM_SWAP n=9, SEVERITY_FLIP n=3 → 목표치까지 표적 표집
  (2) 충돌 케이스    대체어가 원문에 이미 존재해 무효였던 사례 → 원천 차단
  (3) 집계 방식 의심 서식 청크가 최댓값을 오염시킨 정황 → 4가지 집계를 동시 측정

집계 4종 (청크별 점수를 한 번만 계산하고 파생 — 추가 GPU 비용 없음)
  GLOBAL    전체 청크 entailment 최댓값                (v1 방식)
  FILTERED  서식 뼈대 청크를 제외한 최댓값
  TOPK      서식 제외 + 어휘 겹침 상위 k개만 대상       (실전 적용 가능한 방식)
  LOCAL     원문 문장을 포함한 청크만 대상              (오라클 상한)

  LOCAL 은 잘 분리되는데 GLOBAL 이 무너지면 → 범인은 NLI 모델이 아니라 max 집계.
  그 경우 해결책은 모델 교체가 아니라 후보 청크를 검색으로 좁히는 것.

주요 개선
  · 표적 표집: 각 오염 유형에 해당하는 표현이 실제로 들어있는 문장만 골라 목표 n 확보
  · 용어 치환: 같은 범주 안에서 '원문에 등장하지 않는' 대체어만 선택 → collides 항상 False
  · 조사 보정: 탈구이 → 탈구가, 심장와 → 심장과 (문법 파괴가 탐지를 돕는 교란 제거)
  · 합성 문서(syn*) 기본 제외

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python nli_sensitivity2.py
    python nli_sensitivity2.py --target-n 120 --topk 10
    python nli_sensitivity2.py --include-syn --dump-scaffold
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


# ===========================================================================
# 설정
# ===========================================================================

@dataclass
class Config:
    docs_dir: str = "docs"
    out_prefix: str = "nli_sens2"
    include_syn: bool = False
    target_n: int = 80           # 오염 유형별 목표 표본 수
    min_sent_chars: int = 20
    max_sent_chars: int = 300
    chunk_max_span: int = 3
    topk: int = 10               # TOPK 집계에서 볼 청크 수
    batch_size: int = 128
    max_length: int = 256
    seed: int = 20260826
    tau: float = 0.5


MODES = ("GLOBAL", "FILTERED", "TOPK", "LOCAL")


# ===========================================================================
# 전처리
# ===========================================================================

class TextNormalizer:
    _CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, t: str) -> str:
        t = unicodedata.normalize("NFKC", t)
        t = cls._CTRL.sub(" ", t)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = cls._WS.sub(" ", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()


class SentenceSplitter:
    _NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    _BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls._BOUND.split(t):
            if not p:
                continue
            s = p.strip()
            if s and not cls._NUM_PREFIX.match(s):
                out.append(s)
        return out


# ===========================================================================
# 조사 보정
# ===========================================================================

class JosaFixer:
    """치환으로 깨진 조사를 받침에 맞게 되돌린다.

    문법 오류 자체가 NLI 탐지를 도와주면 오염 탐지율이 부풀려지므로,
    '의미만 바뀌고 문법은 멀쩡한' 오염문을 만들기 위해 필요하다.
    """

    PAIRS = (("이", "가"), ("을", "를"), ("은", "는"),
             ("과", "와"), ("으로", "로"), ("이었", "였"), ("아", "야"))

    @staticmethod
    def has_batchim(ch: str) -> Optional[bool]:
        if not ch or not ("가" <= ch <= "힣"):
            return None
        return (ord(ch) - 0xAC00) % 28 != 0

    @classmethod
    def fix_after(cls, text: str, idx_end: int) -> str:
        """idx_end 위치(치환된 단어 바로 뒤)의 조사를 앞 글자 받침에 맞춘다."""
        if idx_end <= 0 or idx_end > len(text):
            return text
        b = cls.has_batchim(text[idx_end - 1])
        if b is None:
            return text
        rest = text[idx_end:]
        for withb, without in cls.PAIRS:
            if rest.startswith(withb) and not b:
                return text[:idx_end] + without + rest[len(withb):]
            if rest.startswith(without) and b:
                return text[:idx_end] + withb + rest[len(without):]
        return text


# ===========================================================================
# 오염 주입 v2
# ===========================================================================

@dataclass
class Perturbation:
    kind: str
    text: str
    detail: str


class Perturber:
    TERM_GROUPS: Dict[str, List[str]] = {
        "장기": ["갑상선", "담낭", "신장", "간", "폐", "심장", "위", "대장",
                 "방광", "전립선", "유방", "식도", "췌장", "비장", "자궁"],
        "관절": ["고관절", "슬관절", "견관절", "주관절", "족관절", "수근관절"],
        "조직": ["인대", "힘줄", "연골", "신경", "혈관", "근육", "뼈", "피부"],
        "병태": ["골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양"],
    }

    NUM_PAT = re.compile(r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번)")
    UNIT_MAP = {"주": "개월", "일": "주", "시간": "일", "분": "시간", "개월": "년"}

    NEGATE_RULES = (("있습니다", "없습니다"), ("없습니다", "있습니다"),
                    ("가능합니다", "불가능합니다"), ("필요합니다", "필요하지 않습니다"),
                    ("해야 합니다", "하지 않아도 됩니다"),
                    ("됩니다", "되지 않습니다"), ("합니다", "하지 않습니다"))

    SIDE_RULES = (("좌측", "우측"), ("우측", "좌측"),
                  ("왼쪽", "오른쪽"), ("오른쪽", "왼쪽"))

    SEVERITY_RULES = (("드물게", "흔하게"), ("드문", "흔한"), ("드뭅니다", "흔합니다"),
                      ("흔하게", "드물게"), ("흔히", "드물게"), ("흔한", "드문"),
                      ("간혹", "대부분"), ("때때로", "항상"), ("거의 없", "매우 많"),
                      ("낮습니다", "높습니다"), ("낮은", "높은"), ("낮게", "높게"),
                      ("높습니다", "낮습니다"), ("높은", "낮은"),
                      ("경미한", "심각한"), ("가벼운", "치명적인"), ("경한", "중한"),
                      ("심각한", "경미한"), ("심한", "가벼운"),
                      ("일시적", "영구적"), ("영구적", "일시적"),
                      ("대부분", "일부"), ("일부", "대부분"), ("매우", "거의"))

    FABRICATIONS = ("또한 이 수술은 약 1시간 정도 소요됩니다.",
                    "또한 이 수술의 성공률은 99% 이상입니다.",
                    "또한 수술 후 3일째부터 정상 식사가 가능합니다.",
                    "또한 이 시술은 건강보험이 전액 적용됩니다.")

    OFF_TOPIC = ("오늘 서울의 최고 기온은 28도이며 오후 늦게 비가 내리겠습니다.",
                 "이 제품의 보증 기간은 구매일로부터 2년이며 소모품은 제외됩니다.",
                 "다음 정기 주주총회는 본사 대회의실에서 개최될 예정입니다.")

    def __init__(self, doc_text: str, rng: random.Random):
        self.doc = doc_text
        self.rng = rng

    # -- 적격 판정 (표적 표집용) -----------------------------------------

    def eligible(self, s: str) -> Set[str]:
        out: Set[str] = {"FABRICATE", "OFF_TOPIC"}
        if self._find_term(s):
            out.add("TERM_SWAP")
        m = self.NUM_PAT.search(s)
        if m:
            out.add("NUM_CHANGE")
            if m.group(2) in self.UNIT_MAP:
                out.add("UNIT_CHANGE")
        if any(a in s for a, _ in self.NEGATE_RULES):
            out.add("NEGATE")
        if any(a in s for a, _ in self.SIDE_RULES):
            out.add("SIDE_FLIP")
        if any(a in s for a, _ in self.SEVERITY_RULES):
            out.add("SEVERITY_FLIP")
        return out

    def _find_term(self, s: str) -> Optional[Tuple[str, str, str]]:
        """(원어, 대체어, 범주). 대체어는 원문에 존재하지 않는 것만 고른다."""
        cands = []
        for cat, terms in self.TERM_GROUPS.items():
            for t in sorted(terms, key=len, reverse=True):
                if t in s:
                    alts = [x for x in terms if x != t and x not in self.doc]
                    if alts:
                        cands.append((t, alts, cat))
                    break
        if not cands:
            return None
        t, alts, cat = cands[0]
        return (t, self.rng.choice(sorted(alts)), cat)

    # -- 생성 -------------------------------------------------------------

    def make(self, s: str, kind: str) -> Optional[Perturbation]:
        fn = {
            "TERM_SWAP": self._term, "NUM_CHANGE": self._num,
            "UNIT_CHANGE": self._unit, "NEGATE": self._negate,
            "SIDE_FLIP": self._side, "SEVERITY_FLIP": self._severity,
            "FABRICATE": self._fab, "OFF_TOPIC": self._off,
        }[kind]
        p = fn(s)
        if p and p.text.strip() and p.text.strip() != s.strip():
            return p
        return None

    def _term(self, s: str) -> Optional[Perturbation]:
        f = self._find_term(s)
        if not f:
            return None
        src, dst, cat = f
        i = s.find(src)
        out = s[:i] + dst + s[i + len(src):]
        out = JosaFixer.fix_after(out, i + len(dst))
        return Perturbation("TERM_SWAP", out, f"{src}→{dst}({cat})")

    def _num(self, s: str) -> Optional[Perturbation]:
        m = self.NUM_PAT.search(s)
        if not m:
            return None
        old = int(m.group(1))
        new = old * 3 + 1
        rep = f"{new}{m.group(2)}"
        return Perturbation("NUM_CHANGE", s[:m.start()] + rep + s[m.end():],
                            f"{old}{m.group(2)}→{rep}")

    def _unit(self, s: str) -> Optional[Perturbation]:
        m = self.NUM_PAT.search(s)
        if not m or m.group(2) not in self.UNIT_MAP:
            return None
        rep = f"{m.group(1)}{self.UNIT_MAP[m.group(2)]}"
        return Perturbation("UNIT_CHANGE", s[:m.start()] + rep + s[m.end():],
                            f"{m.group(1)}{m.group(2)}→{rep}")

    def _rule(self, s: str, kind: str, rules) -> Optional[Perturbation]:
        for a, b in rules:
            if a in s:
                i = s.find(a)
                out = s[:i] + b + s[i + len(a):]
                out = JosaFixer.fix_after(out, i + len(b))
                return Perturbation(kind, out, f"{a}→{b}")
        return None

    def _negate(self, s):
        return self._rule(s, "NEGATE", self.NEGATE_RULES)

    def _side(self, s):
        return self._rule(s, "SIDE_FLIP", self.SIDE_RULES)

    def _severity(self, s):
        return self._rule(s, "SEVERITY_FLIP", self.SEVERITY_RULES)

    def _fab(self, s):
        add = self.rng.choice(self.FABRICATIONS)
        return Perturbation("FABRICATE", s.rstrip() + " " + add, add)

    def _off(self, s):
        return Perturbation("OFF_TOPIC", self.rng.choice(self.OFF_TOPIC), "무관 문장")


# ===========================================================================
# 청크 저장소 (서식 뼈대 판정 포함)
# ===========================================================================

SCAFFOLD_LABELS = ("병록번호", "생년월일", "주민등록", "성명", "환자명", "보호자",
                   "진료과", "주치의", "수술명", "시술명", "휴대전화", "집전화",
                   "전화번호", "주 소", "주소", "서명", "날인", "작성일", "성별",
                   "나이", "동의서", "설명의사", "면허번호", "일 시", "관계")


@dataclass
class Chunk:
    text: str
    sent_ids: Tuple[int, ...]
    scaffold: bool


class ChunkStore:
    def __init__(self, sents: List[str], span: int):
        self.sents = sents
        self.chunks: List[Chunk] = []
        seen = set()
        for k in range(1, span + 1):
            for i in range(0, len(sents) - k + 1):
                t = " ".join(sents[i:i + k])
                if t in seen:
                    continue
                seen.add(t)
                self.chunks.append(Chunk(t, tuple(range(i, i + k)),
                                         self.is_scaffold(t)))
        if not self.chunks:
            self.chunks = [Chunk("(빈 문서)", (0,), False)]
        self._grams = [self._bigrams(c.text) for c in self.chunks]

    @staticmethod
    def is_scaffold(t: str) -> bool:
        """내용 없는 서식 뼈대(머리글·서명란·개인정보 항목)인가."""
        if len(t) < 25:
            return True
        if t.count(":") >= 2:
            return True
        hits = sum(1 for lab in SCAFFOLD_LABELS if lab in t)
        # 라벨이 여러 개인데 종결어미가 없으면 문장이 아니라 서식이다.
        if hits >= 2 and not re.search(r"(다|요|음|임|함)[.\s]*$", t):
            return True
        if len(re.findall(r"[_\-]{3,}", t)) >= 1:
            return True
        hangul = len(re.findall(r"[가-힣]", t))
        return hangul / max(len(t), 1) < 0.35

    @staticmethod
    def _bigrams(t: str) -> Set[str]:
        s = re.sub(r"\s+", "", t)
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}

    def topk_idx(self, hypothesis: str, k: int, pool: Sequence[int]) -> List[int]:
        hg = self._bigrams(hypothesis)
        scored = []
        for i in pool:
            g = self._grams[i]
            inter = len(hg & g)
            scored.append((inter / max(len(hg | g), 1), i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]


# ===========================================================================
# NLI 채점
# ===========================================================================

class NLIScorer:
    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.cfg = torch, cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] device={self.device} model={MODEL_ID}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()
        id2 = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))
        print(f"[INIT] id2label={id2} -> entail={self.i_ent}", flush=True)

    def all_chunk_scores(self, hypothesis: str, chunks: List[Chunk]) -> List[float]:
        torch = self.torch
        out: List[float] = []
        bs = self.cfg.batch_size
        prem = [c.text for c in chunks]
        for i in range(0, len(prem), bs):
            batch = prem[i:i + bs]
            enc = self.tok(batch, [hypothesis] * len(batch), truncation=True,
                           padding=True, max_length=self.cfg.max_length,
                           return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(p[:, self.i_ent].tolist())
        return out


# ===========================================================================
# 통계
# ===========================================================================

def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = [-x for x in pos], [-x for x in neg]
    allv = sorted(p + n)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = sum(ranks[x] for x in p)
    return (r - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ===========================================================================
# 실험
# ===========================================================================

@dataclass
class Case:
    doc: str
    sent_id: int
    kind: str
    detail: str
    text: str
    scores: Dict[str, float] = field(default_factory=dict)
    best_chunk: str = ""


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.cases: List[Case] = []
        self.scaffold_samples: List[str] = []
        self.n_chunk_total = 0
        self.n_chunk_scaffold = 0

    # -- 로딩 -------------------------------------------------------------

    def load(self) -> Dict[str, str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.docs_dir, "*.txt")))
        if not paths:
            sys.exit(f"[FATAL] {self.cfg.docs_dir}/*.txt 없음")
        docs = {}
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if not self.cfg.include_syn and name.startswith("syn"):
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                docs[name] = TextNormalizer.clean(f.read())
        print(f"[LOAD] 문서 {len(docs)}건: {', '.join(docs)}", flush=True)
        return docs

    # -- 표적 표집 ---------------------------------------------------------

    def plan(self, docs: Dict[str, str]):
        """유형별로 적격 문장을 모아 목표 n 까지 균등 배분한다."""
        prepared = {}
        pool: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for name, text in docs.items():
            sents = SentenceSplitter.split(text)
            store = ChunkStore(sents, self.cfg.chunk_max_span)
            pert = Perturber(text, self.rng)
            prepared[name] = (sents, store, pert)
            self.n_chunk_total += len(store.chunks)
            self.n_chunk_scaffold += sum(1 for c in store.chunks if c.scaffold)
            self.scaffold_samples.extend(
                c.text for c in store.chunks[:200] if c.scaffold)
            for i, s in enumerate(sents):
                if not (self.cfg.min_sent_chars <= len(s) <= self.cfg.max_sent_chars):
                    continue
                for kind in pert.eligible(s):
                    pool[kind].append((name, i))

        assign: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
        print("\n[PLAN] 유형별 적격 문장 수 → 배정")
        for kind, items in sorted(pool.items()):
            self.rng.shuffle(items)
            # 문서별로 고르게 뽑기 위해 라운드로빈
            bydoc: Dict[str, List[int]] = defaultdict(list)
            for d, i in items:
                bydoc[d].append(i)
            picked, docs_cycle = [], sorted(bydoc)
            idx = {d: 0 for d in docs_cycle}
            while len(picked) < self.cfg.target_n:
                progressed = False
                for d in docs_cycle:
                    if idx[d] < len(bydoc[d]) and len(picked) < self.cfg.target_n:
                        picked.append((d, bydoc[d][idx[d]]))
                        idx[d] += 1
                        progressed = True
                if not progressed:
                    break
            for key in picked:
                assign[key].add(kind)
            print(f"  {kind:<16} 적격 {len(items):>5}  →  배정 {len(picked):>4}")
        return prepared, assign

    # -- 실행 -------------------------------------------------------------

    def run(self):
        docs = self.load()
        prepared, assign = self.plan(docs)
        scorer = NLIScorer(self.cfg)
        print(f"\n[CHUNK] 총 {self.n_chunk_total}개 중 서식 판정 "
              f"{self.n_chunk_scaffold}개 ({self.n_chunk_scaffold / max(self.n_chunk_total,1):.1%})",
              flush=True)

        bydoc: Dict[str, List[Tuple[int, Set[str]]]] = defaultdict(list)
        for (d, i), kinds in assign.items():
            bydoc[d].append((i, kinds))

        t0 = time.time()
        for di, doc in enumerate(sorted(bydoc), 1):
            sents, store, pert = prepared[doc]
            content_pool = [i for i, c in enumerate(store.chunks) if not c.scaffold] \
                or list(range(len(store.chunks)))
            todo: List[Case] = []
            for sid, kinds in sorted(bydoc[doc]):
                todo.append(Case(doc, sid, "ORIGINAL", "-", sents[sid]))
                for kind in sorted(kinds):
                    p = pert.make(sents[sid], kind)
                    if p:
                        todo.append(Case(doc, sid, p.kind, p.detail, p.text))

            for c in todo:
                sc = scorer.all_chunk_scores(c.text, store.chunks)
                local_pool = [i for i, ch in enumerate(store.chunks)
                              if c.sent_id in ch.sent_ids] or list(range(len(sc)))
                topk_pool = store.topk_idx(c.text, self.cfg.topk, content_pool)
                gi = max(range(len(sc)), key=lambda j: sc[j])
                c.scores = {
                    "GLOBAL": max(sc),
                    "FILTERED": max(sc[i] for i in content_pool),
                    "TOPK": max(sc[i] for i in topk_pool),
                    "LOCAL": max(sc[i] for i in local_pool),
                }
                c.best_chunk = store.chunks[gi].text[:200]
            self.cases.extend(todo)
            print(f"[{di}/{len(bydoc)}] {doc}: 문장 {len(sents)} / 청크 {len(store.chunks)}"
                  f" (서식 {sum(1 for x in store.chunks if x.scaffold)})"
                  f" / 케이스 {len(todo)} — 누적 {len(self.cases)}, {time.time()-t0:.0f}s",
                  flush=True)
        print(f"[DONE] {len(self.cases)} 케이스 / {time.time()-t0:.0f}초", flush=True)

    # -- 집계 -------------------------------------------------------------

    def report(self) -> dict:
        cfg = self.cfg
        orig = [c for c in self.cases if c.kind == "ORIGINAL"]
        kinds = sorted({c.kind for c in self.cases if c.kind != "ORIGINAL"})
        summary = {"config": asdict(cfg), "n_cases": len(self.cases),
                   "n_original": len(orig),
                   "scaffold_ratio": self.n_chunk_scaffold / max(self.n_chunk_total, 1),
                   "auroc": {}, "detect": {}, "specificity": {}}

        print("\n" + "=" * 78)
        print("NLI 민감도 v2 — 집계 방식별 AUROC (임계값 무관, 1.0=완전 분리)")
        print("=" * 78)
        head = "    {:<16}{:>5}".format("유형", "n") + "".join(f"{m:>11}" for m in MODES)
        print(head)
        for k in kinds:
            pos = [c for c in self.cases if c.kind == k]
            row = f"    {k:<16}{len(pos):>5}"
            summary["auroc"][k] = {}
            for m in MODES:
                a = auroc([c.scores[m] for c in pos], [c.scores[m] for c in orig])
                summary["auroc"][k][m] = round(a, 4)
                row += f"{a:>11.3f}"
            print(row)

        print("\n" + "=" * 78)
        print(f"정상 통과율 / 탐지율  (tau={cfg.tau})")
        print("=" * 78)
        row = f"    {'정상 통과율':<16}{len(orig):>5}"
        for m in MODES:
            s = mean(1.0 if c.scores[m] >= cfg.tau else 0.0 for c in orig)
            summary["specificity"][m] = round(s, 4)
            row += f"{s:>10.1%} "
        print(row)
        print("    " + "-" * 70)
        print("    {:<16}{:>5}".format("유형", "n") + "".join(f"{m:>11}" for m in MODES))
        for k in kinds:
            pos = [c for c in self.cases if c.kind == k]
            row = f"    {k:<16}{len(pos):>5}"
            summary["detect"][k] = {}
            for m in MODES:
                d = sum(1 for c in pos if c.scores[m] < cfg.tau)
                summary["detect"][k][m] = round(d / len(pos), 4) if pos else None
                row += f"{d/len(pos):>10.1%} " if pos else f"{'—':>11}"
            lo, hi = wilson(sum(1 for c in pos if c.scores["TOPK"] < cfg.tau), len(pos))
            print(row + f"   TOPK 95%CI [{lo:.0%},{hi:.0%}]")

        self._verdict(summary, kinds)
        return summary

    def _verdict(self, s: dict, kinds: List[str]):
        print("\n" + "=" * 78)
        print("판정")
        print("=" * 78)
        best = {}
        for k in kinds:
            m = max(MODES, key=lambda m: s["auroc"][k][m])
            best[k] = m
        gains = [(k, s["auroc"][k]["TOPK"] - s["auroc"][k]["GLOBAL"]) for k in kinds]
        gains.sort(key=lambda x: -x[1])
        print("  집계 방식 변경(GLOBAL → TOPK)에 따른 AUROC 변화:")
        for k, g in gains:
            print(f"    {k:<16}{g:+.3f}")
        avg = mean(g for _, g in gains)
        print(f"\n  평균 변화 {avg:+.3f}")
        if avg > 0.03:
            print("  → 서식 청크 제거 + 후보 축소만으로 지표가 개선됩니다.")
            print("     NLI 모델이 아니라 max 집계 방식이 병목이었다는 뜻입니다.")
        elif avg < -0.03:
            print("  → 후보를 좁히면 오히려 나빠집니다. 전역 최댓값을 유지하세요.")
        else:
            print("  → 집계 방식은 큰 영향이 없습니다. 취약 유형은 NLI 자체의 한계입니다.")

        weak = [k for k in kinds if k != "OFF_TOPIC"
                and s["auroc"][k][best[k]] < 0.75]
        print()
        if weak:
            print("  최선의 집계로도 AUROC 0.75 미만인 유형:")
            for k in weak:
                print(f"    · {k}  {s['auroc'][k][best[k]]:.3f} ({best[k]})")
            print("  → 이 유형은 NLI 계열로 해결 불가. 규칙 기반 모듈이 필요합니다.")
        else:
            print("  최선의 집계 기준 전 유형 AUROC 0.75 이상.")
        off = s["auroc"].get("OFF_TOPIC", {}).get("GLOBAL", 1.0)
        print(f"\n  코드 정상성 확인 OFF_TOPIC AUROC(GLOBAL) = {off:.3f}")
        print("=" * 78)

    # -- 저장 -------------------------------------------------------------

    def save(self, summary: dict, dump_scaffold: bool):
        p = self.cfg.out_prefix
        with open(f"{p}_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(f"{p}_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc", "sent_id", "kind", "detail"] + list(MODES)
                       + ["text", "best_chunk"])
            for c in sorted(self.cases, key=lambda x: (x.doc, x.sent_id, x.kind)):
                w.writerow([c.doc, c.sent_id, c.kind, c.detail]
                           + [round(c.scores[m], 4) for m in MODES]
                           + [c.text, c.best_chunk])
        print(f"\n[SAVE] {p}_report.json / {p}_cases.csv")
        if dump_scaffold:
            with open(f"{p}_scaffold.txt", "w", encoding="utf-8") as f:
                for t in self.scaffold_samples[:400]:
                    f.write(t + "\n")
            print(f"[SAVE] {p}_scaffold.txt — 서식 판정된 청크 표본 "
                  f"(내용 있는 문장이 섞였는지 눈으로 확인하세요)")


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out-prefix", default="nli_sens2")
    ap.add_argument("--target-n", type=int, default=80)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--include-syn", action="store_true")
    ap.add_argument("--dump-scaffold", action="store_true")
    a = ap.parse_args()

    cfg = Config(docs_dir=a.docs, out_prefix=a.out_prefix, target_n=a.target_n,
                 topk=a.topk, tau=a.tau, batch_size=a.batch_size,
                 include_syn=a.include_syn)
    exp = Experiment(cfg)
    exp.run()
    exp.save(exp.report(), a.dump_scaffold)


if __name__ == "__main__":
    main()
