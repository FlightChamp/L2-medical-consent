#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nli_sensitivity.py
==================
NLI 기반 '근거율' 지표의 민감도(재현율)를 통제 실험으로 측정한다.

[왜 필요한가]
    지금까지 측정한 것은 "NLI가 무근거라고 찍은 12건 중 진짜는 1건"  = 정밀도(precision).
    측정하지 않은 것은 "실제 오류 중 NLI가 몇 개를 잡아내는가" = 재현율(recall).
    CometKiwi를 미채택한 근거가 바로 '의료 오류에 둔감함'(갑상선→식도 -0.006)이었으므로,
    같은 시험을 NLI에도 적용하지 않으면 96.2%라는 숫자를 방어할 수 없다.

[설계]
    원문(docs/*.txt)에서 문장을 뽑아
      · 그대로 두면  -> 정답(근거 있음)이어야 한다      : 정상 통과율(specificity)
      · 규칙으로 오염시키면 -> 오답(근거 없음)이어야 한다 : 유형별 탐지율(recall)
    정답을 사람이 만들 필요가 없다. 오염 방식 자체가 정답 라벨이다.

    오염 유형
      TERM_SWAP      해부/질환 용어 치환   (갑상선 -> 식도)
      NUM_CHANGE     수치 변조             (3주 -> 10주)
      UNIT_CHANGE    단위 변조             (3주 -> 3개월)
      NEGATE         긍/부정 반전          (있습니다 -> 없습니다)
      SIDE_FLIP      좌우 반전             (좌측 -> 우측)
      SEVERITY_FLIP  정도 반전             (드물게 -> 흔하게)
      FABRICATE      원문에 없는 절 추가   (실제 환각 "약 1시간" 유형 재현)
      OFF_TOPIC      완전 무관 문장        (코드 정상 동작 확인용 상한 기준)

[출력]
    nli_sensitivity_report.json   집계 결과(임계값 스윕 포함)
    nli_sensitivity_cases.csv     전 케이스 원본/오염문/점수 — 사람이 눈으로 검증 가능

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python nli_sensitivity.py
    python nli_sensitivity.py --max-sents 25          # 문서당 표본 늘리기
    python nli_sensitivity.py --docs docs --out-prefix nli_sens_v2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------------

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


@dataclass
class Config:
    docs_dir: str = "docs"
    out_prefix: str = "nli_sensitivity"
    max_sents_per_doc: int = 15      # 문서당 표본 문장 수
    min_sent_chars: int = 20         # 너무 짧은 문장은 대조 의미가 없음
    max_sent_chars: int = 300
    chunk_max_span: int = 3          # 원문 청크: 연속 1~3문장 (v2 채점과 동일)
    batch_size: int = 128
    max_length: int = 256
    seed: int = 20260826             # 사람 판정 시트와 동일 시드
    thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    target_specificity: float = 0.95  # 이 정상 통과율을 만족하는 최소 임계값에서 재현율 보고


# ----------------------------------------------------------------------------
# 1. 텍스트 정규화 / 문장 분리
# ----------------------------------------------------------------------------

class TextNormalizer:
    """PDF 추출 텍스트의 제어문자·공백 문제를 정리한다.

    KOFIH 자료에서 한글 사이 공백이 \\u0001(SOH)로 들어오는 사례를 확인했으므로
    경기도의료원 텍스트에도 같은 방어를 적용한다.
    """

    _CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WS = re.compile(r"[ \t\u00a0\u3000]+")

    @classmethod
    def clean(cls, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = cls._CTRL.sub(" ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = cls._WS.sub(" ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class SentenceSplitter:
    """한국어 동의서용 경량 문장 분리기.

    마침표만으로 자르면 '1. 수술의 목적' 같은 번호가 끊기므로,
    번호 패턴은 문장 경계로 보지 않는다.
    """

    _NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    _BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, text: str) -> List[str]:
        raw = cls._BOUND.split(text)
        out: List[str] = []
        for piece in raw:
            if piece is None:
                continue
            s = piece.strip()
            if not s:
                continue
            if cls._NUM_PREFIX.match(s):
                continue
            out.append(s)
        return out


# ----------------------------------------------------------------------------
# 2. 오염 주입기
# ----------------------------------------------------------------------------

@dataclass
class Perturbation:
    kind: str
    text: str
    detail: str
    collides: bool = False   # 바꿔 넣은 표현이 원문 다른 곳에 이미 존재하는가


class Perturber:
    """규칙 기반 오염 주입. 각 규칙은 '의미가 실제로 달라지는' 변경만 수행한다."""

    TERM_MAP: Dict[str, str] = {
        "갑상선": "식도",
        "담낭": "신장",
        "척추": "골반",
        "유방": "간",
        "전립선": "방광",
        "무릎": "어깨",
        "고관절": "견관절",
        "슬관절": "주관절",
        "십자인대": "아킬레스건",
        "치핵": "치루",
        "기흉": "복막염",
        "탈장": "종양",
        "골절": "탈구",
        "대장": "위",
        "폐": "심장",
        "혈관": "신경",
        "인공관절": "인공심장판막",
    }

    NUM_PAT = re.compile(r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번)")

    UNIT_MAP: Dict[str, str] = {
        "주": "개월", "일": "주", "시간": "일", "분": "시간", "개월": "년",
    }

    NEGATE_RULES: Sequence[Tuple[str, str]] = (
        ("있습니다", "없습니다"),
        ("없습니다", "있습니다"),
        ("가능합니다", "불가능합니다"),
        ("필요합니다", "필요하지 않습니다"),
        ("해야 합니다", "하지 않아도 됩니다"),
        ("됩니다", "되지 않습니다"),
        ("합니다", "하지 않습니다"),
    )

    SIDE_RULES: Sequence[Tuple[str, str]] = (
        ("좌측", "우측"), ("우측", "좌측"),
        ("왼쪽", "오른쪽"), ("오른쪽", "왼쪽"),
    )

    SEVERITY_RULES: Sequence[Tuple[str, str]] = (
        ("드물게", "흔하게"), ("드문", "흔한"), ("드뭅니다", "흔합니다"),
        ("낮습니다", "높습니다"), ("낮은", "높은"),
        ("경미한", "심각한"), ("심각한", "경미한"),
        ("일시적", "영구적"), ("영구적", "일시적"),
        ("대부분", "일부"),
    )

    FABRICATIONS: Sequence[str] = (
        "또한 이 수술은 약 1시간 정도 소요됩니다.",
        "또한 이 수술의 성공률은 99% 이상입니다.",
        "또한 수술 후 3일째부터 정상 식사가 가능합니다.",
        "또한 이 시술은 건강보험이 전액 적용됩니다.",
    )

    OFF_TOPIC: Sequence[str] = (
        "오늘 서울의 최고 기온은 28도이며 오후 늦게 비가 내리겠습니다.",
        "이 제품의 보증 기간은 구매일로부터 2년이며 소모품은 제외됩니다.",
        "다음 정기 주주총회는 본사 대회의실에서 개최될 예정입니다.",
    )

    def __init__(self, doc_text: str):
        self.doc_text = doc_text

    # -- 개별 규칙 ---------------------------------------------------------

    def _term_swap(self, s: str) -> Optional[Perturbation]:
        for src, dst in self.TERM_MAP.items():
            if src in s:
                return Perturbation("TERM_SWAP", s.replace(src, dst, 1),
                                    f"{src}→{dst}", dst in self.doc_text)
        return None

    def _num_change(self, s: str) -> Optional[Perturbation]:
        m = self.NUM_PAT.search(s)
        if not m:
            return None
        old = int(m.group(1))
        new = old * 3 + 1
        rep = f"{new}{m.group(2)}"
        out = s[:m.start()] + rep + s[m.end():]
        return Perturbation("NUM_CHANGE", out,
                            f"{old}{m.group(2)}→{rep}", rep in self.doc_text)

    def _unit_change(self, s: str) -> Optional[Perturbation]:
        m = self.NUM_PAT.search(s)
        if not m or m.group(2) not in self.UNIT_MAP:
            return None
        new_unit = self.UNIT_MAP[m.group(2)]
        rep = f"{m.group(1)}{new_unit}"
        out = s[:m.start()] + rep + s[m.end():]
        return Perturbation("UNIT_CHANGE", out,
                            f"{m.group(1)}{m.group(2)}→{rep}", rep in self.doc_text)

    def _rule_swap(self, s: str, kind: str,
                   rules: Sequence[Tuple[str, str]]) -> Optional[Perturbation]:
        for src, dst in rules:
            if src in s:
                return Perturbation(kind, s.replace(src, dst, 1),
                                    f"{src}→{dst}", False)
        return None

    def _fabricate(self, s: str) -> Optional[Perturbation]:
        add = self.FABRICATIONS[abs(hash(s)) % len(self.FABRICATIONS)]
        core = add.replace("또한 ", "").rstrip(".")
        return Perturbation("FABRICATE", s.rstrip() + " " + add,
                            add, core[:10] in self.doc_text)

    def _off_topic(self, s: str) -> Optional[Perturbation]:
        add = self.OFF_TOPIC[abs(hash(s)) % len(self.OFF_TOPIC)]
        return Perturbation("OFF_TOPIC", add, "무관 문장 치환", False)

    # -- 진입점 ------------------------------------------------------------

    def make_all(self, s: str) -> List[Perturbation]:
        out: List[Perturbation] = []
        for fn in (self._term_swap, self._num_change, self._unit_change,
                   self._fabricate, self._off_topic):
            p = fn(s)
            if p and p.text.strip() != s.strip():
                out.append(p)
        for kind, rules in (("NEGATE", self.NEGATE_RULES),
                            ("SIDE_FLIP", self.SIDE_RULES),
                            ("SEVERITY_FLIP", self.SEVERITY_RULES)):
            p = self._rule_swap(s, kind, rules)
            if p and p.text.strip() != s.strip():
                out.append(p)
        return out


# ----------------------------------------------------------------------------
# 3. NLI 채점기 (SummaC 방식: 청크별 entailment 최댓값)
# ----------------------------------------------------------------------------

class NLIScorer:
    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] device={self.device}  model={MODEL_ID}", flush=True)

        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()

        # 라벨 인덱스를 하드코딩하지 않고 config에서 해석한다.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.idx_ent = next(i for i, v in id2label.items() if v.startswith("entail"))
        self.idx_con = next(i for i, v in id2label.items() if v.startswith("contra"))
        print(f"[INIT] id2label={id2label} -> entail={self.idx_ent}, contra={self.idx_con}",
              flush=True)

    def score_pairs(self, premises: List[str], hypotheses: List[str]):
        """(entailment 확률, contradiction 확률) 배열 반환."""
        torch = self.torch
        ents, cons = [], []
        bs = self.cfg.batch_size
        for i in range(0, len(premises), bs):
            enc = self.tok(premises[i:i + bs], hypotheses[i:i + bs],
                           truncation=True, padding=True,
                           max_length=self.cfg.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            ents.extend(probs[:, self.idx_ent].tolist())
            cons.extend(probs[:, self.idx_con].tolist())
        return ents, cons

    def score_hypothesis(self, hypothesis: str, chunks: List[str]):
        ents, cons = self.score_pairs(chunks, [hypothesis] * len(chunks))
        best = int(max(range(len(ents)), key=lambda j: ents[j]))
        return {
            "max_ent": float(ents[best]),
            "max_con": float(max(cons)),
            "best_chunk": chunks[best][:200],
        }


# ----------------------------------------------------------------------------
# 4. 실험 오케스트레이션
# ----------------------------------------------------------------------------

@dataclass
class Case:
    doc: str
    kind: str            # ORIGINAL 또는 오염 유형
    detail: str
    sent_id: int
    text: str
    max_ent: float = 0.0
    max_con: float = 0.0
    best_chunk: str = ""
    collides: bool = False


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cases: List[Case] = []

    # -- 데이터 로딩 --------------------------------------------------------

    def load_docs(self) -> Dict[str, str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.docs_dir, "*.txt")))
        if not paths:
            sys.exit(f"[FATAL] {self.cfg.docs_dir}/*.txt 를 찾지 못했습니다. "
                     f"--docs 로 경로를 지정하세요.")
        docs = {}
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            with open(p, encoding="utf-8", errors="replace") as f:
                docs[name] = TextNormalizer.clean(f.read())
        print(f"[LOAD] 문서 {len(docs)}건: {', '.join(docs)}", flush=True)
        return docs

    def sample_sentences(self, sents: List[str]) -> List[Tuple[int, str]]:
        import random
        rng = random.Random(self.cfg.seed)
        pool = [(i, s) for i, s in enumerate(sents)
                if self.cfg.min_sent_chars <= len(s) <= self.cfg.max_sent_chars]
        rng.shuffle(pool)
        return sorted(pool[:self.cfg.max_sents_per_doc])

    # -- 본 실행 -----------------------------------------------------------

    def run(self):
        docs = self.load_docs()
        scorer = NLIScorer(self.cfg)
        t0 = time.time()

        for di, (name, text) in enumerate(docs.items(), 1):
            sents = SentenceSplitter.split(text)
            chunks = self._build_chunks(sents)
            picked = self.sample_sentences(sents)
            perturber = Perturber(text)

            todo: List[Case] = []
            for sid, s in picked:
                todo.append(Case(name, "ORIGINAL", "-", sid, s))
                for p in perturber.make_all(s):
                    todo.append(Case(name, p.kind, p.detail, sid, p.text,
                                     collides=p.collides))

            for c in todo:
                r = scorer.score_hypothesis(c.text, chunks)
                c.max_ent, c.max_con, c.best_chunk = (
                    r["max_ent"], r["max_con"], r["best_chunk"])
            self.cases.extend(todo)

            print(f"[{di}/{len(docs)}] {name}: 문장 {len(sents)} / 청크 {len(chunks)} "
                  f"/ 케이스 {len(todo)} (누적 {len(self.cases)}, "
                  f"{time.time() - t0:.0f}s)", flush=True)

        print(f"[DONE] 총 {len(self.cases)} 케이스, {time.time() - t0:.0f}초", flush=True)

    def _build_chunks(self, sents: List[str]) -> List[str]:
        seen, chunks = set(), []
        for span in range(1, self.cfg.chunk_max_span + 1):
            for i in range(0, len(sents) - span + 1):
                c = " ".join(sents[i:i + span])
                if c not in seen:
                    seen.add(c)
                    chunks.append(c)
        return chunks or ["(빈 문서)"]

    # -- 집계 --------------------------------------------------------------

    def summarize(self) -> dict:
        cfg = self.cfg
        kinds = sorted({c.kind for c in self.cases if c.kind != "ORIGINAL"})
        originals = [c for c in self.cases if c.kind == "ORIGINAL"]

        by_kind_scores = {
            k: [c.max_ent for c in self.cases if c.kind == k] for k in kinds
        }
        orig_mean = self._mean([c.max_ent for c in originals])

        sweep = {}
        for tau in cfg.thresholds:
            spec = self._mean([1.0 if c.max_ent >= tau else 0.0 for c in originals])
            rec = {}
            for k in kinds:
                sel = [c for c in self.cases if c.kind == k]
                clean = [c for c in sel if not c.collides]
                rec[k] = {
                    "n": len(sel),
                    "detected": self._mean([1.0 if c.max_ent < tau else 0.0 for c in sel]),
                    "n_no_collision": len(clean),
                    "detected_no_collision": self._mean(
                        [1.0 if c.max_ent < tau else 0.0 for c in clean]),
                }
            real = [c for c in self.cases if c.kind not in ("ORIGINAL", "OFF_TOPIC")]
            sweep[f"{tau:.1f}"] = {
                "specificity_original_pass": spec,
                "recall_overall_excl_offtopic": self._mean(
                    [1.0 if c.max_ent < tau else 0.0 for c in real]),
                "by_kind": rec,
            }

        tau_star = None
        for tau in cfg.thresholds:
            if sweep[f"{tau:.1f}"]["specificity_original_pass"] >= cfg.target_specificity:
                tau_star = tau
                break

        return {
            "config": asdict(cfg),
            "model": MODEL_ID,
            "n_cases": len(self.cases),
            "n_original": len(originals),
            "mean_max_ent": {
                "ORIGINAL": orig_mean,
                **{k: self._mean(v) for k, v in by_kind_scores.items()},
            },
            "delta_vs_original": {
                k: self._mean(v) - orig_mean for k, v in by_kind_scores.items()
            },
            "threshold_sweep": sweep,
            "tau_star": tau_star,
            "tau_star_summary": sweep[f"{tau_star:.1f}"] if tau_star else None,
        }

    @staticmethod
    def _mean(xs: List[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else float("nan")

    # -- 저장 / 출력 --------------------------------------------------------

    def save(self, summary: dict):
        p = self.cfg.out_prefix
        with open(f"{p}_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(f"{p}_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc", "sent_id", "kind", "detail", "collides",
                        "max_ent", "max_con", "text", "best_chunk"])
            for c in sorted(self.cases, key=lambda x: (x.doc, x.sent_id, x.kind)):
                w.writerow([c.doc, c.sent_id, c.kind, c.detail, int(c.collides),
                            round(c.max_ent, 4), round(c.max_con, 4),
                            c.text, c.best_chunk])
        print(f"\n[SAVE] {p}_report.json / {p}_cases.csv", flush=True)

    def report(self, s: dict):
        print("\n" + "=" * 74)
        print("NLI 민감도 실험 결과")
        print("=" * 74)
        print(f"총 케이스 {s['n_cases']}건 (정상 원문 {s['n_original']}건)\n")

        print("[1] 유형별 평균 entailment 최댓값 (정상 대비 하락폭)")
        base = s["mean_max_ent"]["ORIGINAL"]
        print(f"  {'ORIGINAL':<16} {base:.4f}   (기준)")
        for k, v in sorted(s["delta_vs_original"].items(), key=lambda kv: kv[1]):
            print(f"  {k:<16} {s['mean_max_ent'][k]:.4f}   {v:+.4f}")

        print("\n[2] 임계값 스윕  (정상 통과율 / 오염 탐지율, OFF_TOPIC 제외)")
        for tau, d in s["threshold_sweep"].items():
            print(f"  tau={tau}   정상통과 {d['specificity_original_pass']:.1%}"
                  f"   탐지 {d['recall_overall_excl_offtopic']:.1%}")

        if s["tau_star"] is not None:
            d = s["tau_star_summary"]
            print(f"\n[3] 운영 임계값 tau*={s['tau_star']:.1f} "
                  f"(정상 통과율 {d['specificity_original_pass']:.1%} 이상 만족)")
            print(f"    {'유형':<16}{'n':>5}{'탐지율':>10}{'충돌제외':>12}")
            for k, v in sorted(d["by_kind"].items(),
                               key=lambda kv: -kv[1]["detected"]):
                print(f"    {k:<16}{v['n']:>5}{v['detected']:>9.1%}"
                      f"{v['detected_no_collision']:>11.1%}")
            off = d["by_kind"].get("OFF_TOPIC", {}).get("detected", 1.0)
            if off < 0.90:
                print("\n  [!] 경고: OFF_TOPIC 탐지율이 90% 미만입니다. "
                      "지표가 아니라 채점 코드/청크 구성을 먼저 의심하세요.")
        else:
            print("\n[3] 목표 정상 통과율을 만족하는 임계값이 없습니다 "
                  "— 청크 구성 또는 문장 분리를 점검하세요.")
        print("=" * 74)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out-prefix", default="nli_sensitivity")
    ap.add_argument("--max-sents", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    cfg = Config(docs_dir=a.docs, out_prefix=a.out_prefix,
                 max_sents_per_doc=a.max_sents, batch_size=a.batch_size)
    exp = Experiment(cfg)
    exp.run()
    summary = exp.summarize()
    exp.report(summary)
    exp.save(summary)


if __name__ == "__main__":
    main()
