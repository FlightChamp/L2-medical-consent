#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medgemma_en_pivot.py — 영어 피벗 실험 생성 단계
================================================
가설 (전제가 아니라 검증 대상):
    MedGemma 1.5 4B 는 한국어 직접 입력에서 Grounding 75.6 / Copy 0.65 로
    HARI(83.6 / 0.30)보다 낮았다. 그러나 이 모델의 의료 특화 학습과 공식 평가는
    영어 환경 기준이다. 영어로 입력하면 결과가 달라질 수 있다.

세 경로 (모두 같은 K 를 출발점으로 한다):
    Route A   K → HARI 한국어 평이화 → 영어 번역        [기존 파이프라인]
    Route B   K → 영어 번역(E0) → MedGemma 영어 평이화(E1)  [실험]
    Route C   K → 영어 번역(E0)                          [대조군]

    C 가 있어야 B 의 개선이 MedGemma 때문인지 영어 피벗 때문인지 분리된다.

K 의 정의 — 중요:
    기존 outputs_translate/ 의 영어 번역은 report*.json 의 '원문' 필드를 소스로 썼고,
    그 길이가 extract_sections() 결과와 다르다 (doc1: 1625 vs 1380).
    세 경로를 같은 기준으로 비교하려면 K 가 하나여야 하므로,
    실제 파이프라인과 같은 extract_sections() 기준으로 통일하고 E0 를 새로 만든다.

생성물 (전부 outputs_medgemma_pivot/ 에 저장, 기존 결과는 건드리지 않는다):
    {doc}__E0.json     K → 영어 번역        (HARI)      = Route C 산출물
    {doc}__E1.json     E0 → 영어 평이화     (MedGemma)  = Route B 산출물
    {doc}__A_en.json   HARI 변환문 → 영어   (HARI)      = Route A 산출물

사용법:
    cd ~/이윤우 && source .venv/bin/activate

    # smoke test — 짧은/중간/장문압축 3건
    python medgemma_en_pivot.py --docs doc7 doc12 doc11

    # 전체
    python medgemma_en_pivot.py

    # 특정 단계만
    python medgemma_en_pivot.py --stages e0
    python medgemma_en_pivot.py --stages e1
    python medgemma_en_pivot.py --stages a_en
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from prompt_ladder import extract_sections, Cleaner
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py 가 같은 폴더에 있어야 합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
SIMPDIR = os.path.join(HOME, "outputs")          # HARI 한국어 평이화 결과 (읽기만)
OUTDIR = os.path.join(HOME, "outputs_medgemma_pivot")

HARI = "snuh/hari-q3-8b"
MEDGEMMA = "google/medgemma-1.5-4b-it"

# ── 프롬프트 (버전 고정, 결과 파일에 함께 저장한다) ───────────────────
PROMPT_VERSION = "pivot-v1"

TRANS_SYS = "You are a careful medical translator."
TRANS_PROMPT = (
    "Translate the following Korean surgical consent form into English. "
    "Translate every sentence faithfully. Do not add explanations.\n\n{text}")

SIMP_SYS = "You are a helpful medical assistant."
SIMP_PROMPT = (
    "Rewrite the following surgical consent form in plain English that a "
    "patient without medical training can understand.\n"
    "Keep every number, duration, percentage, condition, and the name of the "
    "surgery exactly as in the original. Do not add any information that is "
    "not in the original. Do not omit any item.\n"
    "Output only the rewritten text.\n\n{text}")

# 생성 설정 — 결과 파일에 기록한다
GEN_CONFIG = {
    "do_sample": False,
    "repetition_penalty": None,      # 아래에서 모델별로 결정
    "no_repeat_ngram_size": None,
    "max_new_tokens": 4096,
}


# ===========================================================================

class Gen:
    """causal(HARI) / imagetext(MedGemma) 두 아키텍처 지원."""

    def __init__(self, model_id: str, arch: str, gpu: int, cap: int,
                 rep_penalty: Optional[float], no_repeat: Optional[int]):
        import torch
        self.torch = torch
        self.arch = arch
        self.cap = cap
        self.model_id = model_id
        self.rep = rep_penalty
        self.nrg = no_repeat
        print(f"[GEN] {model_id} ({arch}) → cuda:{gpu} "
              f"rep={rep_penalty} nrg={no_repeat}", flush=True)

        if arch == "causal":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_id)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.bfloat16, device_map={"": gpu})
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, device_map={"": gpu})
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.tok = AutoProcessor.from_pretrained(model_id)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, dtype=torch.bfloat16, device_map={"": gpu})
            except TypeError:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, device_map={"": gpu})
        self.model.eval()

    def _kw(self) -> dict:
        kw = {"do_sample": False, "max_new_tokens": self.cap}
        if self.rep is not None:
            kw["repetition_penalty"] = self.rep
        if self.nrg is not None:
            kw["no_repeat_ngram_size"] = self.nrg
        return kw

    def run(self, system: str, user: str) -> str:
        torch = self.torch
        if self.arch == "causal":
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
            try:
                t = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                t = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            inp = self.tok(t, return_tensors="pt").to(self.model.device)
            n = inp.input_ids.shape[1]
            with torch.no_grad():
                o = self.model.generate(**inp, pad_token_id=self.tok.eos_token_id,
                                        **self._kw())
            raw = self.tok.decode(o[0][n:], skip_special_tokens=True).strip()
        else:
            msgs = [{"role": "system",
                     "content": [{"type": "text", "text": system}]},
                    {"role": "user",
                     "content": [{"type": "text", "text": user}]}]
            inp = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            n = inp["input_ids"].shape[-1]
            with torch.no_grad():
                o = self.model.generate(**inp, **self._kw())
            raw = self.tok.decode(o[0][n:], skip_special_tokens=True).strip()

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        if "</think>" in raw:
            raw = raw.split("</think>")[-1]
        return raw.strip()

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


# ===========================================================================

def load_K(doc: str, docdir: str) -> Optional[str]:
    """실제 파이프라인과 동일한 전처리로 한국어 원문을 만든다."""
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_hari_simplified(doc: str) -> Optional[str]:
    """Route A 용 — 기존 HARI 한국어 평이화 결과를 읽기만 한다."""
    p = os.path.join(SIMPDIR, f"{doc}__whole.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    return " ".join(s["out"] for s in r["sections"])


def save(doc: str, stage: str, payload: dict):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{doc}__{stage}.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def exists(doc: str, stage: str) -> bool:
    return os.path.exists(os.path.join(OUTDIR, f"{doc}__{stage}.json"))


def load_stage(doc: str, stage: str) -> Optional[dict]:
    p = os.path.join(OUTDIR, f"{doc}__{stage}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stat_line(tag: str, src: str, out: str) -> str:
    ko = len(re.findall(r"[가-힣]", out))
    en = len(re.findall(r"[A-Za-z]", out))
    return (f"{tag}: {len(src)}자 → {len(out)}자 "
            f"({100*(len(out)-len(src))/max(len(src),1):+.0f}%) "
            f"한글 {ko} / 영문 {en}")


# ===========================================================================

def run_e0(docs: List[str], a):
    """K → 영어 번역 (HARI). Route C 산출물이자 Route B 의 중간 단계."""
    todo = [d for d in docs if a.force or not exists(d, "E0")]
    if not todo:
        print("[E0] 생성할 것 없음")
        return
    print(f"\n=== [E0] K → 영어 번역  ({len(todo)}건, {HARI})")
    g = Gen(HARI, "causal", a.gpu, a.max_new, None, None)
    t0 = time.time()
    for n, doc in enumerate(todo, 1):
        K = load_K(doc, a.docdir)
        if not K:
            print(f"  [{doc}] K 생성 실패 — 건너뜀")
            continue
        out = g.run(TRANS_SYS, TRANS_PROMPT.format(text=K))
        save(doc, "E0", {
            "doc": doc, "stage": "E0", "route": "B,C",
            "model": HARI, "arch": "causal",
            "prompt_version": PROMPT_VERSION,
            "system": TRANS_SYS, "prompt": TRANS_PROMPT,
            "gen_config": {"do_sample": False, "max_new_tokens": a.max_new},
            "K_source": "extract_sections(docs/*.txt)",
            "src": K, "out": out,
        })
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {stat_line(doc, K, out)} "
              f"[{el:.0f}s, 남은 {el/n*(len(todo)-n):.0f}s]", flush=True)
    g.free()


def run_e1(docs: List[str], a):
    """E0 → 영어 평이화 (MedGemma). Route B 산출물."""
    todo = [d for d in docs
            if exists(d, "E0") and (a.force or not exists(d, "E1"))]
    if not todo:
        print("[E1] 생성할 것 없음 (E0 가 먼저 필요합니다)")
        return
    print(f"\n=== [E1] E0 → 영어 평이화  ({len(todo)}건, {MEDGEMMA})")
    print("    반복 억제 적용 — 한국어 입력에서 루프가 확인되었기 때문")
    g = Gen(MEDGEMMA, "imagetext", a.gpu, a.max_new,
            a.rep_penalty, a.no_repeat)
    t0 = time.time()
    for n, doc in enumerate(todo, 1):
        e0 = load_stage(doc, "E0")
        E0 = e0["out"]
        out = g.run(SIMP_SYS, SIMP_PROMPT.format(text=E0))
        cleaned, removed = Cleaner.clean(out)
        save(doc, "E1", {
            "doc": doc, "stage": "E1", "route": "B",
            "model": MEDGEMMA, "arch": "imagetext",
            "prompt_version": PROMPT_VERSION,
            "system": SIMP_SYS, "prompt": SIMP_PROMPT,
            "gen_config": {"do_sample": False, "max_new_tokens": a.max_new,
                           "repetition_penalty": a.rep_penalty,
                           "no_repeat_ngram_size": a.no_repeat},
            "upstream": f"{doc}__E0.json",
            "src": E0, "out": cleaned, "out_raw": out,
            "meta_removed": removed,
        })
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {stat_line(doc, E0, cleaned)} "
              f"[{el:.0f}s, 남은 {el/n*(len(todo)-n):.0f}s]", flush=True)
    g.free()


def run_a_en(docs: List[str], a):
    """HARI 한국어 평이화문 → 영어 번역. Route A 산출물."""
    todo = [d for d in docs if a.force or not exists(d, "A_en")]
    if not todo:
        print("[A_en] 생성할 것 없음")
        return
    print(f"\n=== [A_en] HARI 평이화문 → 영어 번역  ({len(todo)}건, {HARI})")
    g = Gen(HARI, "causal", a.gpu, a.max_new, None, None)
    t0 = time.time()
    for n, doc in enumerate(todo, 1):
        S = load_hari_simplified(doc)
        if not S:
            print(f"  [{doc}] outputs/{doc}__whole.json 없음 — 건너뜀")
            continue
        out = g.run(TRANS_SYS, TRANS_PROMPT.format(text=S))
        save(doc, "A_en", {
            "doc": doc, "stage": "A_en", "route": "A",
            "model": HARI, "arch": "causal",
            "prompt_version": PROMPT_VERSION,
            "system": TRANS_SYS, "prompt": TRANS_PROMPT,
            "gen_config": {"do_sample": False, "max_new_tokens": a.max_new},
            "upstream": f"outputs/{doc}__whole.json (HARI 한국어 평이화)",
            "src": S, "out": out,
        })
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {stat_line(doc, S, out)} "
              f"[{el:.0f}s, 남은 {el/n*(len(todo)-n):.0f}s]", flush=True)
    g.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--stages", nargs="+", default=["e0", "e1", "a_en"],
                    choices=["e0", "e1", "a_en"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--rep-penalty", type=float, default=1.1,
                    help="MedGemma 전용. 한국어 입력에서 반복 루프가 확인됨")
    ap.add_argument("--no-repeat", type=int, default=12,
                    help="MedGemma 전용")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")

    print("=" * 78)
    print("영어 피벗 실험 — 생성 단계")
    print("=" * 78)
    print(f"  문서 {len(docs)}건: {', '.join(docs)}")
    print(f"  단계: {', '.join(a.stages)}")
    print(f"  K 기준: extract_sections(docs/*.txt)  ← 실제 파이프라인과 동일")
    print(f"  출력: {OUTDIR}/")
    print(f"  프롬프트 버전: {PROMPT_VERSION}")
    print("  기존 결과(outputs/, outputs_translate/)는 읽기만 합니다")

    os.makedirs(OUTDIR, exist_ok=True)
    if "e0" in a.stages:
        run_e0(docs, a)
    if "e1" in a.stages:
        run_e1(docs, a)
    if "a_en" in a.stages:
        run_a_en(docs, a)

    print("\n" + "=" * 78)
    print("생성 완료. 검증은 compare_routes.py 로 진행하세요.")
    print(f"  python compare_routes.py --docs {' '.join(docs)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
