#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_medllama.py — Medical-Llama3 평이화 실험
=============================================
연구 질문:
    1) HARI 에서 나타난 정보보존-평이화 trade-off 가 다른 8B 의료 LLM 에서도
       반복되는가?
    2) 한국어에서 실패한다면 그것이 모델 자체의 문제인가, 한국어 입력 조건의
       문제인가?

두 실험:
    --lang ko   K → Medical-Llama3 한국어 평이화        HARI 와 직접 비교
    --lang en   E0(기존 영어 번역) → Medical-Llama3 영어 평이화
                MedGemma 영어 피벗 실험과 동일 구조·동일 harness

알려진 제약 — 반드시 결과와 함께 보고할 것:
    이 모델은 tokenizer.chat_template 이 배포되지 않았다.
        ValueError: Cannot use chat template functions because
        tokenizer.chat_template is not set
    즉 "이 모델에 지시를 어떤 형식으로 넣어야 하는가"가 모델과 함께
    제공되지 않았다. 베이스 모델인 Llama-3 의 표준 형식을 적용하며,
    이 선택 자체가 실험 조건의 일부임을 명시한다.

    사전 점검(3문서)에서 출력 첫 줄이 프롬프트 문장 그 자체였고 이어서
    원문을 복사했다. 한국어·영어 프롬프트 모두 동일했다. 본 실험은 그
    관찰을 13문서로 정량화하는 것이 목적이다.

기존 자산 재사용:
    K       : prompt_ladder.extract_sections  (HARI baseline 과 동일한 전처리)
    프롬프트 : prompt_ladder.PROMPTS['P0'], SYS_PROMPT
    정제    : prompt_ladder.Cleaner
    E0      : outputs_medgemma_pivot/{doc}__E0.json  (읽기만)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python experiment_medllama/run_medllama.py --lang ko --docs doc7 doc12 doc11
    python experiment_medllama/run_medllama.py --lang ko
    python experiment_medllama/run_medllama.py --lang en
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

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, Cleaner, PROMPTS, SYS_PROMPT
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py 가 상위 폴더에 있어야 합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
PIVOTDIR = os.path.join(HOME, "outputs_medgemma_pivot")
OUTDIR = os.path.join(HERE, "outputs")

MODEL = "ruslanmv/Medical-Llama3-8B"
PROMPT_VERSION = "medllama-v1"

# 영어 평이화 프롬프트 — MedGemma 영어 피벗 실험과 동일 문구
EN_SYS = "You are a helpful medical assistant."
EN_PROMPT = (
    "Rewrite the following surgical consent form in plain English that a "
    "patient without medical training can understand.\n"
    "Keep every number, duration, percentage, condition, and the name of the "
    "surgery exactly as in the original. Do not add any information that is "
    "not in the original. Do not omit any item.\n"
    "Output only the rewritten text.\n\n{text}")

LLAMA3_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n")


class Gen:
    def __init__(self, gpu: int, cap: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.cap = cap
        self.template_source = None
        print(f"[GEN] {MODEL} → cuda:{gpu}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map={"": gpu})
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map={"": gpu})
        self.model.eval()
        has = getattr(self.tok, "chat_template", None)
        print(f"[GEN] chat_template: {'있음' if has else '없음 — Llama-3 표준 형식 적용'}")

    def run(self, system: str, user: str) -> str:
        torch = self.torch
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            t = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            self.template_source = "model"
        except (ValueError, AttributeError):
            t = LLAMA3_TEMPLATE.format(system=system, user=user)
            self.template_source = "llama3-standard-fallback"
        inp = self.tok(t, return_tensors="pt").to(self.model.device)
        n = inp.input_ids.shape[1]
        with torch.no_grad():
            o = self.model.generate(**inp, max_new_tokens=self.cap,
                                    do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0][n:], skip_special_tokens=True).strip()

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, f"{doc}.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_E0(doc: str) -> Optional[str]:
    p = os.path.join(PIVOTDIR, f"{doc}__E0.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")

    stage = "KO" if a.lang == "ko" else "EN"
    outdir = os.path.join(OUTDIR, a.lang)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 82)
    print(f"Medical-Llama3 평이화 실험 — {stage}")
    print("=" * 82)
    print(f"  모델: {MODEL}")
    print(f"  문서 {len(docs)}건")
    if a.lang == "ko":
        print(f"  입력: extract_sections(docs/*.txt)  ← HARI baseline 과 동일")
        print(f"  프롬프트: P0 (HARI baseline 과 동일 문자열)")
    else:
        print(f"  입력: outputs_medgemma_pivot/*__E0.json  ← MedGemma 실험과 동일")
        print(f"  프롬프트: MedGemma 영어 피벗과 동일 문구")
    print(f"  출력: {outdir}/")

    todo = [d for d in docs
            if a.force or not os.path.exists(os.path.join(outdir, f"{d}.json"))]
    if not todo:
        print("\n생성할 것이 없습니다 (--force 로 재생성)")
        return

    gen = Gen(a.gpu, a.max_new)
    t0 = time.time()
    for n, doc in enumerate(todo, 1):
        if a.lang == "ko":
            src = load_K(doc, a.docdir)
            system, prompt = SYS_PROMPT, PROMPTS["P0"] + "\n\n" + (src or "")
        else:
            src = load_E0(doc)
            system, prompt = EN_SYS, EN_PROMPT.format(text=src or "")
        if not src:
            print(f"  [{doc}] 입력 없음 — 건너뜀")
            continue

        t1 = time.time()
        raw = gen.run(system, prompt)
        rt = time.time() - t1
        cleaned, removed = Cleaner.clean(raw)

        ko = len(re.findall(r"[가-힣]", cleaned))
        en = len(re.findall(r"[A-Za-z]", cleaned))
        with open(os.path.join(outdir, f"{doc}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "doc": doc, "stage": stage, "lang": a.lang,
                "model": MODEL, "model_revision": "main",
                "prompt_version": PROMPT_VERSION,
                "chat_template_source": gen.template_source,
                "system": system,
                "prompt": PROMPTS["P0"] if a.lang == "ko" else EN_PROMPT,
                "gen_config": {"do_sample": False,
                               "max_new_tokens": a.max_new},
                "input_language": "ko" if a.lang == "ko" else "en",
                "output_language_expected": "ko" if a.lang == "ko" else "en",
                "route": "medllama-ko" if a.lang == "ko" else "medllama-en-pivot",
                "upstream": ("docs/%s.txt" % doc if a.lang == "ko"
                             else "outputs_medgemma_pivot/%s__E0.json" % doc),
                "src": src, "out": cleaned, "out_raw": raw,
                "meta_removed": removed,
                "runtime_sec": round(rt, 1),
            }, f, ensure_ascii=False, indent=2)

        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {doc}: {len(src)}자 → {len(cleaned)}자 "
              f"({100*(len(cleaned)-len(src))/max(len(src),1):+.0f}%) "
              f"한글 {ko} / 영문 {en}  "
              f"[{el:.0f}s, 남은 {el/n*(len(todo)-n):.0f}s]", flush=True)
    gen.free()

    print("\n" + "=" * 82)
    print("생성 완료. 평가는 evaluate_medllama.py 로 진행하세요.")
    print(f"  python experiment_medllama/evaluate_medllama.py --lang {a.lang}")
    print("=" * 82)


if __name__ == "__main__":
    main()
