#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_compare.py — ③ 평이화 단계 모델 비교 하네스
===================================================
목적:
    같은 프롬프트, 같은 13문서, 같은 지표로 여러 모델을 재서 직접 비교한다.

기준선 (HARI 8B, prompt_ladder.py P0 조건에서 측정):
    근거율 83.64 / 커버리지 63.20 / 용어보존 39.62
    복사유사도 0.30 / 수치환각 2건 / 분량 -10.35%

지표 계산의 동일성 보장:
    지표 함수를 새로 짜지 않고 prompt_ladder.py 에서 **직접 import** 한다.
    따라서 기준선과 계산 방식이 한 글자도 다르지 않다.
    (prompt_ladder.py 가 같은 폴더에 있어야 한다)

아키텍처 차이 처리:
    HARI/Qwen3    AutoModelForCausalLM      + AutoTokenizer
    MedGemma      AutoModelForImageTextToText + AutoProcessor
                  메시지 content 가 리스트 형식, enable_thinking 없음,
                  출력 한도 8192 토큰

사용법:
    cd ~/이윤우 && source .venv/bin/activate

    # MedGemma 1.5 4B (멀티모달이지만 텍스트만 넣는다)
    python model_compare.py --model google/medgemma-1.5-4b-it --arch imagetext --docs doc1
    python model_compare.py --model google/medgemma-1.5-4b-it --arch imagetext

    # 기준선 재현 확인 (기존 결과와 일치해야 함)
    python model_compare.py --model snuh/hari-q3-8b --arch causal

    # 생성 없이 채점만
    python model_compare.py --model google/medgemma-1.5-4b-it --arch imagetext --eval-only

주의 — MedGemma 는 Hugging Face 에서 약관 동의가 필요합니다:
    1) https://huggingface.co/google/medgemma-1.5-4b-it 접속해 약관 동의
    2) 서버에서  huggingface-cli login   (토큰 입력)
    동의 없이 받으면 401/403 오류가 납니다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── 지표를 기준선 스크립트에서 그대로 가져온다 ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from prompt_ladder import (
        SYS_PROMPT, PROMPTS, extract_sections, Splitter, build_chunks,
        jaccard3, extract_terms, numeric_units, numeric_halluc,
        readability, Cleaner, NLI, BLANK_MARK, ORIG_BLANK,
    )
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py 를 같은 폴더에 두세요: {e}")

DOCDIR = os.path.expanduser("~/이윤우/docs")
OUTROOT = os.path.expanduser("~/이윤우/outputs_compare")

BASELINE = {          # HARI 8B, P0, 13문서 (비교 표시용)
    "name": "HARI 8B",
    "delta": -10.35, "term_keep": 39.62, "num_keep": 99.45,
    "n_halluc_total": 2, "ground": 83.64, "cover": 63.20, "copy_sim": 0.30,
}


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


# ===========================================================================
# 생성기 — 두 아키텍처를 모두 지원
# ===========================================================================

class Generator:
    def __init__(self, model_id: str, arch: str, gpu: int, cap: int,
                 extra: dict = None):
        import torch
        self.torch = torch
        self.arch = arch
        self.cap = cap
        self.model_id = model_id
        # 생성 설정. 기본은 그리디(빈 dict).
        # 반복 억제는 모델마다 반대 방향으로 작용하므로 명시할 때만 적용한다.
        self.extra = extra or {}
        print(f"[GEN] {model_id}  ({arch})  → cuda:{gpu}", flush=True)

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

    def _messages(self, user: str) -> List[dict]:
        if self.arch == "causal":
            return [{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user}]
        # Gemma 3 / MedGemma 형식 — content 가 블록 리스트
        return [
            {"role": "system",
             "content": [{"type": "text", "text": SYS_PROMPT}]},
            {"role": "user",
             "content": [{"type": "text", "text": user}]},
        ]

    def run(self, user_prompt: str, text: str) -> str:
        torch = self.torch
        msgs = self._messages(f"{user_prompt}\n\n{text}")

        if self.arch == "causal":
            try:
                t = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                t = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            inp = self.tok(t, return_tensors="pt").to(self.model.device)
            n_in = inp.input_ids.shape[1]
            with torch.no_grad():
                o = self.model.generate(
                    **inp, max_new_tokens=self.cap, do_sample=False,
                    pad_token_id=self.tok.eos_token_id, **self.extra)
            return self.tok.decode(o[0][n_in:], skip_special_tokens=True).strip()

        # image-text-to-text
        inp = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        n_in = inp["input_ids"].shape[-1]
        with torch.no_grad():
            o = self.model.generate(**inp, max_new_tokens=self.cap,
                                    do_sample=False, **self.extra)
        return self.tok.decode(o[0][n_in:], skip_special_tokens=True).strip()

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


# ===========================================================================

def path_of(outdir: str, doc: str) -> str:
    return os.path.join(outdir, f"{doc}.json")


def stage_generate(docs: List[str], outdir: str, a):
    todo = [d for d in docs
            if a.force or not os.path.exists(path_of(outdir, d))]
    if not todo:
        print("[GEN] 생성할 것이 없습니다 (--force 로 재생성)")
        return
    extra = {}
    if a.rep_penalty is not None:
        extra["repetition_penalty"] = a.rep_penalty
    if a.no_repeat is not None:
        extra["no_repeat_ngram_size"] = a.no_repeat
    if extra:
        print(f"[GEN] 생성 설정: {extra}")
    else:
        print("[GEN] 생성 설정: greedy (do_sample=False)")
    gen = Generator(a.model, a.arch, a.gpu, a.max_new, extra)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    for n, doc in enumerate(todo, 1):
        with open(os.path.join(a.docdir, f"{doc}.txt"),
                  encoding="utf-8", errors="replace") as f:
            secs = extract_sections(f.read())
        if not secs:
            print(f"  [{doc}] 섹션 추출 실패 — 건너뜀")
            continue
        body = " ".join(b for _, b in secs)
        raw = gen.run(PROMPTS[a.cond], body)
        out, removed = Cleaner.clean(raw)
        with open(path_of(outdir, doc), "w", encoding="utf-8") as f:
            json.dump({"doc": doc, "model": a.model, "arch": a.arch,
                       "cond": a.cond, "prompt": PROMPTS[a.cond],
                       "gen_config": {"do_sample": False,
                                      "max_new_tokens": a.max_new,
                                      **(gen.extra or {})},
                       "src": body, "out": out, "out_raw": raw,
                       "meta_removed": removed},
                      f, ensure_ascii=False, indent=2)
        el = time.time() - t0
        ko = len(re.findall(r"[가-힣]", out))
        print(f"  [{n}/{len(todo)}] {doc}: {len(body)}자 → {len(out)}자 "
              f"({100*(len(out)-len(body))/max(len(body),1):+.0f}%) "
              f"한글 {ko}자  [{el:.0f}s, 남은 예상 "
              f"{el/n*(len(todo)-n):.0f}s]", flush=True)
    gen.free()


def stage_eval(docs: List[str], outdir: str, a) -> List[dict]:
    nli = NLI(a.gpu, a.batch)
    rows = []
    print(f"\n  {'문서':<7}{'분량':>8}{'용어':>7}{'수치':>7}{'환각':>6}"
          f"{'근거':>7}{'커버':>7}{'복사':>7}{'한글비':>8}")
    for doc in docs:
        p = path_of(outdir, doc)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        src, out = r["src"], r["out"]
        terms = extract_terms(src)
        nums = sorted(set(numeric_units(src)))
        ss, oo = Splitter.split(src), Splitter.split(out)
        ro = readability(out)
        ko = len(re.findall(r"[가-힣]", out))
        row = {
            "doc": doc, "model": a.model,
            "src_chars": len(src), "out_chars": len(out),
            "delta": 100 * (len(out) - len(src)) / max(len(src), 1),
            "term_keep": round(100 * sum(1 for t in terms if t in out)
                               / max(len(terms), 1), 1),
            "num_keep": round(100 * sum(1 for x in nums if x in out)
                              / len(nums), 1) if nums else None,
            "n_halluc": len(numeric_halluc(src, out)),
            "ground": round(nli.rate(oo, build_chunks(ss), a.k, a.tau), 1),
            "cover": round(nli.rate(ss, build_chunks(oo), a.k, a.tau), 1),
            "copy_sim": round(jaccard3(src, out), 3),
            "ko_ratio": round(100 * ko / max(len(out), 1), 1),
            "blank_mark": len(BLANK_MARK.findall(out)),
            "out_term": round(ro["term"], 2),
            "out_hanja": round(ro["hanja"], 2),
            "out_sent_len": round(ro["sent_len"], 2),
        }
        rows.append(row)
        nk = f"{row['num_keep']:.1f}" if row["num_keep"] is not None else "—"
        print(f"  {doc:<7}{row['delta']:>+7.0f}%{row['term_keep']:>7.1f}"
              f"{nk:>7}{row['n_halluc']:>6}{row['ground']:>7.1f}"
              f"{row['cover']:>7.1f}{row['copy_sim']:>7.2f}"
              f"{row['ko_ratio']:>7.1f}%", flush=True)
    return rows


def report(rows: List[dict], a):
    if not rows:
        print("결과 없음")
        return

    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return st.fmean(v) if v else float("nan")

    tot_hal = sum(r["n_halluc"] for r in rows)
    docs_hal = sum(1 for r in rows if r["n_halluc"] > 0)

    print("\n" + "=" * 88)
    print(f"모델 비교 — {a.model}  (조건 {a.cond}, 문서 {len(rows)}건)")
    print("=" * 88)
    print(f"  {'지표':<20}{BASELINE['name']:>12}{'이 모델':>12}{'차이':>12}")
    pairs = [
        ("근거율 (Grounding)", "ground", BASELINE["ground"]),
        ("커버리지 (Coverage)", "cover", BASELINE["cover"]),
        ("용어보존 상위20", "term_keep", BASELINE["term_keep"]),
        ("복사유사도", "copy_sim", BASELINE["copy_sim"]),
        ("분량 증감%", "delta", BASELINE["delta"]),
        ("수치보존", "num_keep", BASELINE["num_keep"]),
    ]
    for label, key, base in pairs:
        v = avg(key)
        print(f"  {label:<20}{base:>12.2f}{v:>12.2f}{v-base:>+12.2f}")
    print(f"  {'수치환각 (총 건수)':<20}{BASELINE['n_halluc_total']:>12}"
          f"{tot_hal:>12}{tot_hal-BASELINE['n_halluc_total']:>+12}")
    print(f"  {'  └ 발생 문서 수':<20}{'2':>12}{docs_hal:>12}")

    print(f"\n  {'참고 지표':<20}{'이 모델':>12}")
    for label, key in [("한글 비율 %", "ko_ratio"),
                       ("[미기재] 표시 수", "blank_mark"),
                       ("전문용어 비율", "out_term"),
                       ("한자어 비율", "out_hanja"),
                       ("평균 어절", "out_sent_len")]:
        print(f"  {label:<20}{avg(key):>12.2f}")

    # -- 사전 판정 (prompt_ladder 와 동일 기준) ---------------------------
    print("\n" + "=" * 88)
    print("판정 — 사전에 정한 기준")
    print("=" * 88)
    fails = []
    cs, kor = avg("copy_sim"), avg("ko_ratio")
    g = avg("ground")
    if cs >= 0.6:
        fails.append(f"복사유사도 {cs:.2f} ≥ 0.6 — 변환하지 않고 복사")
    if kor < 40:
        fails.append(f"한글 비율 {kor:.1f}% < 40% — 한국어로 답하지 않음")
    if g < BASELINE["ground"] - 10:
        fails.append(f"근거율 {g:.1f} — 기준선 {BASELINE['ground']}보다 10 이상 낮음")
    if not fails:
        print("  기준 통과. 기준선과 지표별로 비교 가능합니다.")
    else:
        for f in fails:
            print(f"  기각 사유: {f}")
    print("""
  해석 주의
    · 용어보존이 높아도 복사유사도가 함께 높으면 '변환하지 않은 것'입니다.
      HARI 프롬프트 실험에서 용어보존 39.6→86.5 상승이 복사(0.30→0.73) 때문이었습니다.
    · 한글 비율은 모델이 한국어로 답하는지 보는 장치입니다.
      Medical-Llama3 는 지시를 따르지 않고 원문을 복사했습니다.""")
    print("=" * 88)

    out = f"model_compare_{slug(a.model)}_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "arch": a.arch, "cond": a.cond,
                   "baseline": BASELINE, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", choices=["causal", "imagetext"], default="causal")
    ap.add_argument("--cond", default="P0", choices=list(PROMPTS))
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=4096,
                    help="MedGemma 출력 한도는 8192 입니다")
    ap.add_argument("--rep-penalty", type=float, default=None,
                    help="지정할 때만 적용. 기본은 그리디")
    ap.add_argument("--no-repeat", type=int, default=None,
                    help="지정할 때만 적용. 기본은 그리디")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")

    outdir = os.path.join(OUTROOT, slug(a.model))
    print(f"[PLAN] 모델 {a.model} / 조건 {a.cond} / 문서 {len(docs)}건")
    print(f"[PLAN] 프롬프트: {PROMPTS[a.cond][:70]}...")
    print(f"[PLAN] 출력 폴더: {outdir}")

    os.makedirs(outdir, exist_ok=True)
    if not a.eval_only:
        stage_generate(docs, outdir, a)
    rows = stage_eval(docs, outdir, a)
    report(rows, a)


if __name__ == "__main__":
    main()
