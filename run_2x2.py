#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_2x2.py — P0/P4 × Protected ON/OFF 생성
============================================
네 조건:
    A  P0 / Protected OFF   기존 outputs_prompt/{doc}__P0.json 재사용
    B  P0 / Protected ON
    C  P4 / Protected OFF
    D  P4 / Protected ON

A 는 이미 기준선 재현이 검증되었으므로 재생성하지 않는다
(model_compare.py 로 Grounding 83.64 / 용어 39.62 / Copy 0.30 소수점 일치 확인).

Protected ON 흐름:
    K ──mask──> 마스킹문 ──HARI 생성──> 출력 ──bijection audit──> 복원 ──> 최종
                                                    │ 실패
                                                    └──> fail-closed: 원문 K 사용
    실패 시 평이화를 포기하고 원문을 그대로 둔다. 일부만 복원하는 것은
    원문보다 위험하므로 허용하지 않는다. 실패 건수는 결과에 기록한다.

P4 프롬프트 근거:
    Phase 1 에서 HARI 의 L2 이독성 개선을 분해한 결과
        전체 이득 +15.42
          문장 길이에서 +14.20  (92%)
          어휘에서      +1.21   ( 8%)
    즉 문장은 충분히 짧아졌으나(22.59 → 9.00 어절) 어휘 난이도는 거의
    그대로다(중급이상 70.26% → 65.63%). P4 는 남은 축인 어휘를 겨냥한다.

    목표 수준 표기는 고승연(2025) 환산표에 근거한다.
        >= 45 쉬움 (대부분의 중급 학습자가 이해 가능) / 35-44.99 보통 / < 35 어려움
        논문 준거 지문 평균 문장당 10.58 어절, TOPIK I 지문 5.3~10 어절
    근거 없는 "TOPIK 2급 수준" 같은 표현은 넣지 않는다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate

    # 1단계 — B 만 먼저 (Protected 효과 확인)
    python run_2x2.py --conds B

    # smoke test (3문서)
    python run_2x2.py --conds B --docs doc7 doc12 doc10

    # 전체
    python run_2x2.py --conds B C D
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
    from protected_span import mask, protect_and_restore, PH_RE
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder / protected_span 필요: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
OUTDIR = os.path.join(HOME, "outputs_2x2")
MODEL = "snuh/hari-q3-8b"
PROMPT_VERSION = "2x2-v1"

# ── P4 — L2 이독성을 겨냥한 평이화 프롬프트 ──────────────────────────
P4 = (
    "다음 수술 동의서를 한국어를 배우는 외국인 환자가 읽을 수 있도록 "
    "쉬운 한국어로 다시 써 주세요.\n"
    "\n"
    "반드시 지킬 것\n"
    "- 숫자, 단위, 기간, 비율을 바꾸거나 새로 만들지 마세요.\n"
    "- '없다', '아니다' 같은 부정 표현을 바꾸지 마세요.\n"
    "- '드물게', '흔히', '가능성이 있다' 같은 빈도·위험도 표현을 바꾸지 마세요.\n"
    "- 환자의 권리, 비용, 동의에 관한 내용을 빼지 마세요.\n"
    "- 내용을 요약하거나 중요하지 않다고 판단해 빼지 마세요.\n"
    "\n"
    "쉽게 쓰는 방법\n"
    "- 한 문장은 10어절 이내로 짧게 나누세요.\n"
    "- 어려운 일반 낱말은 쉬운 일상 낱말로 바꾸세요.\n"
    "  (예: 소요된다 → 걸린다, 시행한다 → 한다, 발생한다 → 생긴다)\n"
    "- 의료 전문용어는 지우거나 다른 용어로 바꾸지 마세요. "
    "대신 '전문용어(쉬운 설명)' 형태로 설명을 붙이세요.\n"
    "  (예: 연하곤란(음식을 삼키기 어려운 것), 불유합(뼈가 붙지 않는 것))\n"
    "- 한자어 표현보다 풀어 쓴 표현을 쓰세요.\n"
    "\n"
    "다시 쓴 한국어 본문만 출력하세요.")

PROMPT_BY_COND = {"A": PROMPTS["P0"], "B": PROMPTS["P0"], "C": P4, "D": P4}
PROTECT_BY_COND = {"A": False, "B": True, "C": False, "D": True}

# Protected ON 일 때 프롬프트에 덧붙이는 안내
# 주의 — 이 안내문에 구체적 기호명(예: N1, B2)을 쓰면 모델이 그것을
# 출력에 베껴 넣어 bijection audit 가 실패한다. 실제로 1차 실행에서
# B 조건 4건, D 조건 2건이 이 때문에 fail-closed 되었다.
# 따라서 괄호 모양만 설명하고 기호명은 예시로 들지 않는다.
PH_NOTE = (
    "\n\n[중요] 본문에는 ⟦ 와 ⟧ 로 둘러싸인 기호들이 들어 있습니다. "
    "이 기호는 숫자나 빈칸을 대신하는 표시입니다.\n"
    "- 기호를 글자 그대로 그 자리에 남겨 두세요.\n"
    "- 기호의 내용을 바꾸지 마세요.\n"
    "- 기호를 지우지 마세요.\n"
    "- 기호를 새로 만들거나 개수를 늘리지 마세요.\n"
    "- 본문에 있던 기호는 각각 딱 한 번만 나와야 합니다.")


class Gen:
    def __init__(self, gpu: int, cap: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.cap = cap
        print(f"[GEN] {MODEL} → cuda:{gpu}  greedy", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map={"": gpu})
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map={"": gpu})
        self.model.eval()

    def run(self, user: str) -> str:
        torch = self.torch
        msgs = [{"role": "system", "content": SYS_PROMPT},
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
            o = self.model.generate(**inp, max_new_tokens=self.cap,
                                    do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(o[0][n:], skip_special_tokens=True).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        if "</think>" in raw:
            raw = raw.split("</think>")[-1]
        return raw.strip()

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


def path_of(cond: str, doc: str) -> str:
    return os.path.join(OUTDIR, cond, f"{doc}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["B"],
                    choices=["B", "C", "D"],
                    help="A 는 기존 outputs_prompt 를 재사용하므로 생성하지 않음")
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

    print("=" * 92)
    print("2x2 생성 — P0/P4 × Protected OFF/ON")
    print("=" * 92)
    print(f"  모델 {MODEL} / greedy / 문서 {len(docs)}건")
    print(f"  조건: {', '.join(a.conds)}   (A 는 outputs_prompt 재사용)")
    for c in a.conds:
        print(f"    {c}: {'P4' if c in 'CD' else 'P0'} / "
              f"Protected {'ON' if PROTECT_BY_COND[c] else 'OFF'}")
    print(f"  출력: {OUTDIR}/<조건>/")

    todo = [(c, d) for c in a.conds for d in docs
            if a.force or not os.path.exists(path_of(c, d))]
    if not todo:
        print("\n생성할 것이 없습니다 (--force 로 재생성)")
        return
    for c in a.conds:
        os.makedirs(os.path.join(OUTDIR, c), exist_ok=True)

    gen = Gen(a.gpu, a.max_new)
    stats: Dict[str, Dict[str, int]] = {c: {"ok": 0, "fail": 0, "ph": 0}
                                        for c in a.conds}
    t0 = time.time()
    for n, (cond, doc) in enumerate(todo, 1):
        K = load_K(doc, a.docdir)
        if not K:
            print(f"  [{cond}/{doc}] K 없음 — 건너뜀")
            continue

        protect = PROTECT_BY_COND[cond]
        prompt_body = PROMPT_BY_COND[cond]
        if protect:
            m = mask(K)
            user = prompt_body + PH_NOTE + "\n\n" + m.text
        else:
            m = None
            user = prompt_body + "\n\n" + K

        raw = gen.run(user)
        cleaned, removed = Cleaner.clean(raw)

        audit_ok, reasons, detail = True, [], {}
        final = cleaned
        if protect:
            restored, ar = protect_and_restore(cleaned, m)
            audit_ok, reasons, detail = ar.ok, ar.reasons, ar.detail
            # fail-closed — 평이화를 포기하고 원문을 그대로 둔다
            final = restored if ar.ok else K
            stats[cond]["ph"] += m.n
            stats[cond]["ok" if ar.ok else "fail"] += 1
        else:
            stats[cond]["ok"] += 1

        with open(path_of(cond, doc), "w", encoding="utf-8") as f:
            json.dump({
                "doc": doc, "cond": cond,
                "prompt_kind": "P4" if cond in "CD" else "P0",
                "protected": protect,
                "model": MODEL, "prompt_version": PROMPT_VERSION,
                "prompt": prompt_body,
                "placeholder_note": PH_NOTE if protect else None,
                "gen_config": {"do_sample": False, "max_new_tokens": a.max_new},
                "src": K,
                "masked_src": m.text if m else None,
                "n_placeholder": (m.n if m else 0),
                "placeholder_map": (m.mapping if m else None),
                "out_raw": raw,
                "out_cleaned_before_restore": cleaned if protect else None,
                "out": final,
                "meta_removed": removed,
                "audit_ok": audit_ok,
                "audit_reasons": reasons,
                "audit_detail": detail,
                "fail_closed_used_source": bool(protect and not audit_ok),
            }, f, ensure_ascii=False, indent=2)

        el = time.time() - t0
        mark = "" if audit_ok else "  [AUDIT 실패 → 원문 사용]"
        ph = f" ph={m.n}" if m else ""
        print(f"  [{n}/{len(todo)}] {cond}/{doc}: {len(K)}자 → {len(final)}자 "
              f"({100*(len(final)-len(K))/max(len(K),1):+.0f}%){ph}"
              f"  [{el:.0f}s, 남은 {el/n*(len(todo)-n):.0f}s]{mark}", flush=True)
        if not audit_ok:
            for r in reasons[:3]:
                print(f"        {r}")
    gen.free()

    print("\n" + "=" * 92)
    print("bijection audit 결과")
    print("=" * 92)
    print(f"  {'조건':<6}{'통과':>6}{'실패':>6}{'placeholder 총계':>18}")
    for c in a.conds:
        s = stats[c]
        print(f"  {c:<6}{s['ok']:>6}{s['fail']:>6}{s['ph']:>18}")
    tot_fail = sum(stats[c]["fail"] for c in a.conds)
    if tot_fail:
        print(f"""
  실패한 {tot_fail}건은 fail-closed 로 원문을 그대로 두었습니다.
  해당 문서는 평이화 효과가 0 이고 Safety 는 Identity 와 같아집니다.
  평가 시 이 사실을 함께 보고해야 합니다.""")
    else:
        print("\n  모든 문서가 audit 를 통과했습니다.")

    print(f"""
다음
  python phase2_eval.py --conds {' '.join(['A'] + a.conds)}
""")


if __name__ == "__main__":
    main()
