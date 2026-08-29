#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_kormedmcqa.py v2 — hari-q3-8b를 KorMedMCQA로 평가

채점 방식(발표 설명용):
  프롬프트를 「문제 + 보기 A~E + '정답:'」까지 만들어 모델에 한 번 넣는다.
  모델이 그다음 글자로 무엇을 예측하는지 확인해, A~E 다섯 기호의
  점수만 비교하고 가장 높은 것을 모델의 답으로 본다.
  정답과 맞은 비율이 정답률이다.
  → 생성이 아니라 확률 비교이므로 몇 번을 돌려도 결과가 같다.

v1에서 고친 점:
  v1은 보기 문장 전체("A. 대한의사협회장")의 확률을 쟀는데,
  그 문장이 프롬프트에 이미 들어 있어 모델이 베끼기만 하면 됐다.
  그래서 다섯 보기 점수가 모두 0에 붙어 변별되지 않았다.
  MMLU 계열의 표준대로 보기 기호만 채점하도록 바꿨다.
"""
import os, json, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL, GPU = "snuh/hari-q3-8b", 0
DATASET = "sean0042/KorMedMCQA"
OUT = os.path.expanduser("~/이윤우/kormedmcqa_result.json")
LETTERS = ["A", "B", "C", "D", "E"]

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--subjects", default="doctor,nurse,pharm,dentist")
args = ap.parse_args()
subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

print("[모델 로딩]", MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()

# A~E 각 기호의 토큰 id 결정 (앞 공백 포함/미포함 중 1토큰이 되는 쪽)
LETTER_IDS = {}
for L in LETTERS:
    cand = None
    for v in (f" {L}", L):
        ids = tok(v, add_special_tokens=False).input_ids
        if len(ids) == 1: cand = ids[0]; break
    if cand is None:
        cand = tok(f" {L}", add_special_tokens=False).input_ids[0]
    LETTER_IDS[L] = cand
print("[보기 토큰 id]", LETTER_IDS)

def opts_of(row):
    return [str(row[L]) for L in LETTERS if row.get(L) not in (None, "")]

def gold_of(row, n):
    v = row.get("answer")
    if isinstance(v, int): return v - 1 if 1 <= v <= n else v
    v = str(v).strip()
    if v.isdigit():
        iv = int(v); return iv - 1 if 1 <= iv <= n else iv
    return LETTERS.index(v.upper()) if v.upper() in LETTERS else None

def build_prompt(shots, q, opts):
    s = "다음은 한국 보건의료인 국가시험 문제입니다. 알맞은 답을 고르시오.\n\n"
    for sq, so, sa in shots:
        s += f"문제: {sq}\n"
        for i, o in enumerate(so): s += f"{LETTERS[i]}. {o}\n"
        s += f"정답: {LETTERS[sa]}\n\n"
    s += f"문제: {q}\n"
    for i, o in enumerate(opts): s += f"{LETTERS[i]}. {o}\n"
    s += "정답:"
    return s

@torch.no_grad()
def letter_scores(prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    logits = model(ids).logits[0, -1].float()        # 마지막 위치의 다음 토큰 예측
    lp = torch.log_softmax(logits, dim=-1)
    return [lp[LETTER_IDS[LETTERS[j]]].item() for j in range(n)]

RESULT, SUMMARY = {}, []
for subj in subjects:
    print("\n" + "=" * 74); print(f"[{subj}] 로딩")
    dd = load_dataset(DATASET, subj)
    print("  split:", {k: len(v) for k, v in dd.items()})
    test = dd["test"]
    src = dd["fewshot"] if "fewshot" in dd else dd["train"]
    shots = []
    for row in src:
        o = opts_of(row); a = gold_of(row, len(o))
        if a is not None: shots.append((str(row["question"]), o, a))
        if len(shots) >= 5: break
    print(f"  few-shot {len(shots)}개 (fewshot 스플릿) / 평가 {len(test)}문항")

    items = list(test)
    if args.limit: items = items[:args.limit]
    correct = total = 0; records = []; t0 = time.time()

    for i, row in enumerate(items):
        opts = opts_of(row); gold = gold_of(row, len(opts)); q = str(row["question"])
        if len(opts) < 2 or gold is None: continue
        prompt = build_prompt(shots, q, opts)
        sc = letter_scores(prompt, len(opts))
        pred = int(max(range(len(sc)), key=lambda j: sc[j]))
        ok = (pred == gold); correct += ok; total += 1
        records.append({"year": row.get("year"), "q": q[:180],
                        "pred": LETTERS[pred], "gold": LETTERS[gold],
                        "ok": bool(ok), "scores": [round(s, 3) for s in sc]})
        if i == 0:
            spread = max(sc) - min(sc)
            print("\n  ── 첫 문항 실행 예시 (발표 자료용) ──")
            print("  문제:", q[:140].replace("\n", " "))
            for j, (o, s) in enumerate(zip(opts, sc)):
                mk = "  ← 모델 선택" if j == pred else ("  (정답)" if j == gold else "")
                print(f"   {LETTERS[j]}. {str(o)[:50]:<50} 점수 {s:+.3f}{mk}")
            print(f"  최고-최저 점수 차이 {spread:.3f}  "
                  f"{'(정상 — 변별됨)' if spread > 0.5 else '(경고 — 변별 안 됨)'}")
            print("  ─────────────────────────────────\n", flush=True)
        if total % 100 == 0:
            print(f"  {total}문항 / {correct/total*100:.1f}% / {time.time()-t0:.0f}초", flush=True)

    acc = correct / total * 100 if total else 0; dt = time.time() - t0
    print(f"\n  [{subj}] {correct}/{total} = {acc:.2f}%   ({dt/60:.1f}분)")
    RESULT[subj] = {"correct": correct, "total": total, "accuracy": acc,
                    "minutes": round(dt/60, 1), "records": records}
    SUMMARY.append((subj, correct, total, acc))

CARD = {"doctor": 76.78, "nurse": 83.60, "pharm": 84.41}
print("\n" + "=" * 74); print("[요약] 모델 카드 대비"); print("=" * 74)
print(f"{'과목':<10}{'정답률':>10}{'모델카드':>10}{'차이':>10}{'문항수':>8}")
for subj, c, t, a in SUMMARY:
    ref = CARD.get(subj)
    print(f"{subj:<10}{a:>9.2f}%{(f'{ref:.2f}%' if ref else '-'):>10}"
          f"{(f'{a-ref:+.2f}%p' if ref else '-'):>10}{t:>8}")
json.dump(RESULT, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[저장] {OUT}")
