import torch, time, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "snuh/hari-q3-8b"
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval(); print("[로딩 완료]\n", flush=True)

# 압축이 일어나도록 긴 원문 (Oh et al. 조건에 근접)
SOURCE = """복강경 담낭절제술 동의서

1. 수술의 목적 및 필요성
담낭에 결석이 있거나 만성 염증이 반복되는 경우, 담낭을 제거하여 통증과 합병증을 예방합니다.
급성 담낭염이 동반된 경우 천공, 복막염으로 진행할 위험이 있어 수술이 권고됩니다.

2. 수술 방법
전신마취 하에 복부에 3~4개의 작은 구멍을 만들어 복강경 기구를 삽입하고 담낭을 절제합니다.
수술 시간은 평균 1시간 내외이며, 유착이 심한 경우 개복 수술로 전환될 수 있습니다.
개복 전환율은 약 5% 내외로 보고됩니다.

3. 예상되는 후유증 및 부작용
- 담즙 누출: 드물게 발생하며, 담낭 염증이 심한 경우 발생률이 높아집니다. 배액관 삽입이 필요할 수 있습니다.
- 담관 손상: 약 1~2%에서 나타나며, 이 경우 추가 수술이나 내시경적 처치가 필요합니다.
- 출혈: 수술 중 또는 수술 후 발생할 수 있으며, 수혈이 필요한 경우는 1% 미만입니다.
- 감염: 고령이거나 당뇨가 있는 환자에서 위험이 증가하며, 항생제 치료가 필요합니다.
- 잔여 결석: 총담관에 결석이 남아 있는 경우 약 3%에서 추가 시술이 필요합니다.

4. 수술 전후 주의사항
수술 전 8시간 이상 금식이 필요하며, 항응고제 복용 중인 경우 최소 5일 전 중단해야 합니다.
수술 후 2주간 무거운 물건을 들지 않도록 하며, 발열이나 심한 복통 시 즉시 내원해야 합니다."""

# 사실 단위(atomic fact)로 분해 — 여러 표현을 허용
FACTS = {
    "술기명 복강경 담낭절제술":  [r"복강경.{0,4}담낭절제"],
    "전신마취":                 [r"전신\s*마취"],
    "구멍 3~4개":               [r"3\s*[~\-–]\s*4|서너\s*개|3개.{0,6}4개"],
    "수술시간 1시간":            [r"1\s*시간"],
    "개복 전환 가능성":          [r"개복"],
    "개복 전환율 5%":            [r"5\s*%|5퍼센트"],
    "담즙누출 '드물게'":         [r"드물"],
    "담즙누출 조건(염증 심함)":   [r"염증.{0,10}(심|중증)|(심|중증).{0,10}염증"],
    "배액관":                   [r"배액"],
    "담관손상 1~2%":            [r"1\s*[~\-–]\s*2\s*%|1%.{0,6}2%"],
    "담관손상시 추가처치":       [r"추가.{0,4}수술|재수술|내시경"],
    "수혈 1% 미만":             [r"1\s*%\s*미만|1퍼센트\s*미만"],
    "감염 조건(고령·당뇨)":      [r"고령", r"당뇨"],
    "잔여결석 3%":              [r"3\s*%|3퍼센트"],
    "금식 8시간":               [r"8\s*시간"],
    "항응고제 5일 전 중단":      [r"5\s*일"],
    "수술 후 2주":              [r"2\s*주"],
    "발열·복통 시 내원":         [r"발열", r"복통"],
}

CONDS = [
    ("1. 힌트 없음",
     "다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요."),
    ("2. 정보 유지 요청 (Oh et al. 방식)",
     "다음 수술 동의서를 중학교 1학년이 쉽게 이해할 수 있게, "
     "모든 필수 정보를 유지하면서 다시 써 주세요."),
    ("3. 구체적 지시",
     "다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
     "수술 이름, 모든 백분율 수치, 발생 조건, 시간·기간 정보는 "
     "하나도 빠짐없이 그대로 유지해 주세요."),
]

SYS = "당신은 한국어 의료 어시스턴트입니다. 원문에 없는 내용은 절대 추가하지 마세요."
results = {}

for name, instr in CONDS:
    msgs = [{"role":"system","content":SYS},
            {"role":"user","content":instr + "\n\n" + SOURCE}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=2048, do_sample=False)
    ans = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"{'='*70}\n[{name}]  ({time.time()-t0:.1f}초)\n{'='*70}")
    print(ans)
    print(f"\n--- 사실 보존 판정 ({len(FACTS)}개 항목) ---")
    kept, missing = 0, []
    for label, pats in FACTS.items():
        ok = all(re.search(p, ans) for p in pats)
        kept += ok
        if not ok: missing.append(label)
        print(f"  {'O' if ok else 'X'}  {label}")
    rate = kept/len(FACTS)*100
    ratio = len(ans)/len(SOURCE)*100
    print(f"\n  >> 사실 보존율 {rate:.1f}%  ({kept}/{len(FACTS)})")
    print(f"  >> 길이 {len(SOURCE)}자 → {len(ans)}자 ({ratio:.0f}%) "
          f"{'[압축]' if ratio<100 else '[확장]'}")
    if missing: print(f"  >> 소실: {', '.join(missing)}")
    print(flush=True)
    results[name] = {"보존율":rate, "길이비":ratio, "소실":missing, "출력":ans}

print(f"\n{'='*70}\n[전체 요약]\n{'='*70}")
print(f"  {'조건':36s} {'보존율':>8s} {'길이비':>8s}")
for k,v in results.items():
    print(f"  {k:36s} {v['보존율']:7.1f}% {v['길이비']:7.0f}%")

json.dump(results, open("/home/hufs/이윤우/ablation2_result.json","w"),
          ensure_ascii=False, indent=2)
print("\n결과 저장: ~/이윤우/ablation2_result.json")
