import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "snuh/hari-q3-8b"
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()
print("[로딩 완료]\n", flush=True)

SOURCE = (
    "복강경 담낭절제술은 담낭을 제거하는 수술입니다. "
    "수술 후 드물게 담즙 누출이 발생할 수 있으며, 담낭 염증이 심한 경우 발생률이 높아집니다. "
    "담관 손상은 약 1~2%에서 나타나며, 이 경우 추가 수술이 필요할 수 있습니다. "
    "고령이거나 당뇨가 있는 환자에서는 감염 위험이 증가합니다."
)

# 보존 여부를 판정할 핵심 요소
CHECKS = {
    "술기명(복강경 담낭절제술)": ["복강경", "담낭절제"],
    "한정어 '드물게'":            ["드물"],
    "조건 '염증이 심한 경우'":     ["염증", "심한"],
    "수치 '1~2%'":               ["1~2", "1-2", "1%", "2%"],
    "조건 '고령·당뇨'":           ["고령", "당뇨"],
    "결과 '추가 수술'":           ["추가 수술", "재수술"],
}

CONDS = [
    ("조건1: 힌트 없음",
     "다음 수술 동의서 내용을 환자가 이해하기 쉬운 말로 설명해 주세요.\n\n" + SOURCE),
    ("조건2: 정보 유지 요청",
     "다음 수술 동의서 내용을 환자가 이해하기 쉬운 말로 설명해 주세요. "
     "단, 모든 정보를 빠짐없이 유지해 주세요.\n\n" + SOURCE),
    ("조건3: 구체적 지시",
     "다음 수술 동의서 내용을 환자가 이해하기 쉬운 말로 설명해 주세요. "
     "수술 이름, 발생 빈도, 발생 조건, 수치는 반드시 그대로 유지해 주세요.\n\n" + SOURCE),
]

SYS = ("당신은 임상 지식을 갖춘 한국어 의료 어시스턴트입니다. "
       "원문에 없는 내용은 절대 추가하지 마세요.")

results = {}
for name, prompt in CONDS:
    msgs = [{"role":"system","content":SYS},{"role":"user","content":prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
    ans = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"{'='*66}\n[{name}]  ({time.time()-t0:.1f}초)\n{'='*66}")
    print(ans)
    print(f"\n--- 보존 판정 ---")
    kept = {}
    for label, kws in CHECKS.items():
        ok = any(k in ans for k in kws)
        kept[label] = ok
        print(f"  {'O' if ok else 'X'}  {label}")
    rate = sum(kept.values())/len(kept)*100
    print(f"  >> 보존율 {rate:.0f}% ({sum(kept.values())}/{len(kept)})")
    print(f"  >> 원문 {len(SOURCE)}자 → 출력 {len(ans)}자 ({len(ans)/len(SOURCE)*100:.0f}%)\n", flush=True)
    results[name] = {"보존율": rate, "항목": kept, "출력": ans}

print(f"\n{'='*66}\n[전체 요약]\n{'='*66}")
for k, v in results.items():
    print(f"  {k:22s} 보존율 {v['보존율']:.0f}%")

with open("/home/hufs/이윤우/ablation_result.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n결과 저장: ~/이윤우/ablation_result.json")
