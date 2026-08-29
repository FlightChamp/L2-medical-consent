import torch, time, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [("hari-q3-8b", "snuh/hari-q3-8b", 0),
          ("Qwen3-8B",   "Qwen/Qwen3-8B",   1)]

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

INSTR = ("다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
         "수술 이름, 모든 백분율 수치, 발생 조건, 시간·기간 정보는 "
         "하나도 빠짐없이 그대로 유지해 주세요.")
SYS = "당신은 한국어 의료 어시스턴트입니다. 원문에 없는 내용은 절대 추가하지 마세요."

# 의미 변형까지 잡는 판정 (지난 실험에서 놓친 것 포함)
CHECKS = {
    "술기명":        [r"복강경.{0,4}담낭절제"],
    "마취":          [r"전신\s*마취|잠을\s*자|수면"],
    "구멍 3~4개":    [r"3\s*[~\-–]\s*4|서너\s*개"],
    "1시간":         [r"1\s*시간"],
    "개복 전환":     [r"개복|큰\s*절개|절개.{0,6}(크|넓)"],
    "5%":           [r"5\s*%|5퍼센트"],
    "드물게":        [r"드물|흔하지\s*않"],
    "염증 조건":     [r"염증.{0,10}(심|중증)|(심|중증).{0,10}염증"],
    "배액관":        [r"배액"],
    "1~2%":         [r"1\s*[~\-–]\s*2\s*%"],
    "추가처치":      [r"추가.{0,6}(수술|시술)|재수술|내시경"],
    "1% 미만":      [r"1\s*%\s*미만|1퍼센트\s*미만"],
    "고령·당뇨":     [r"고령|나이가\s*많|연세", r"당뇨"],
    "3%":           [r"3\s*%|3퍼센트"],
    "8시간 금식":    [r"8\s*시간"],
    "5일 전 중단":   [r"5\s*일"],
    "2주":          [r"2\s*주"],
    "발열·복통":     [r"발열|열이", r"복통|배.{0,4}아프|복부.{0,6}아프"],
}
# 사실 오류 탐지 (담낭절제술 후 담낭에 결석이 남을 수 없음)
def check_errors(ans):
    errs = []
    m = re.search(r"[^.]{0,40}(결석|돌)[^.]{0,20}(남|잔여)[^.]{0,30}", ans)
    seg = m.group(0) if m else ""
    if seg and "총담관" not in seg and "담관" not in seg:
        errs.append(f"잔여결석 위치 오류 → '{seg.strip()[:45]}'")
    if re.search(r"복부에\s*(큰\s*)?구멍이\s*생기", ans):
        errs.append("천공 오해 (담낭 천공 → 복부에 구멍)")
    if re.search(r"(설명해\s*드릴|필요하시면|도움이\s*되었|이해할\s*수\s*있도록\s*했)", ans):
        errs.append("메타 발화 (동의서 본문에 부적합)")
    return errs

results = {}
for label, path, gpu in MODELS:
    print(f"\n{'#'*72}\n### {label}  (GPU {gpu})\n{'#'*72}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": gpu})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": gpu})
    model.eval()
    print(f"[로딩] {time.time()-t0:.1f}초 | VRAM {torch.cuda.memory_allocated(gpu)/1024**3:.2f} GB", flush=True)

    msgs = [{"role":"system","content":SYS},{"role":"user","content":INSTR+"\n\n"+SOURCE}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=2048, do_sample=False)
    gen_t = time.time() - t1
    ans = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    ntok = out.shape[1] - inp.input_ids.shape[1]

    print(f"\n{ans}\n")
    print("--- 사실 보존 ---")
    kept, missing = 0, []
    for k, pats in CHECKS.items():
        ok = all(re.search(p, ans) for p in pats)
        kept += ok
        if not ok: missing.append(k)
        print(f"  {'O' if ok else 'X'}  {k}")
    errs = check_errors(ans)
    print(f"\n  >> 보존율 {kept/len(CHECKS)*100:.1f}% ({kept}/{len(CHECKS)})")
    print(f"  >> 길이 {len(SOURCE)} → {len(ans)}자 ({len(ans)/len(SOURCE)*100:.0f}%)")
    print(f"  >> 속도 {gen_t:.1f}초 / {ntok}토큰 / {ntok/gen_t:.1f} tok/s")
    if missing: print(f"  >> 미검출: {', '.join(missing)}")
    print(f"  >> 사실 오류: {'; '.join(errs) if errs else '없음'}", flush=True)

    results[label] = {"보존율":kept/len(CHECKS)*100, "길이비":len(ans)/len(SOURCE)*100,
                      "속도":ntok/gen_t, "미검출":missing, "오류":errs, "출력":ans}
    del model; torch.cuda.empty_cache()

print(f"\n{'='*72}\n[비교 요약]\n{'='*72}")
print(f"  {'모델':14s} {'보존율':>8s} {'길이비':>8s} {'tok/s':>8s}  사실오류")
for k,v in results.items():
    print(f"  {k:14s} {v['보존율']:7.1f}% {v['길이비']:7.0f}% {v['속도']:7.1f}  {len(v['오류'])}건")

json.dump(results, open("/home/hufs/이윤우/compare_result.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/compare_result.json")
