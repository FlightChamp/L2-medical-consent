import torch, time, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "snuh/hari-q3-8b"
print(f"[{time.strftime('%H:%M:%S')}] 시작 | {torch.cuda.get_device_name(0)}", flush=True)

# --- 로딩 (transformers 버전별 dtype 인자 대응) ---
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()
load_t = time.time() - t0
print(f"[로딩] {load_t:.1f}초 | torch VRAM {torch.cuda.memory_allocated(0)/1024**3:.2f} GB", flush=True)

def nvsmi():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                        "--format=csv,noheader,nounits","-i","0"],
                       capture_output=True, text=True)
    return f"{int(r.stdout.strip())/1024:.2f} GB"
print(f"[nvidia-smi 실측] {nvsmi()}", flush=True)

SYS = ("당신은 임상 지식을 갖춘 한국어 의료 어시스턴트입니다. "
       "수술 동의서 내용을 환자가 이해하기 쉬운 말로 설명하되, "
       "원문에 없는 내용은 절대 추가하지 마세요.")

cases = [
    ("A. 동의서 문장 해석",
     "수술 동의서에 '복강경 담낭절제술 후 담즙 누출 및 담관 손상이 발생할 수 있습니다'라고 "
     "적혀 있습니다. 환자가 이해할 수 있게 쉬운 말로 설명해 주세요."),
    ("B. 위험 정보 추출 (한정어 보존 확인)",
     "다음 문장에서 예상되는 후유증과 부작용을 목록으로 정리해 주세요. "
     "발생 조건이나 빈도가 명시된 경우 그것도 함께 적어 주세요.\n\n"
     "'수술 후 간부전이 발생할 수 있으며, 간경변이 동반된 경우 발생률이 높고 "
     "전해질 불균형으로 이어질 수 있습니다. 드물게 출혈과 감염이 나타납니다.'"),
]

for title, q in cases:
    print(f"\n{'='*64}\n[{title}]\n{'='*64}", flush=True)
    msgs = [{"role":"system","content":SYS}, {"role":"user","content":q}]
    # Qwen3 thinking 모드 비활성화 (미지원 템플릿이면 자동 우회)
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    gen_t = time.time() - t1
    n = out.shape[1] - inputs.input_ids.shape[1]
    print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True), flush=True)
    print(f"\n-- {gen_t:.1f}초 / {n} 토큰 / {n/gen_t:.1f} tok/s", flush=True)

print(f"\n{'='*64}")
print(f"[요약] 로딩 {load_t:.1f}초 | torch 최대 VRAM "
      f"{torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB | nvidia-smi {nvsmi()}")
