import torch, time, json, re, os
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = ("hari-q3-8b", "snuh/hari-q3-8b", 0)   # 판사 모델
PREV  = "/home/hufs/이윤우/compare2_result.json"

# compare2.py와 동일한 원문
DOCS = {
"담낭절제술": """복강경 담낭절제술 동의서
전신마취 하에 복부에 3~4개의 구멍을 만들어 담낭을 절제합니다. 수술 시간은 평균 1시간이며,
유착이 심한 경우 개복 수술로 전환될 수 있습니다(약 5%).
급성 담낭염 시 천공, 복막염으로 진행할 위험이 있습니다.
- 담즙 누출: 드물게 발생, 담낭 염증이 심한 경우 발생률 상승. 배액관 삽입 필요.
- 담관 손상: 약 1~2%, 추가 수술이나 내시경적 처치 필요.
- 잔여 결석: 총담관에 결석이 남은 경우 약 3%에서 추가 시술.
수술 전 8시간 금식, 항응고제는 5일 전 중단. 수술 후 2주간 중량물 금지.""",

"갑상선절제술": """갑상선 절제술 동의서
전신마취 하에 경부 절개를 통해 갑상선의 일부 또는 전부를 절제합니다.
- 되돌이후두신경 손상: 일시적 성대마비 약 5%, 영구적 마비는 1% 미만.
- 부갑상선 기능저하증: 전절제 시 일시적으로 약 20%, 영구적은 2% 미만에서 발생하며
  저칼슘혈증으로 손발 저림이 나타날 수 있습니다. 칼슘제 복용이 필요합니다.
- 출혈로 인한 혈종: 드물게 발생하나 기도 압박 시 응급 재수술이 필요합니다.
갑상선 전절제 시 평생 갑상선호르몬제를 복용해야 합니다.
수술 후 3일간 목을 뒤로 젖히지 않도록 주의합니다.""",

"백내장수술": """백내장 수술 동의서
점안마취 후 각막을 절개하고 초음파로 혼탁된 수정체를 제거한 뒤 인공수정체를 삽입합니다.
수술 시간은 약 20분이며 대부분 당일 퇴원합니다.
- 후발백내장: 수술 후 수개월~수년 뒤 약 20%에서 발생하며 레이저 시술로 치료합니다.
- 안내염: 0.05% 미만으로 매우 드물지만 실명에 이를 수 있는 중대한 합병증입니다.
- 망막박리: 고도근시 환자에서 위험이 증가합니다.
수술 후 1주간 눈을 비비거나 물이 들어가지 않도록 해야 합니다.""",
}

# 우리가 수작업으로 확인한 정답 (판사 성능 검증용)
GROUND_TRUTH = {
 ("hari-q3-8b","담낭절제술"):   ["천공·복막염 문장 통째 누락"],
 ("hari-q3-8b","갑상선절제술"): ["저칼슘혈증 진단명 소실(증상만 남음)", "성대마비 → 목소리 마비"],
 ("hari-q3-8b","백내장수술"):   ["수정체 위치 오류('눈의 뒷부분')", "고도근시 설명 추가(원문에 없음)"],
 ("Qwen3-8B","담낭절제술"):     [],
 ("Qwen3-8B","갑상선절제술"):   ["저칼슘혈증 진단명 소실(증상만 남음)", "성대마비 → 목소리 마비"],
 ("Qwen3-8B","백내장수술"):     ["점안마취 누락", "메타 발화 추가", "고도근시 설명 추가(원문에 없음)"],
}

JUDGE_SYS = """당신은 의료 문서 검수자입니다. 원문과 변환문을 대조하여 문제를 찾아냅니다.
표현이 쉬운 말로 바뀐 것 자체는 문제가 아닙니다. 정보가 실제로 사라지거나 달라진 경우만 지적하세요."""

JUDGE_TMPL = """다음은 수술 동의서 원문과, 이를 환자가 이해하기 쉽게 다시 쓴 변환문입니다.
원문의 정보가 정확히 보존되었는지 검수해 주세요.

[원문]
{src}

[변환문]
{out}

다음 세 가지를 찾아 주세요.

1. 누락: 원문에 있는데 변환문에서 빠진 정보 (문장 전체가 빠진 경우 특히 주의)
2. 변형: 의미가 달라진 부분. 특히 진단명이 증상 설명으로 바뀐 경우
   (예: '저칼슘혈증' → '손이 저림')
3. 추가: 원문에 없는데 변환문에 들어간 내용 (용어 설명, 인사말 등)

주의: '고령'→'나이가 많은'처럼 같은 뜻을 쉬운 말로 바꾼 것은 문제가 아닙니다.

아래 JSON 형식으로만 답하세요. 문제가 없으면 빈 배열 []을 넣으세요.

{{"누락": ["항목1", "항목2"], "변형": ["원문표현 → 변환문표현"], "추가": ["항목1"]}}"""

def extract_json(t):
    m = re.search(r'\{.*\}', t, re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s.replace("'", '"')):
        try: return json.loads(cand)
        except Exception: pass
    # 부분 추출 폴백
    res = {}
    for k in ("누락","변형","추가"):
        mm = re.search(rf'"{k}"\s*:\s*\[(.*?)\]', s, re.S)
        res[k] = re.findall(r'"([^"]+)"', mm.group(1)) if mm else []
    return res if any(res.values()) else None

# ---- 실행 ----
if not os.path.exists(PREV):
    raise SystemExit(f"이전 결과 파일이 없습니다: {PREV}\ncompare2.py를 먼저 실행하세요.")
prev = json.load(open(PREV))

label, path, gpu = JUDGE
print(f"[판사 모델] {label} (GPU {gpu})", flush=True)
tok = AutoTokenizer.from_pretrained(path)
try:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": gpu})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": gpu})
model.eval()
print("[로딩 완료]\n", flush=True)

judged = {}
for gen_model in prev:
    judged[gen_model] = {}
    for dname in DOCS:
        out_text = prev[gen_model][dname]["출력"]
        msgs = [{"role":"system","content":JUDGE_SYS},
                {"role":"user","content":JUDGE_TMPL.format(src=DOCS[dname], out=out_text)}]
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(t, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=1024, do_sample=False)
        raw = tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = extract_json(raw)

        print(f"{'='*72}\n[생성: {gen_model}] × [문서: {dname}]  ({time.time()-t0:.1f}초)\n{'='*72}")
        if parsed:
            for k in ("누락","변형","추가"):
                items = parsed.get(k, []) or []
                print(f"  ▸ {k} ({len(items)}건)")
                for it in items: print(f"      - {it}")
        else:
            print("  !! JSON 파싱 실패 — 원문 출력:")
            print("  " + raw[:600].replace("\n","\n  "))
        gt = GROUND_TRUTH.get((gen_model, dname), [])
        print(f"\n  [수작업 확인 정답] {len(gt)}건")
        for g in gt: print(f"      · {g}")
        print(flush=True)
        judged[gen_model][dname] = {"판사출력": raw, "파싱": parsed, "정답": gt}

# ---- 판사 성능 요약 ----
print(f"\n{'='*72}\n[판사 성능 요약]\n{'='*72}")
print(f"  {'생성모델 × 문서':30s}{'판사 지적':>10s}{'정답':>8s}{'파싱':>8s}")
tot_j = tot_g = ok_parse = n = 0
for gm in judged:
    for dn in judged[gm]:
        r = judged[gm][dn]
        p = r["파싱"]
        cnt = sum(len(p.get(k,[]) or []) for k in ("누락","변형","추가")) if p else 0
        print(f"  {gm+' × '+dn:30s}{cnt:>10d}{len(r['정답']):>8d}{'O' if p else 'X':>8s}")
        tot_j += cnt; tot_g += len(r["정답"]); ok_parse += bool(p); n += 1
print(f"\n  판사 총 지적 {tot_j}건 / 수작업 정답 {tot_g}건 / JSON 파싱 성공 {ok_parse}/{n}")
print("\n  ※ 지적 건수가 정답보다 많아도 오탐이 아닐 수 있고,")
print("     적으면 놓친 것입니다. 위 상세 출력을 직접 대조하세요.")

json.dump(judged, open("/home/hufs/이윤우/judge_result.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/judge_result.json")
