import torch, time, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [("hari-q3-8b", "snuh/hari-q3-8b", 0), ("Qwen3-8B", "Qwen/Qwen3-8B", 1)]

DOCS = {
"담낭절제술": {
"text": """복강경 담낭절제술 동의서
전신마취 하에 복부에 3~4개의 구멍을 만들어 담낭을 절제합니다. 수술 시간은 평균 1시간이며,
유착이 심한 경우 개복 수술로 전환될 수 있습니다(약 5%).
급성 담낭염 시 천공, 복막염으로 진행할 위험이 있습니다.
- 담즙 누출: 드물게 발생, 담낭 염증이 심한 경우 발생률 상승. 배액관 삽입 필요.
- 담관 손상: 약 1~2%, 추가 수술이나 내시경적 처치 필요.
- 잔여 결석: 총담관에 결석이 남은 경우 약 3%에서 추가 시술.
수술 전 8시간 금식, 항응고제는 5일 전 중단. 수술 후 2주간 중량물 금지.""",
"terms": ["복강경","담낭절제","전신마취","개복","천공","복막염","담즙 누출","배액관","담관","총담관","항응고제"],
"facts": {"3~4개":r"3\s*[~\-–]\s*4|서너","1시간":r"1\s*시간","5%":r"5\s*%","1~2%":r"1\s*[~\-–]\s*2\s*%",
          "3%":r"3\s*%","8시간":r"8\s*시간","5일":r"5\s*일","2주":r"2\s*주","드물게":r"드물|흔하지\s*않"}},

"갑상선절제술": {
"text": """갑상선 절제술 동의서
전신마취 하에 경부 절개를 통해 갑상선의 일부 또는 전부를 절제합니다.
- 되돌이후두신경 손상: 일시적 성대마비 약 5%, 영구적 마비는 1% 미만.
- 부갑상선 기능저하증: 전절제 시 일시적으로 약 20%, 영구적은 2% 미만에서 발생하며
  저칼슘혈증으로 손발 저림이 나타날 수 있습니다. 칼슘제 복용이 필요합니다.
- 출혈로 인한 혈종: 드물게 발생하나 기도 압박 시 응급 재수술이 필요합니다.
갑상선 전절제 시 평생 갑상선호르몬제를 복용해야 합니다.
수술 후 3일간 목을 뒤로 젖히지 않도록 주의합니다.""",
"terms": ["갑상선","전신마취","되돌이후두신경","성대마비","부갑상선","저칼슘혈증","혈종","기도 압박","갑상선호르몬"],
"facts": {"성대마비 5%":r"5\s*%","영구마비 1%미만":r"1\s*%\s*미만","일시적 20%":r"20\s*%",
          "영구 2%미만":r"2\s*%\s*미만","3일":r"3\s*일","평생 복용":r"평생|계속|지속적으로"}},

"백내장수술": {
"text": """백내장 수술 동의서
점안마취 후 각막을 절개하고 초음파로 혼탁된 수정체를 제거한 뒤 인공수정체를 삽입합니다.
수술 시간은 약 20분이며 대부분 당일 퇴원합니다.
- 후발백내장: 수술 후 수개월~수년 뒤 약 20%에서 발생하며 레이저 시술로 치료합니다.
- 안내염: 0.05% 미만으로 매우 드물지만 실명에 이를 수 있는 중대한 합병증입니다.
- 망막박리: 고도근시 환자에서 위험이 증가합니다.
수술 후 1주간 눈을 비비거나 물이 들어가지 않도록 해야 합니다.""",
"terms": ["백내장","점안마취","각막","수정체","인공수정체","후발백내장","안내염","실명","망막박리","고도근시"],
"facts": {"20분":r"20\s*분","후발 20%":r"20\s*%","안내염 0.05%미만":r"0\.05\s*%","1주":r"1\s*주",
          "고도근시 조건":r"고도\s*근시|심한\s*근시"}},
}

INSTR = ("다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
         "수술 이름, 모든 백분율 수치, 발생 조건, 시간·기간 정보는 하나도 빠짐없이 유지해 주세요.")
SYS = "당신은 한국어 의료 어시스턴트입니다. 원문에 없는 내용은 절대 추가하지 마세요."

results = {}
for label, path, gpu in MODELS:
    print(f"\n{'#'*72}\n### {label} (GPU {gpu})\n{'#'*72}", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": gpu})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": gpu})
    model.eval()
    results[label] = {}

    for dname, d in DOCS.items():
        msgs = [{"role":"system","content":SYS},{"role":"user","content":INSTR+"\n\n"+d["text"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=2048, do_sample=False)
        ans = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

        kept_t = [t for t in d["terms"] if t in ans]
        lost_t = [t for t in d["terms"] if t not in ans]
        kept_f = [k for k,p in d["facts"].items() if re.search(p, ans)]
        lost_f = [k for k,p in d["facts"].items() if not re.search(p, ans)]
        tr, fr = len(kept_t)/len(d["terms"])*100, len(kept_f)/len(d["facts"])*100

        print(f"\n{'='*72}\n[{dname}]\n{'='*72}")
        print(ans)
        print(f"\n  >> 의료용어 보존 {tr:.0f}% ({len(kept_t)}/{len(d['terms'])})"
              f"{'  소실: '+', '.join(lost_t) if lost_t else ''}")
        print(f"  >> 수치·조건 보존 {fr:.0f}% ({len(kept_f)}/{len(d['facts'])})"
              f"{'  소실: '+', '.join(lost_f) if lost_f else ''}")
        print(f"  >> 길이 {len(d['text'])} → {len(ans)}자 ({len(ans)/len(d['text'])*100:.0f}%)", flush=True)
        results[label][dname] = {"용어":tr,"수치":fr,"용어소실":lost_t,"수치소실":lost_f,"출력":ans}

    del model; torch.cuda.empty_cache()

print(f"\n{'='*72}\n[종합]\n{'='*72}")
print(f"  {'문서':14s}{'hari 용어':>10s}{'Qwen 용어':>10s}{'hari 수치':>10s}{'Qwen 수치':>10s}")
for dname in DOCS:
    h, q = results["hari-q3-8b"][dname], results["Qwen3-8B"][dname]
    print(f"  {dname:14s}{h['용어']:9.0f}%{q['용어']:9.0f}%{h['수치']:9.0f}%{q['수치']:9.0f}%")
for m in results:
    avg_t = sum(results[m][d]["용어"] for d in DOCS)/len(DOCS)
    avg_f = sum(results[m][d]["수치"] for d in DOCS)/len(DOCS)
    print(f"\n  {m:14s} 평균 용어 {avg_t:.1f}% / 수치 {avg_f:.1f}%")

json.dump(results, open("/home/hufs/이윤우/compare2_result.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/compare2_result.json")
