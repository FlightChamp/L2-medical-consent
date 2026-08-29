import torch, time, json, re, os
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = ("hari-q3-8b", "snuh/hari-q3-8b", 0)
PREV  = "/home/hufs/이윤우/compare2_result.json"

# 원문을 '확인해야 할 사실' 단위로 미리 분해 (사람이 정의)
UNITS = {
"담낭절제술": [
 "전신마취 하에 수술한다",
 "복부에 3~4개의 구멍을 만든다",
 "수술 시간은 평균 1시간이다",
 "유착이 심하면 개복 수술로 전환될 수 있다",
 "개복 전환율은 약 5%다",
 "급성 담낭염 시 천공, 복막염으로 진행할 위험이 있다",
 "담즙 누출은 드물게 발생한다",
 "담낭 염증이 심하면 담즙 누출 발생률이 상승한다",
 "담즙 누출 시 배액관 삽입이 필요하다",
 "담관 손상은 약 1~2%다",
 "담관 손상 시 추가 수술이나 내시경적 처치가 필요하다",
 "총담관에 결석이 남으면 약 3%에서 추가 시술이 필요하다",
 "수술 전 8시간 금식한다",
 "항응고제는 5일 전 중단한다",
 "수술 후 2주간 중량물을 금지한다",
],
"갑상선절제술": [
 "전신마취 하에 수술한다",
 "경부 절개를 통해 갑상선의 일부 또는 전부를 절제한다",
 "되돌이후두신경이 손상될 수 있다",
 "일시적 성대마비는 약 5%다",
 "영구적 마비는 1% 미만이다",
 "부갑상선 기능저하증이 생길 수 있다",
 "전절제 시 일시적 부갑상선 기능저하증은 약 20%다",
 "영구적 부갑상선 기능저하증은 2% 미만이다",
 "저칼슘혈증으로 손발 저림이 나타날 수 있다",
 "칼슘제 복용이 필요하다",
 "출혈로 인한 혈종이 드물게 발생한다",
 "기도 압박 시 응급 재수술이 필요하다",
 "전절제 시 평생 갑상선호르몬제를 복용해야 한다",
 "수술 후 3일간 목을 뒤로 젖히지 않는다",
],
"백내장수술": [
 "점안마취를 한다",
 "각막을 절개한다",
 "초음파로 혼탁된 수정체를 제거한다",
 "인공수정체를 삽입한다",
 "수술 시간은 약 20분이다",
 "대부분 당일 퇴원한다",
 "후발백내장은 수술 후 수개월~수년 뒤 발생한다",
 "후발백내장은 약 20%에서 발생한다",
 "후발백내장은 레이저 시술로 치료한다",
 "안내염은 0.05% 미만이다",
 "안내염은 실명에 이를 수 있는 중대한 합병증이다",
 "망막박리는 고도근시 환자에서 위험이 증가한다",
 "수술 후 1주간 눈을 비비지 않는다",
 "수술 후 1주간 물이 들어가지 않도록 한다",
],
}

SYS = "당신은 의료 문서 검수자입니다. 주어진 문장 하나만 판단하고, 지정된 형식으로만 답합니다."

TMPL = """아래 [변환문]을 읽고, [확인할 사실]이 변환문에 담겨 있는지 판단하세요.

[변환문]
{out}

[확인할 사실]
{unit}

표현이 쉬운 말로 바뀐 것은 담긴 것으로 봅니다.
(예: '고령' → '나이가 많은'은 담긴 것)
전문 진단명이 증상 설명으로만 바뀐 경우는 '부분'으로 봅니다.
(예: '저칼슘혈증' → '손발이 저림'은 부분)

다음 중 하나만 답하세요. 다른 말은 쓰지 마세요.
있음 / 부분 / 없음"""

def verdict(t):
    t = t.strip()
    for k in ("있음","부분","없음"):
        if k in t[:20]: return k
    return "판독불가"

if not os.path.exists(PREV):
    raise SystemExit(f"파일 없음: {PREV}")
prev = json.load(open(PREV))

label, path, gpu = JUDGE
tok = AutoTokenizer.from_pretrained(path)
try:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": gpu})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": gpu})
model.eval()
print(f"[판사] {label} | 총 {sum(len(v) for v in UNITS.values())*len(prev)}회 판정\n", flush=True)

out_all = {}
for gm in prev:
    out_all[gm] = {}
    for dname, units in UNITS.items():
        ans_text = prev[gm][dname]["출력"]
        print(f"{'='*72}\n[{gm}] × [{dname}]\n{'='*72}", flush=True)
        rec, t0 = [], time.time()
        for u in units:
            msgs = [{"role":"system","content":SYS},
                    {"role":"user","content":TMPL.format(out=ans_text, unit=u)}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(t, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=8, do_sample=False)
            v = verdict(tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
            mark = {"있음":"O","부분":"△","없음":"X"}.get(v,"?")
            print(f"  {mark}  {u}")
            rec.append((u, v))
        n_o = sum(1 for _,v in rec if v=="있음")
        n_p = sum(1 for _,v in rec if v=="부분")
        n_x = sum(1 for _,v in rec if v=="없음")
        print(f"\n  >> 있음 {n_o} / 부분 {n_p} / 없음 {n_x}  (총 {len(rec)}, {time.time()-t0:.0f}초)")
        prob = [u for u,v in rec if v in ("부분","없음")]
        if prob: print(f"  >> 문제 항목: {'; '.join(prob)}")
        print(flush=True)
        out_all[gm][dname] = {"판정":rec, "있음":n_o, "부분":n_p, "없음":n_x}

print(f"\n{'='*72}\n[종합]\n{'='*72}")
print(f"  {'생성모델 × 문서':30s}{'있음':>6s}{'부분':>6s}{'없음':>6s}{'보존율':>9s}")
for gm in out_all:
    for dn in out_all[gm]:
        r = out_all[gm][dn]
        tot = r["있음"]+r["부분"]+r["없음"]
        rate = (r["있음"] + r["부분"]*0.5)/tot*100
        print(f"  {gm+' × '+dn:30s}{r['있음']:>6d}{r['부분']:>6d}{r['없음']:>6d}{rate:>8.1f}%")

json.dump(out_all, open("/home/hufs/이윤우/judge2_result.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/judge2_result.json")
