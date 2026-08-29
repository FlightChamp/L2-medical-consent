"""동의서 변환 검증기 v1 — LLM 사실검증 + 룰 기반 용어검증 하이브리드"""
import torch, time, json, re, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 0
PREV = "/home/hufs/이윤우/compare2_result.json"

# ── 검증 명세: 사실 단위 + 필수 용어(정규식) + 법정 항목 매핑 ──
SPEC = {
"담낭절제술": {
 "facts": [
  ("전신마취 하에 수술한다", "②방법"),
  ("복부에 3~4개의 구멍을 만든다", "②방법"),
  ("수술 시간은 평균 1시간이다", "②방법"),
  ("유착이 심하면 개복 수술로 전환될 수 있다", "②방법"),
  ("개복 전환율은 약 5%다", "②방법"),
  ("급성 담낭염 시 천공, 복막염으로 진행할 위험이 있다", "②필요성"),
  ("담즙 누출은 드물게 발생한다", "④후유증"),
  ("담낭 염증이 심하면 담즙 누출 발생률이 상승한다", "④후유증"),
  ("담즙 누출 시 배액관 삽입이 필요하다", "④후유증"),
  ("담관 손상은 약 1~2%다", "④후유증"),
  ("담관 손상 시 추가 수술이나 내시경적 처치가 필요하다", "④후유증"),
  ("총담관에 결석이 남으면 약 3%에서 추가 시술이 필요하다", "④후유증"),
  ("수술 전 8시간 금식한다", "⑤준수사항"),
  ("항응고제는 5일 전 중단한다", "⑤준수사항"),
  ("수술 후 2주간 중량물을 금지한다", "⑤준수사항"),
 ],
 # 반드시 원문 그대로 유지되어야 할 진단명·술기명 (정규식, 조사/띄어쓰기 허용)
 "terms": {"복강경 담낭절제술": r"복강경\s*담낭\s*절제", "전신마취": r"전신\s*마취",
           "천공": r"천공", "복막염": r"복막염", "담즙 누출": r"담즙\s*누출",
           "배액관": r"배액관", "담관 손상": r"담관\s*손상", "총담관": r"총담관",
           "항응고제": r"항응고제"},
},
"갑상선절제술": {
 "facts": [
  ("전신마취 하에 수술한다", "②방법"),
  ("경부 절개를 통해 갑상선의 일부 또는 전부를 절제한다", "②방법"),
  ("되돌이후두신경이 손상될 수 있다", "④후유증"),
  ("일시적 성대마비는 약 5%다", "④후유증"),
  ("영구적 마비는 1% 미만이다", "④후유증"),
  ("부갑상선 기능저하증이 생길 수 있다", "④후유증"),
  ("전절제 시 일시적 부갑상선 기능저하증은 약 20%다", "④후유증"),
  ("영구적 부갑상선 기능저하증은 2% 미만이다", "④후유증"),
  ("저칼슘혈증으로 손발 저림이 나타날 수 있다", "④후유증"),
  ("칼슘제 복용이 필요하다", "⑤준수사항"),
  ("출혈로 인한 혈종이 드물게 발생한다", "④후유증"),
  ("기도 압박 시 응급 재수술이 필요하다", "④후유증"),
  ("전절제 시 평생 갑상선호르몬제를 복용해야 한다", "⑤준수사항"),
  ("수술 후 3일간 목을 뒤로 젖히지 않는다", "⑤준수사항"),
 ],
 "terms": {"갑상선 절제술": r"갑상선\s*절제", "전신마취": r"전신\s*마취",
           "되돌이후두신경": r"되돌이\s*후두\s*신경|반회\s*후두\s*신경",
           "성대마비": r"성대\s*마비", "부갑상선 기능저하증": r"부갑상선\s*기능\s*저하",
           "저칼슘혈증": r"저\s*칼슘\s*혈증", "혈종": r"혈종", "기도 압박": r"기도.{0,3}압박",
           "갑상선호르몬제": r"갑상선\s*호르몬"},
},
"백내장수술": {
 "facts": [
  ("점안마취를 한다", "②방법"),
  ("각막을 절개한다", "②방법"),
  ("초음파로 혼탁된 수정체를 제거한다", "②방법"),
  ("인공수정체를 삽입한다", "②방법"),
  ("수술 시간은 약 20분이다", "②방법"),
  ("대부분 당일 퇴원한다", "②방법"),
  ("후발백내장은 수술 후 수개월~수년 뒤 발생한다", "④후유증"),
  ("후발백내장은 약 20%에서 발생한다", "④후유증"),
  ("후발백내장은 레이저 시술로 치료한다", "④후유증"),
  ("안내염은 0.05% 미만이다", "④후유증"),
  ("안내염은 실명에 이를 수 있는 중대한 합병증이다", "④후유증"),
  ("망막박리는 고도근시 환자에서 위험이 증가한다", "④후유증"),
  ("수술 후 1주간 눈을 비비지 않는다", "⑤준수사항"),
  ("수술 후 1주간 물이 들어가지 않도록 한다", "⑤준수사항"),
 ],
 "terms": {"백내장": r"백내장", "점안마취": r"점안\s*마취", "각막": r"각막",
           "수정체": r"수정체", "인공수정체": r"인공\s*수정체",
           "후발백내장": r"후발\s*백내장", "안내염": r"안내염", "실명": r"실명",
           "망막박리": r"망막\s*박리", "고도근시": r"고도\s*근시"},
},
}

SYS = "당신은 의료 문서 검수자입니다. 주어진 문장 하나만 판단하고 지정된 단어로만 답합니다."
TMPL = """아래 [변환문]에 [확인할 사실]의 내용이 담겨 있습니까?

[변환문]
{out}

[확인할 사실]
{unit}

표현이 쉬운 말로 바뀌어도 같은 내용이면 '있음'입니다.
내용 자체가 나타나지 않으면 '없음'입니다.

있음 / 없음 중 하나만 답하세요."""

prev = json.load(open(PREV))
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()
print("[검증기 v1 시작]\n", flush=True)

report = {}
for gm in prev:
    report[gm] = {}
    for dname, spec in SPEC.items():
        ans = prev[gm][dname]["출력"]

        # ── ① LLM 사실 단위 검증 ──
        fact_res = []
        for unit, art in spec["facts"]:
            msgs = [{"role":"system","content":SYS},
                    {"role":"user","content":TMPL.format(out=ans, unit=unit)}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(t, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=6, do_sample=False)
            v = "있음" if "있음" in tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True)[:12] else "없음"
            fact_res.append((unit, art, v))

        # ── ② 룰 기반 필수 용어 검증 ──
        term_res = [(t, bool(re.search(p, ans))) for t, p in spec["terms"].items()]

        # ── ③ 리포트 ──
        lost_f = [(u,a) for u,a,v in fact_res if v=="없음"]
        lost_t = [t for t,ok in term_res if not ok]
        fr = (len(fact_res)-len(lost_f))/len(fact_res)*100
        tr = (len(term_res)-len(lost_t))/len(term_res)*100

        print(f"{'='*72}")
        print(f"■ {gm} × {dname}")
        print(f"{'='*72}")
        print(f"  [사실 보존]  {fr:5.1f}%  ({len(fact_res)-len(lost_f)}/{len(fact_res)})")
        for u,a in lost_f: print(f"      ❌ 소실  [{a}] {u}")
        print(f"  [용어 보존]  {tr:5.1f}%  ({len(term_res)-len(lost_t)}/{len(term_res)})")
        for t in lost_t: print(f"      ⚠️ 소실  {t}")

        arts = {}
        for u,a,v in fact_res: arts.setdefault(a, []).append(v)
        print(f"  [의료법 항목별]")
        for a in sorted(arts):
            ok = sum(1 for v in arts[a] if v=="있음")
            print(f"      {'✅' if ok==len(arts[a]) else '⚠️'} {a}: {ok}/{len(arts[a])}")
        print(f"  >> 종합 {(fr+tr)/2:.1f}%\n", flush=True)

        report[gm][dname] = {"사실보존":fr, "용어보존":tr, "사실소실":lost_f, "용어소실":lost_t}

print(f"{'='*72}\n[종합]\n{'='*72}")
print(f"  {'생성모델 × 문서':30s}{'사실':>8s}{'용어':>8s}{'종합':>8s}")
for gm in report:
    for dn in report[gm]:
        r = report[gm][dn]
        print(f"  {gm+' × '+dn:30s}{r['사실보존']:7.1f}%{r['용어보존']:7.1f}%{(r['사실보존']+r['용어보존'])/2:7.1f}%")

json.dump(report, open("/home/hufs/이윤우/verifier_report.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/verifier_report.json")
