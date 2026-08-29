"""환각 탐지기 v1 — 출력→원문 역방향 검증
   ① LLM 문장 단위 근거 확인  ② 룰: 괄호 설명 추가  ③ 룰: 메타 발화"""
import torch, time, json, re, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 0
PREV = "/home/hufs/이윤우/compare2_result.json"

SOURCES = {
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

# ── 룰 ① 메타 발화 (동의서 본문에 있으면 안 되는 화자 개입) ──
META = [
 (r"필요하시?면", "사용자에게 추가 제안"),
 (r"드립니다|드릴\s*수\s*있|드리겠", "화자 개입 표현"),
 (r"도움이\s*되(었|시)", "화자 개입 표현"),
 (r"이해할\s*수\s*있도록\s*했", "자기 작업 언급"),
 (r"다시\s*(정리|작성)해", "자기 작업 언급"),
 (r"원문에\s*있는\s*모든", "자기 작업 언급"),
 (r"^\s*물론입니다", "인사말"),
]

def split_sents(t):
    t = re.sub(r"\*\*|^\s*[-–•]\s*|^#+\s*", " ", t, flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" -–—:")
        if len(p) >= 12 and re.search(r"[가-힣]", p):
            out.append(p)
    return out

def find_parens(ans, src):
    """원문에 없는 괄호 설명 찾기"""
    hits = []
    for m in re.finditer(r"([가-힣A-Za-z]{2,12})\s*\(([^)]{4,60})\)", ans):
        term, expl = m.group(1), m.group(2)
        if re.match(r"^[\d\s~\-–.%]+$", expl):      # 숫자·기호만이면 제외
            continue
        core = re.sub(r"[^가-힣]", "", expl)
        if len(core) < 4:
            continue
        # 설명의 핵심 어절이 원문에 있는지
        toks = [w for w in re.findall(r"[가-힣]{2,}", expl) if len(w) >= 2]
        if toks and sum(1 for w in toks if w in src) / len(toks) < 0.5:
            hits.append(f"{term}({expl})")
    return hits

SYS = "당신은 의료 문서 검수자입니다. 주어진 문장 하나만 판단하고 지정된 단어로만 답합니다."
TMPL = """[원문]
{src}

[검사할 문장]
{sent}

위 문장의 내용이 원문에 근거가 있습니까?
표현이 쉬운 말로 바뀐 것은 근거 있음으로 봅니다.
원문에 없는 정보(용어 설명, 부연, 인사말 등)가 들어 있으면 근거 없음입니다.

근거있음 / 근거없음 중 하나만 답하세요."""

prev = json.load(open(PREV))
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()
print("[환각 탐지기 v1 시작]\n", flush=True)

report = {}
for gm in prev:
    report[gm] = {}
    for dname, src in SOURCES.items():
        ans = prev[gm][dname]["출력"]
        sents = split_sents(ans)

        # ① LLM 문장 단위 근거 확인
        ungrounded = []
        t0 = time.time()
        for s in sents:
            msgs = [{"role":"system","content":SYS},
                    {"role":"user","content":TMPL.format(src=src, sent=s)}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(t, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=8, do_sample=False)
            r = tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True)[:14]
            if "근거없음" in r.replace(" ", ""):
                ungrounded.append(s)

        # ② 룰: 원문에 없는 괄호 설명
        parens = find_parens(ans, src)
        # ③ 룰: 메타 발화
        metas = [(m.group(0), why) for pat, why in META
                 for m in [re.search(pat, ans, re.M)] if m]

        gr = (len(sents)-len(ungrounded))/len(sents)*100 if sents else 100
        print(f"{'='*72}\n■ {gm} × {dname}\n{'='*72}")
        print(f"  [문장 근거율] {gr:5.1f}%  ({len(sents)-len(ungrounded)}/{len(sents)})")
        for s in ungrounded: print(f"      ❌ 근거없음  {s[:64]}")
        print(f"  [괄호 설명 추가] {len(parens)}건")
        for p in parens: print(f"      ⚠️ {p[:70]}")
        print(f"  [메타 발화] {len(metas)}건")
        for txt, why in metas: print(f"      ⚠️ '{txt}' — {why}")
        total = len(ungrounded)+len(parens)+len(metas)
        print(f"  >> 환각 의심 총 {total}건  ({time.time()-t0:.0f}초)\n", flush=True)
        report[gm][dname] = {"근거율":gr, "근거없음":ungrounded, "괄호":parens,
                             "메타":[t for t,_ in metas], "총건수":total}

print(f"{'='*72}\n[종합]\n{'='*72}")
print(f"  {'생성모델 × 문서':30s}{'근거율':>9s}{'괄호':>6s}{'메타':>6s}{'합계':>6s}")
for gm in report:
    for dn in report[gm]:
        r = report[gm][dn]
        print(f"  {gm+' × '+dn:30s}{r['근거율']:8.1f}%{len(r['괄호']):>6d}{len(r['메타']):>6d}{r['총건수']:>6d}")

json.dump(report, open("/home/hufs/이윤우/halluc_report.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/halluc_report.json")
