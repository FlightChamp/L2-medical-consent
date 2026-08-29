"""환각 탐지기 v3 — 인용 관련성 검증 (v2 버그 수정)"""
import torch, time, json, re
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

META = [(r"필요하시?면","사용자 제안"),(r"드립니다|드릴\s*수\s*있|드리겠","화자 개입"),
        (r"도움이\s*되(었|시)","화자 개입"),(r"이해할\s*수\s*있도록\s*했","자기 작업 언급"),
        (r"다시\s*(정리|작성)해","자기 작업 언급"),(r"원문에\s*있는\s*모든","자기 작업 언급"),
        (r"물론입니다","인사말")]
PAREN_SKIP = r"간단한?\s*설명|쉬운\s*말|요약|정리|안내|동의서|설명서"

def split_sents(t):
    t = re.sub(r"\*\*|^\s*[-–•]\s*|^#+\s*|^-{3,}$", " ", t, flags=re.M)
    return [s for s in
            (re.sub(r"\s+"," ",p).strip(" -–—:") for p in re.split(r"(?<=[.!?])\s+|\n+", t))
            if len(s) >= 12 and re.search(r"[가-힣]", s)]

def find_parens(ans, src):
    hits=[]
    for m in re.finditer(r"([가-힣A-Za-z]{2,12})\s*\(([^)]{4,60})\)", ans):
        term, expl = m.group(1), m.group(2)
        if re.search(PAREN_SKIP, expl) or re.search(PAREN_SKIP, term): continue
        if re.match(r"^[\d\s~\-–.%]+$", expl): continue
        toks = re.findall(r"[가-힣]{2,}", expl)
        if toks and sum(1 for w in toks if w in src)/len(toks) < 0.5:
            hits.append(f"{term}({expl})")
    return hits

def words(s): return set(re.findall(r"[가-힣]{2,}|\d+", s))
def norm(s):  return re.sub(r"[^가-힣A-Za-z0-9%]", "", s)

SYS = "당신은 의료 문서 검수자입니다. 지정된 형식으로만 답합니다."
TMPL = """[원문]
{src}

[검사할 문장]
{sent}

검사할 문장을 뒷받침하는 원문의 문장을 그대로 옮겨 적으세요.
원문에 해당 내용이 없으면 '근거없음'이라고만 적으세요.
설명을 덧붙이지 말고 원문 문장만 적으세요."""

prev = json.load(open(PREV))
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()
print("[환각 탐지기 v3 — 인용 관련성 검증]\n", flush=True)

report={}
for gm in prev:
    report[gm]={}
    for dname, src in SOURCES.items():
        ans = prev[gm][dname]["출력"]; sents = split_sents(ans)
        src_n = norm(src); bad=[]; t0=time.time()
        for s in sents:
            msgs=[{"role":"system","content":SYS},
                  {"role":"user","content":TMPL.format(src=src, sent=s)}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(t, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=96, do_sample=False)
            cite = tok.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            cite = re.sub(r'^["\'\s]*근거\s*[:：]?\s*', '', cite)          # ★ v2 버그 수정
            cite = cite.split("\n")[0].strip(' "\'-–—')

            if "근거없음" in cite.replace(" ",""):
                bad.append((s,"판사: 근거없음")); continue
            if len(norm(cite)) < 8:
                bad.append((s,"인용 없음")); continue
            if norm(cite)[:18] not in src_n:
                bad.append((s,f"원문에 없는 인용: {cite[:36]}")); continue
            # ★ 인용 관련성 — 검사 문장과 인용문의 내용어 겹침
            ws, wc = words(s), words(cite)
            ov = len(ws & wc)/len(ws) if ws else 0
            if ov < 0.15:
                bad.append((s,f"인용 무관(겹침 {ov*100:.0f}%): {cite[:32]}"))

        parens = find_parens(ans, src)
        metas  = [(m.group(0),why) for pat,why in META for m in [re.search(pat,ans,re.M)] if m]
        gr = (len(sents)-len(bad))/len(sents)*100 if sents else 100

        print(f"{'='*72}\n■ {gm} × {dname}\n{'='*72}")
        print(f"  [문장 근거율] {gr:5.1f}%  ({len(sents)-len(bad)}/{len(sents)})")
        for s,why in bad: print(f"      ❌ {s[:50]}\n         └ {why}")
        print(f"  [지어낸 용어 설명] {len(parens)}건")
        for p in parens: print(f"      ⚠️ {p[:68]}")
        print(f"  [메타 발화] {len(metas)}건")
        for txt,why in metas: print(f"      ⚠️ '{txt}' — {why}")
        print(f"  >> 환각 의심 총 {len(bad)+len(parens)+len(metas)}건  ({time.time()-t0:.0f}초)\n", flush=True)
        report[gm][dname]={"근거율":gr,"근거없음":[s for s,_ in bad],
                           "괄호":parens,"메타":[t for t,_ in metas],
                           "총건수":len(bad)+len(parens)+len(metas)}

print(f"{'='*72}\n[종합]\n{'='*72}")
print(f"  {'생성모델 × 문서':30s}{'근거율':>9s}{'괄호':>6s}{'메타':>6s}{'합계':>6s}")
for gm in report:
    for dn in report[gm]:
        r=report[gm][dn]
        print(f"  {gm+' × '+dn:30s}{r['근거율']:8.1f}%{len(r['괄호']):>6d}{len(r['메타']):>6d}{r['총건수']:>6d}")
json.dump(report, open("/home/hufs/이윤우/halluc3_report.json","w"), ensure_ascii=False, indent=2)
print("\n저장: ~/이윤우/halluc3_report.json")
