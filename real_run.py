"""실제 동의서 검증 — 섹션 기반 추출 + 단순화 + 누락/환각 검증
   사용법: python real_run.py doc1 doc10        (확장자 없이 파일명)"""
import torch, time, json, re, sys, os, glob
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, GPU = "snuh/hari-q3-8b", 0
DOCDIR = os.path.expanduser("~/이윤우/docs")
targets = sys.argv[1:] if len(sys.argv)>1 else None

# ── 서식·표 제거, 설명 섹션만 추출 ──
SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")
DROP = re.compile(
 r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
 r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
 r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")

def extract_sections(raw):
    """번호 섹션(2. 목적, 3. 방법 …)의 본문만 모음"""
    lines = [re.sub(r"\s+"," ",l).strip() for l in raw.split("\n")]
    secs, cur, buf = [], None, []
    for l in lines:
        if not l: continue
        m = SEC.match(l)
        if m:
            if cur and buf: secs.append((cur, buf))
            cur, buf = f"{m.group(1)}. {m.group(2)}", []
            continue
        if cur is None: continue
        if DROP.search(l): continue
        if len(re.findall(r"[가-힣]", l)) < 5: continue
        buf.append(l)
    if cur and buf: secs.append((cur, buf))
    return secs

def split_sents(t):
    out=[]
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+"," ",p).strip(" -–—:·")
        p = re.sub(r"\[\s*\]|\(\s*\)", "", p).strip()
        if len(p) >= 12 and len(re.findall(r"[가-힣]", p)) >= 6: out.append(p)
    return out

STOP = set("""환자 수술 시술 검사 치료 경우 가능 필요 발생 대한 위해 통해 이후 이전 다음 아래 관련
설명 동의 내용 방법 사항 결과 상태 정도 이상 이하 미만 대해 위한 등의 또는 그리고 하지만 있습니다
없습니다 합니다 됩니다 입니다 병원 의사 의료진 서명 보호자 성명 날짜 기록 확인 이해 질문 답변
본인 가족 담당 예정 시행 실시 있으며 하며 그러나 따라서 대체방법 주의사항""".split())
def extract_terms(src,k=20):
    c={}
    for w in re.findall(r"[가-힣]{3,10}", src):
        if w in STOP: continue
        c[w]=c.get(w,0)+1
    return sorted(c, key=lambda w:(-len(w),-c[w]))[:k]

def words(s): return set(re.findall(r"[가-힣]{2,}|\d+", s))
def norm(s):  return re.sub(r"[^가-힣A-Za-z0-9%]","",s)
META=[(r"필요하시?면","사용자 제안"),(r"드립니다|드릴\s*수\s*있","화자 개입"),
      (r"도움이\s*되(었|시)","화자 개입"),(r"다시\s*(정리|작성)해","자기 작업 언급"),
      (r"원문에\s*있는\s*모든","자기 작업 언급"),(r"물론입니다","인사말")]
PAREN_SKIP=r"간단한?\s*설명|쉬운\s*말|요약|정리|안내|동의서|설명서"

tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": GPU})
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": GPU})
model.eval()

def gen(sp,up,mt=4096):
    msgs=[{"role":"system","content":sp},{"role":"user","content":up}]
    try:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=mt,do_sample=False)
    return tok.decode(o[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()

SYS_J="당신은 의료 문서 검수자입니다. 주어진 문장 하나만 판단하고 지정된 단어로만 답합니다."
T_FACT="""아래 [변환문]에 [확인할 사실]의 내용이 담겨 있습니까?

[변환문]
{out}

[확인할 사실]
{unit}

표현이 쉬운 말로 바뀌어도 같은 내용이면 '있음'입니다.
내용 자체가 나타나지 않으면 '없음'입니다.

있음 / 없음 중 하나만 답하세요."""
T_CITE="""[원문]
{src}

[검사할 문장]
{sent}

검사할 문장을 뒷받침하는 원문의 문장을 그대로 옮겨 적으세요.
원문에 해당 내용이 없으면 '근거없음'이라고만 적으세요.
설명을 덧붙이지 말고 원문 문장만 적으세요."""

files = sorted(glob.glob(os.path.join(DOCDIR,"*.txt")))
if targets:
    files = [f for f in files if os.path.basename(f)[:-4] in targets]
print(f"[대상] {len(files)}개: {[os.path.basename(f)[:-4] for f in files]}\n", flush=True)

ALL={}
for path in files:
    name=os.path.basename(path)[:-4]
    raw=open(path,encoding="utf-8").read()
    secs=extract_sections(raw)
    SRC="\n".join(f"{h}\n"+"\n".join(b) for h,b in secs)
    if len(SRC)<200:
        print(f"[건너뜀] {name} — 섹션 추출 {len(SRC)}자\n"); continue
    FACTS=split_sents(SRC); TERMS=extract_terms(SRC)

    print(f"\n{'#'*72}\n### {name}\n{'#'*72}")
    print(f"[섹션] {len(secs)}개: {', '.join(h for h,_ in secs)}")
    print(f"[원문] 정제 {len(SRC)}자 / 사실 {len(FACTS)}개 / 용어후보 {len(TERMS)}개")
    print(f"[용어후보] {', '.join(TERMS)}")
    print(f"\n--- 정제 원문 ---\n{SRC}\n", flush=True)

    print("="*72+"\n[STEP 1] 쉬운 말 변환\n"+"="*72, flush=True)
    t0=time.time()
    OUT=gen("당신은 한국어 의료 어시스턴트입니다. 원문에 없는 내용은 절대 추가하지 마세요.",
            "다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
            "수술 이름, 모든 수치, 발생 조건, 시간·기간 정보는 하나도 빠짐없이 유지해 주세요.\n\n"+SRC)
    print(OUT); print(f"\n-- {time.time()-t0:.1f}초 | {len(SRC)} → {len(OUT)}자 ({len(OUT)/len(SRC)*100:.0f}%)\n", flush=True)

    print("="*72+"\n[STEP 2] 누락 검증\n"+"="*72, flush=True)
    lost_f=[u for u in FACTS if "있음" not in gen(SYS_J,T_FACT.format(out=OUT,unit=u),6)[:12]]
    fr=(len(FACTS)-len(lost_f))/len(FACTS)*100
    print(f"  [사실 보존] {fr:5.1f}%  ({len(FACTS)-len(lost_f)}/{len(FACTS)})")
    for u in lost_f: print(f"      ❌ {u[:66]}")
    lost_t=[t for t in TERMS if t not in OUT]
    tr=(len(TERMS)-len(lost_t))/len(TERMS)*100
    print(f"  [용어 보존] {tr:5.1f}%  ({len(TERMS)-len(lost_t)}/{len(TERMS)})")
    for t in lost_t: print(f"      ⚠️ {t}")
    print(flush=True)

    print("="*72+"\n[STEP 3] 환각 검증\n"+"="*72, flush=True)
    osents=split_sents(OUT); src_n=norm(SRC); bad=[]
    for s in osents:
        c=gen(SYS_J,T_CITE.format(src=SRC,sent=s),96)
        c=re.sub(r'^["\'\s]*근거\s*[:：]?\s*','',c).split("\n")[0].strip(' "\'-–—')
        if "근거없음" in c.replace(" ",""): bad.append((s,"판사: 근거없음")); continue
        if len(norm(c))<8: bad.append((s,"인용 없음")); continue
        if norm(c)[:18] not in src_n: bad.append((s,f"원문에 없는 인용: {c[:32]}")); continue
        ws,wc=words(s),words(c); ov=len(ws&wc)/len(ws) if ws else 0
        if ov<0.15: bad.append((s,f"인용 무관(겹침 {ov*100:.0f}%)"))
    parens=[]
    for m in re.finditer(r"([가-힣A-Za-z]{2,12})\s*\(([^)]{4,60})\)",OUT):
        term,expl=m.group(1),m.group(2)
        if re.search(PAREN_SKIP,expl) or re.search(PAREN_SKIP,term): continue
        if re.match(r"^[\d\s~\-–.%]+$",expl): continue
        tk=re.findall(r"[가-힣]{2,}",expl)
        if tk and sum(1 for w in tk if w in SRC)/len(tk)<0.5: parens.append(f"{term}({expl})")
    metas=[(m.group(0),why) for pat,why in META for m in [re.search(pat,OUT,re.M)] if m]
    gr=(len(osents)-len(bad))/len(osents)*100 if osents else 100
    print(f"  [문장 근거율] {gr:5.1f}%  ({len(osents)-len(bad)}/{len(osents)})")
    for s,why in bad: print(f"      ❌ {s[:50]}\n         └ {why}")
    print(f"  [지어낸 용어 설명] {len(parens)}건")
    for p in parens: print(f"      ⚠️ {p[:68]}")
    print(f"  [메타 발화] {len(metas)}건")
    for t,why in metas: print(f"      ⚠️ '{t}' — {why}")
    print(f"\n  >> 사실 {fr:.1f}% / 용어 {tr:.1f}% / 근거율 {gr:.1f}% / 환각의심 {len(bad)+len(parens)+len(metas)}건\n", flush=True)

    ALL[name]={"섹션":[h for h,_ in secs],"원문자수":len(SRC),"변환자수":len(OUT),
               "사실보존":fr,"용어보존":tr,"근거율":gr,"사실소실":lost_f,"용어소실":lost_t,
               "용어후보":TERMS,"환각":[s for s,_ in bad],"괄호":parens,
               "메타":[t for t,_ in metas],"원문":SRC,"변환문":OUT}

print(f"\n{'='*72}\n[전체 요약]\n{'='*72}")
print(f"  {'문서':16s}{'사실':>8s}{'용어':>8s}{'근거율':>9s}{'환각':>6s}{'길이비':>8s}")
for n,r in ALL.items():
    h=len(r['환각'])+len(r['괄호'])+len(r['메타'])
    print(f"  {n:16s}{r['사실보존']:7.1f}%{r['용어보존']:7.1f}%{r['근거율']:8.1f}%{h:>6d}{r['변환자수']/r['원문자수']*100:7.0f}%")
json.dump(ALL,open(os.path.expanduser("~/이윤우/realdoc_report.json"),"w"),ensure_ascii=False,indent=2)
print("\n저장: ~/이윤우/realdoc_report.json")
