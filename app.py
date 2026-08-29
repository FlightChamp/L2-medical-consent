#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""app.py v2 — L2 동의서 지원 플랫폼 (결과 뷰어)"""
import os, re, json
import streamlit as st

HOME = os.path.expanduser("~/이윤우")
st.set_page_config(page_title="L2 동의서 지원 플랫폼", layout="wide")

@st.cache_data
def load():
    def j(f):
        p = os.path.join(HOME, f)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rep = {}
    for f in ("reportA.json","reportB.json","reportC.json"): rep.update(j(f))
    return rep, j("translate_eval.json"), j("translate_demo.json"), j("nli_eval2_result.json")

REP, TE, TD, NLI = load()
if not REP: st.error("결과 파일 없음"); st.stop()

TITLE = {"doc1":"골절 수술 동의서","doc2":"관절경적 견봉성형술 동의서","doc3":"담낭제거 수술동의서",
 "doc4":"마취 동의서","doc5":"서혜부 탈장 수술동의서","doc7":"시술 및 검사 동의서(기흉)",
 "doc8":"시술 및 검사 동의서(유방의 결절)","doc9":"인공관절치환술(고관절) 동의서",
 "doc10":"인공관절치환술(슬관절) 동의서","doc11":"전/후방 십자인대 재건술 동의서",
 "doc12":"척추 신경 차단술 동의서","doc13":"치핵 수술 동의서","doc14":"항문주위 농양 수술 동의서"}
LANGNAME = {"en":"English","zh":"中文","ja":"日本語"}

# 사전 보정 — 같은 뜻의 다른 표현을 허용
TERM_ALT = {
 "신장":{"en":["kidney","renal","nephro"],"zh":["肾脏","肾"],"ja":["腎臓","腎"]},
 "관절":{"en":["joint","articular","arthro"],"zh":["关节"],"ja":["関節"]},
 "척추":{"en":["spinal","spine","vertebra"],"zh":["脊椎","脊柱","脊髓"],"ja":["脊椎","脊髄"]},
 "절제":{"en":["resect","excis","remov","ectomy"],"zh":["切除"],"ja":["切除"]},
 "봉합":{"en":["sutur","clos","repair"],"zh":["缝合"],"ja":["縫合"]},
 "후유증":{"en":["aftereffect","sequela","after-effect"],"zh":["后遗症"],"ja":["後遺症"]},
 "염증":{"en":["inflammation","inflammatory"],"zh":["炎症","发炎"],"ja":["炎症"]},
 "조직검사":{"en":["biopsy","histolog","patholog","tissue exam"],"zh":["活检","组织检查"],"ja":["生検","組織検査"]},
 "재발":{"en":["recurrence","recur","relapse"],"zh":["复发"],"ja":["再発"]},
 "방광":{"en":["bladder","abdomen"],"zh":["膀胱","下腹"],"ja":["膀胱","下腹"]},
 "유방":{"en":["breast","mammar"],"zh":["乳房","乳腺"],"ja":["乳房","乳腺"]},
 "담낭":{"en":["gallbladder","cholecyst"],"zh":["胆囊"],"ja":["胆囊","胆のう"]},
 "전립선":{"en":["prostate","prostatic"],"zh":["前列腺"],"ja":["前立腺"]},
}
# 원문 오염 — 문서 무결성 검사 결과
CONTAM = {"doc1": ("갑상선은 목 앞부분에 위치한 나비모양의 기관",
                   "골절 수술 동의서인데 갑상선 수술 설명이 인쇄되어 있습니다. "
                   "같은 병원의 갑상선 동의서에서 복사된 것으로 추정됩니다.")}
BLANK_RE = re.compile(r"약\s{2,}정도|:\s*$|\(\s*\)|\[\s*\]")

def sents(t):
    out=[]
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\*+","",re.sub(r"\s+"," ",p)).strip(" -–—:·#")
        if len(p)>=12 and len(re.findall(r"[가-힣]",p))>=6: out.append(p)
    return out
META_RE=[re.compile(p) for p in [r"^아래는",r"물론입니다",r"필요하시?면",r"드릴\s*수\s*있",
 r"원문에?\s*있는\s*모든",r"원문의?\s*모든\s*내용",r"다시\s*(작성|정리)한",r"추가\s*내용은?\s*없",
 r"이해할\s*수\s*있도록\s*(간단|명확)",r"^수술\s*동의서\s*\(",r"도움이\s*되"]]
is_meta=lambda s: any(r.search(s) for r in META_RE)

st.sidebar.title("L2 동의서 지원 플랫폼")
st.sidebar.caption("경기도의료원 공개 동의서 13건")
docs=sorted(REP.keys(), key=lambda x:(len(x),x))
doc=st.sidebar.selectbox("동의서 선택", docs, format_func=lambda d:f"{d} · {TITLE.get(d,'')}")
show_meta=st.sidebar.checkbox("챗봇 말투 표시", value=True)
st.sidebar.divider()
st.sidebar.caption("변환 · snuh/hari-q3-8b")
st.sidebar.caption("검증 · mDeBERTa NLI (SummaC 방식)")
st.sidebar.caption("용어 · AI Hub 의료 말뭉치 사전")

SRC, OUT = REP[doc]["원문"], REP[doc]["변환문"]
nli = NLI.get(doc, {})
g = lambda *keys: next((nli[k] for k in keys if k in nli), 0)

st.title(TITLE.get(doc, doc))

# 지표 (키 이름 여러 후보 대응)
c=st.columns(5)
c[0].metric("원문 문장", g("원문","원문문장") or len(sents(SRC)))
c[1].metric("변환 문장", g("변환","변환문장") or len(sents(OUT)))
c[2].metric("근거율", f"{g('C근거','근거율'):.1f}%", help="변환문 문장 중 원문에 근거가 있는 비율")
c[3].metric("무근거", f"{g('C무근거','무근거율'):.1f}%", help="원문에서 근거를 찾지 못한 비율")
c[4].metric("챗봇 말투", f"{g('메타')}건", help="문서 내용이 아닌 모델의 덧붙임")

uns=nli.get("무근거문장",[]); con=nli.get("모순문장",[])

# ① 원문 무결성
st.subheader("① 원문 무결성 검사")
if doc in CONTAM:
    kw, msg = CONTAM[doc]
    st.error(f"**타 진료과 내용 혼입** — {msg}")
    hit=[l for l in SRC.split("\n") if kw[:10] in l]
    if hit: st.code(hit[0].strip(), language=None)
else:
    st.success("원문 오염 미검출")
blanks=[l.strip() for l in SRC.split("\n") if BLANK_RE.search(l) and len(l.strip())>4][:5]
if blanks:
    st.warning("**빈칸 감지** — 모델이 임의로 채울 수 있는 자리입니다\n\n" +
               "\n".join(f"- `{b[:70]}`" for b in blanks))

# ② 변환 검증
st.subheader("② 변환 검증")
if uns or con:
    for s in uns: st.warning(f"**원문에 근거 없음** — {s}")
    for s in con: st.error(f"**원문과 다름** — {s}")
else:
    st.success("검증 경고 없음")

st.divider()
t1,t2=st.tabs(["원문 · 쉬운 말","번역"])
with t1:
    a,b=st.columns(2)
    with a:
        st.subheader("원문")
        st.text_area("원문", SRC, height=620, label_visibility="collapsed")
    with b:
        st.subheader("쉬운 말로 변환")
        out=[]
        for s in sents(OUT):
            if is_meta(s):
                if show_meta: out.append(f"🗨️ *{s}*")
            elif s in uns: out.append(f"⚠️ **{s}**")
            elif s in con: out.append(f"🔴 **{s}**")
            else: out.append(s)
        st.markdown("\n\n".join(out))

with t2:
    tr=TE.get(doc,{}).get("언어",{})
    vi=TD.get("Vietnamese",{}).get("출력") if doc=="doc1" else None
    codes=[c for c in ("en","zh","ja") if c in tr]
    names=[LANGNAME[c] for c in codes]+(["Tiếng Việt"] if vi else [])
    if not names: st.info("번역 결과 없음")
    else:
        for tab,name in zip(st.tabs(names),names):
            with tab:
                if name=="Tiếng Việt":
                    st.error("**의학 용어 오역** — 「갑상선」이 「Thực quản(식도)」로 번역됨. "
                             "올바른 표현은 「tuyến giáp」입니다.")
                    st.text_area("vi", vi, height=560, label_visibility="collapsed")
                else:
                    code=codes[names.index(name)]; v=tr[code]; txt=v["출력"]; low=txt.lower()
                    real=[]      # 사전 보정 후에도 진짜 미검출인 것
                    for term,corr in v.get("미검출",[]):
                        alts=TERM_ALT.get(term,{}).get(code,[])
                        if not any(a.lower() in low for a in alts): real.append((term,corr))
                    hit=len(v.get("적중",[]))+ (len(v.get("미검출",[]))-len(real))
                    tot=len(v.get("적중",[]))+len(v.get("미검출",[]))
                    m=st.columns(3)
                    m[0].metric("용어 검출", f"{hit} / {tot}", help="사전 보정 후")
                    m[1].metric("한글 잔존", f"{v.get('한글잔존',0)}자")
                    m[2].metric("자동 미검출", f"{len(v.get('미검출',[]))}건",
                                help="보정 전 원자료")
                    if real:
                        st.warning("**확인 필요 용어** — " +
                                   ", ".join(f"{a}→{b}" for a,b in real))
                    elif v.get("미검출"):
                        st.info("자동 미검출 " + ", ".join(f"{a}" for a,_ in v["미검출"]) +
                                " — 사전에 없는 다른 표현을 사용한 것으로 확인됨")
                    st.text_area(code, txt, height=520, label_visibility="collapsed")

st.divider()
st.caption("검증 · NLI 함의 판정(SummaC 방식, mDeBERTa-v3-xnli) · "
           "용어 대응 검출(AI Hub 의료 말뭉치 사전) · 문서 무결성 검사")
