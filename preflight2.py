#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preflight2.py — real_run.py와 동일 입력(docs/*.txt)·동일 로직으로 절단 위험 점검"""
import os, re, glob, json
from transformers import AutoTokenizer

MODEL  = "snuh/hari-q3-8b"
DOCDIR = os.path.expanduser("~/이윤우/docs")
OUTDIR = os.path.join(DOCDIR, "_preflight2")
MAX_NEW = 2048          # real_run.py gen()의 기본값
RATIO   = 1.15          # 기존 실측 길이비 112~114% → 여유 포함
os.makedirs(OUTDIR, exist_ok=True)

# ── real_run.py 원본 로직 그대로 ──
SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")
DROP = re.compile(
 r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
 r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
 r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")

def extract_sections(raw):
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

txts = sorted(glob.glob(os.path.join(DOCDIR, "*.txt")))
print(f"[docs/*.txt] {len(txts)}개 발견: {[os.path.basename(t)[:-4] for t in txts]}\n")
if not txts:
    raise SystemExit("[중단] docs 안에 .txt가 없습니다 — real_run.py가 처리할 대상이 0개입니다.")

print("[토크나이저 로드 중…]")
tok = AutoTokenizer.from_pretrained(MODEL)
print()

hdr = f"{'문서':<10}{'섹션':>5}{'정제자수':>9}{'입력tok':>9}{'예상출력tok':>12}{'한도대비':>9}{'사실':>6}{'용어':>6}{'추정분':>8}"
print(hdr); print("-" * len(hdr))
rows = []
for path in txts:
    name = os.path.basename(path)[:-4]
    raw  = open(path, encoding="utf-8").read()
    secs = extract_sections(raw)
    SRC  = "\n".join(f"{h}\n" + "\n".join(b) for h, b in secs)
    if len(SRC) < 200:
        print(f"{name:<10}  [건너뜀 대상] 정제 {len(SRC)}자 — real_run.py도 스킵함")
        rows.append({"file": name, "skipped": True, "src_len": len(SRC)}); continue
    facts = split_sents(SRC); terms = extract_terms(SRC)
    n_in  = len(tok(SRC)["input_ids"])
    n_out = int(n_in * RATIO)
    pct   = n_out / MAX_NEW * 100
    flag  = "❌ 절단" if pct >= 100 else ("⚠ 위험" if pct >= 85 else "OK")
    mins  = (n_out/63 + len(facts)*1.5 + len(facts)*4.0) / 60
    print(f"{name:<10}{len(secs):>5}{len(SRC):>9}{n_in:>9}{n_out:>12}{pct:>7.0f}% {flag:<7}"
          f"{len(facts):>6}{len(terms):>6}{mins:>7.1f}")
    with open(os.path.join(OUTDIR, f"{name}_SRC.txt"), "w", encoding="utf-8") as f:
        f.write(SRC)
    rows.append({"file": name, "sections": [h for h,_ in secs], "src_len": len(SRC),
                 "in_tok": n_in, "est_out_tok": n_out, "pct_of_limit": round(pct,1),
                 "facts": len(facts), "terms": len(terms), "est_min": round(mins,1),
                 "flag": flag})

tot = sum(r.get("est_min", 0) for r in rows)
print("-" * len(hdr))
print(f"{'합계':<10}{'':>5}{'':>9}{'':>9}{'':>12}{'':>9}{'':>6}{'':>6}{tot:>7.1f}")

d1 = [r for r in rows if r["file"] == "doc1"]
p = os.path.join(OUTDIR, "doc1_SRC.txt")
if os.path.exists(p):
    t = open(p, encoding="utf-8").read()
    print("\n[doc1 갑상선 문장]", "입력에 포함됨 ✅" if "갑상선" in t else "입력에서 누락됨 ❌")

with open(os.path.join(OUTDIR, "preflight2.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\n[저장]", OUTDIR)
