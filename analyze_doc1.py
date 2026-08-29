#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_doc1.py — 저장된 결과만으로 지표 재계산 (GPU/모델 불필요)"""
import os, re, json

SRC_JSON = os.path.expanduser("~/이윤우/doc1_test_report.json")
OUT_MD   = os.path.expanduser("~/이윤우/doc1_analysis.md")

data = json.load(open(SRC_JSON, encoding="utf-8"))
name = list(data.keys())[0]
r    = data[name]
SRC, OUT = r["원문"], r["변환문"]
TERMS    = r["용어후보"]

def nsp(s): return re.sub(r"\s+", "", s)          # 공백 제거
SRC_N, OUT_N = nsp(SRC), nsp(OUT)

def split_sents(t):
    out=[]
    for p in re.split(r"(?<=[.!?])\s+|\n+", t):
        p = re.sub(r"\s+"," ",p).strip(" -–—:·")
        p = re.sub(r"\[\s*\]|\(\s*\)", "", p).strip()
        if len(p) >= 12 and len(re.findall(r"[가-힣]", p)) >= 6: out.append(p)
    return out

# ── A. 용어 보존 3가지 기준 ──
def match_exact(t):  return t in OUT
def match_space(t):  return nsp(t) in OUT_N
def match_stem(t):
    """조사·어미를 최대 3음절까지 떼며 접두 매칭"""
    b = nsp(t)
    for cut in range(0, 4):
        if len(b) - cut < 3: break
        if b[:len(b)-cut] in OUT_N: return True
    return False

rows = []
for t in TERMS:
    e, s, m = match_exact(t), match_space(t), match_stem(t)
    rows.append((t, e, s, m))

def rate(i): return sum(1 for x in rows if x[i]) / len(rows) * 100

print("=" * 74)
print("[A] 용어 보존율 — 판정 기준별 비교")
print("=" * 74)
print(f"{'용어':<16}{'원본방식':>10}{'공백무시':>10}{'어간매칭':>10}")
print("-" * 74)
for t, e, s, m in rows:
    f = lambda b: "O" if b else "X"
    mark = "   ← 진짜 소실" if not m else ""
    print(f"{t:<16}{f(e):>10}{f(s):>10}{f(m):>10}{mark}")
print("-" * 74)
print(f"{'보존율':<16}{rate(1):>9.1f}%{rate(2):>9.1f}%{rate(3):>9.1f}%")
real_lost = [t for t, e, s, m in rows if not m]
print(f"\n진짜 소실 {len(real_lost)}건: {', '.join(real_lost) if real_lost else '없음'}")

# ── B. 숫자 환각 ──
NUM = re.compile(r"(\d[\d,.]*)\s*(시간|분|일|주일|주|개월|개월간|년|세|%|퍼센트|cc|ml|mL|mg|g|cm|mm|회|번|명|개)?")
def nums(text):
    out=[]
    for m in NUM.finditer(text):
        v, u = m.group(1), m.group(2) or ""
        ctx = " ".join(text[max(0,m.start()-25):m.end()+25].split())
        out.append((v+u, ctx))
    return out

src_nums = {v for v, _ in nums(SRC)}
out_nums = nums(OUT)
added = [(v, c) for v, c in out_nums if v not in src_nums and nsp(v) not in SRC_N]

print("\n" + "=" * 74)
print("[B] 숫자 환각 검사 — 변환문에만 있는 수치")
print("=" * 74)
print(f"원문 수치 {len(src_nums)}종 / 변환문 수치 {len(out_nums)}건 / 신규 등장 {len(added)}건")
seen=set()
for v, c in added:
    if v in seen: continue
    seen.add(v)
    print(f"  ❌ '{v}'  …{c}…")
if not added: print("  신규 수치 없음")

# ── C. 근거율 재계산 (마크다운 서식 제외) ──
HEADER = re.compile(r"^\*\*.*\*\*$|^#{1,6}\s|^-+$|^\s*\*\*[^*]+\*\*\s*$")
osents = split_sents(OUT)
bad    = set(r["환각"])
def is_header(s):
    s2 = s.strip()
    if HEADER.match(s2): return True
    if s2.count("**") >= 2 and not re.search(r"[다요]\.$|입니다|합니다|됩니다", s2): return True
    return False
real_sents = [s for s in osents if not is_header(s)]
real_bad   = [s for s in real_sents if any(s.startswith(b[:30]) or b.startswith(s[:30]) for b in bad)]
gr_old = r["근거율"]
gr_new = (len(real_sents)-len(real_bad))/len(real_sents)*100 if real_sents else 0

print("\n" + "=" * 74)
print("[C] 근거율 재계산 — 마크다운 제목 줄 제외")
print("=" * 74)
print(f"  전체 문장 {len(osents)}개 중 제목·서식 {len(osents)-len(real_sents)}개 제외")
print(f"  기존 근거율 {gr_old:.1f}%  →  재계산 {gr_new:.1f}%  ({len(real_sents)-len(real_bad)}/{len(real_sents)})")
print("\n  [남은 실제 지적]")
for s in real_bad: print(f"    ❌ {s[:64]}")

# ── D. 발표용 마크다운 저장 ──
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(f"# doc1 (골절수술동의서) 검증 결과\n\n")
    f.write(f"| 지표 | 원본 측정 | 보정 후 | 비고 |\n|---|---|---|---|\n")
    f.write(f"| 사실 보존 | {r['사실보존']:.1f}% | {r['사실보존']:.1f}% | 변동 없음 |\n")
    f.write(f"| 용어 보존 | {rate(1):.1f}% | {rate(3):.1f}% | 조사·띄어쓰기 오탐 제거 |\n")
    f.write(f"| 문장 근거율 | {gr_old:.1f}% | {gr_new:.1f}% | 마크다운 제목 줄 제외 |\n")
    f.write(f"| 길이비 | {len(OUT)/len(SRC)*100:.0f}% | - | - |\n\n")
    f.write(f"## 진짜 용어 소실 ({len(real_lost)}건)\n")
    for t in real_lost: f.write(f"- {t}\n")
    f.write(f"\n## 사실 소실 ({len(r['사실소실'])}건)\n")
    for s in r["사실소실"]: f.write(f"- {s}\n")
    f.write(f"\n## 숫자 환각 ({len(seen)}건) — 기존 검증기가 놓친 유형\n")
    for v, c in added:
        f.write(f"- `{v}` : …{c}…\n")
    f.write(f"\n## 메타 발화 ({len(r['메타'])}건)\n")
    for m in r["메타"]: f.write(f"- {m}\n")
print(f"\n[저장] {OUT_MD}  ← 발표 자료에 붙여넣기용")
