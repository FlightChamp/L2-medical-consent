#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kofih_termdict.py — KOFIH 자료에서 의료용어 대역 사전 구축

방침:
  KOFIH는 문장 단위 대역이 아니다. 같은 내용을 언어별로 따로 편집한 자료라
  문단 경계도 순서도 일치하지 않는다. 따라서 문장 1:1 정렬을 목표로 삼지 않는다.
  대신 「직접 병기된 용어쌍」을 우선 추출하고, 나머지는 근거 강도에 따라 등급을 매긴다.

신뢰도 등급:
  A  같은 블록 안에서 한국어 용어 바로 뒤 괄호에 외국어 병기
     예) 정중신경(median nerve), 건초(tenosynovium)
  B  외국어판에서도 같은 괄호 병기가 확인됨 (양방향 교차 확인)
  C  같은 페이지의 대응 블록 안에서만 함께 출현 (세부 문맥 일치)
  D  같은 권에 함께 등장하기만 함 → 확정하지 않고 후보로만 남김

전처리:
  · \u0001(SOH) 제어문자를 공백으로 치환
  · 페이지별로 한국어 블록을 합쳐 잘린 문맥을 복원 (문장쌍 생성 목적 아님)
  · 표지·목차·부록(무료진료소·예진표) 페이지 제외

최종 산출:
  우리 수술동의서 13건에 실제로 등장하는 의료용어만 대상으로
  KO-EN-ZH-VI 대응표를 만든다.
"""
import os, re, json, glob, argparse
from collections import defaultdict
try:
    import pymupdf as fitz
except ImportError:
    import fitz

BASE = os.path.expanduser("~/shared/data/kofih_sel")
HOME = os.path.expanduser("~/이윤우")
OUT  = os.path.join(HOME, "kofih_termdict.json")
TSV  = os.path.join(HOME, "kofih_termdict.tsv")

KO   = re.compile(r"[가-힣]")
LANG = {
 "en": re.compile(r"[A-Za-z]"),
 "zh": re.compile(r"[\u4e00-\u9fff]"),
 "vi": re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"),
}
# 부록·목차·표지 판별
SKIP_PAGE = re.compile(r"무료진료소|예방접종\s*예진표|간행물번호|발\s*행\s*처|Phụ lục|PHỤ LỤC|목\s*차")
MAXSIZE = 13.5

ap = argparse.ArgumentParser()
ap.add_argument("--vols", default="all", help="쉼표 구분 또는 all")
ap.add_argument("--show", type=int, default=25)
args = ap.parse_args()

def clean(t):
    t = t.replace("\u0001", " ").replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip(" •·▶●■\u25aa-–—▪")

def read_pages(path, code):
    """페이지별 (한국어 텍스트, 외국어 텍스트, 블록목록) — 부록·표지 제외"""
    doc = fitz.open(path); pages = []
    for i in range(doc.page_count):
        raw = clean(doc[i].get_text())
        if SKIP_PAGE.search(raw) or len(raw) < 60:
            pages.append(None); continue
        ko_parts, fo_parts, blocks = [], [], []
        for blk in doc[i].get_text("dict").get("blocks", []):
            if "lines" not in blk: continue
            txt, sizes = "", []
            for ln in blk["lines"]:
                for sp in ln["spans"]:
                    txt += sp["text"]; sizes.append(sp["size"])
                txt += " "
            t = clean(txt)
            if len(t) < 8 or not sizes: continue
            if sum(sizes)/len(sizes) > MAXSIZE: continue
            k, f = len(KO.findall(t)), len(LANG[code].findall(t))
            blocks.append({"t": t, "ko": k, "fo": f,
                           "y": blk["bbox"][1], "x0": blk["bbox"][0], "x1": blk["bbox"][2]})
            if k >= 4: ko_parts.append(t)
            if f >= 6: fo_parts.append(t)
        pages.append({"ko": " ".join(ko_parts), "fo": " ".join(fo_parts), "blocks": blocks})
    doc.close()
    return pages

# ── A등급: 한국어 용어 바로 뒤 괄호 병기 ──
PAREN = re.compile(r"([가-힣]{2,12})\s*\(\s*([A-Za-z][A-Za-z \-/']{2,40}|[\u4e00-\u9fff]{1,12}"
                   r"|[A-Za-zăâđêôơư][^)]{2,40})\s*\)")
def paren_pairs(text, code):
    out = []
    for m in PAREN.finditer(text):
        ko, fo = m.group(1).strip(), m.group(2).strip()
        if len(LANG[code].findall(fo)) < 2: continue
        if re.fullmatch(r"[0-9\s\.,%]+", fo): continue
        out.append((ko, fo))
    return out

# ── 우리 동의서 용어 목록 ──
def consent_terms():
    rep = {}
    for f in ("reportA.json", "reportB.json", "reportC.json"):
        p = os.path.join(HOME, f)
        if os.path.exists(p): rep.update(json.load(open(p, encoding="utf-8")))
    seed = set()
    for v in rep.values():
        for w in re.findall(r"[가-힣]{2,10}", v["원문"]): seed.add(w)
    STOP = re.compile(r"(합니다|습니다|입니다|하는|되는|있는|없는|경우|위하|대하|따라|"
                      r"통하|또는|그리고|환자|의사|설명|동의|서명|확인|기재|해당|이상|"
                      r"이하|가능|필요|사항|내용|방법|결과|시행|실시|발생)")
    return {w for w in seed if len(w) >= 2 and not STOP.search(w)}

vols = sorted({m.group(1) for f in glob.glob(os.path.join(BASE, "*.pdf"))
               if (m := re.search(r"_(\d\d_[a-z]+)\.pdf$", f))})
targets = vols if args.vols == "all" else args.vols.split(",")
print(f"[편] {targets}\n")

TERMS = consent_terms()
print(f"[동의서 용어 후보] {len(TERMS)}개\n")

# grade → {ko: {code: {fo: count}}}
G = {g: defaultdict(lambda: defaultdict(lambda: defaultdict(int))) for g in "ABCD"}
STAT = defaultdict(int)

for vol in targets:
    print("=" * 76); print(f"[{vol}]")
    per = {}
    for code in ("en", "zh", "vi"):
        f = os.path.join(BASE, f"kofih_{code}_{vol}.pdf")
        if not os.path.exists(f): continue
        per[code] = read_pages(f, code)
        npage = sum(1 for p in per[code] if p)
        # A등급 — 한국어 텍스트 안의 괄호 병기
        a = 0
        for p in per[code]:
            if not p: continue
            for ko, fo in paren_pairs(p["ko"], code):
                G["A"][ko][code][fo] += 1; a += 1
            # 외국어 텍스트 쪽 괄호 병기도 수집(교차 확인용)
            for ko, fo in paren_pairs(p["fo"], code):
                G["B"][ko][code][fo] += 1
        # C등급 — 같은 페이지 블록 안 동시 출현
        c = 0
        for p in per[code]:
            if not p: continue
            kt, ft = p["ko"], p["fo"]
            if not kt or not ft: continue
            for t in TERMS:
                if t in kt.replace(" ", ""):
                    for cand in re.findall(r"[A-Za-zăâđêôơưÀ-ỹ]{4,25}" if code != "zh"
                                           else r"[\u4e00-\u9fff]{2,6}", ft):
                        G["C"][t][code][cand] += 1; c += 1
        print(f"  {code}: 본문 {npage}쪽 / 괄호병기 {a}건 / 페이지 동시출현 {c}건")
    print()

# B등급 확정 — A와 B 양쪽에서 확인된 것
final = {}
for ko in set(list(G["A"].keys()) + list(G["B"].keys())):
    entry = {}
    for code in ("en", "zh", "vi"):
        aset = G["A"].get(ko, {}).get(code, {})
        bset = G["B"].get(ko, {}).get(code, {})
        if not aset and not bset: continue
        best = max({**aset, **bset}.items(), key=lambda x: x[1])
        grade = "B" if (aset and bset) else "A"
        entry[code] = {"term": best[0], "grade": grade, "count": best[1]}
    if entry: final[ko] = entry

print("=" * 76)
print("[결과 — 직접 병기 용어쌍 (A/B등급)]")
print("=" * 76)
print(f"{'한국어':<14}{'EN':<30}{'ZH':<14}{'VI':<22}{'등급'}")
print("-" * 76)
shown = 0
for ko in sorted(final, key=lambda k: -len(final[k])):
    e = final[ko]
    if shown >= args.show: break
    g = "".join(sorted({v["grade"] for v in e.values()}))
    print(f"{ko:<14}{e.get('en',{}).get('term','—')[:28]:<30}"
          f"{e.get('zh',{}).get('term','—')[:12]:<14}"
          f"{e.get('vi',{}).get('term','—')[:20]:<22}{g}")
    shown += 1
print(f"\n총 {len(final)}개 용어 (A/B등급)")

# 동의서 용어와 교집합
hit = {k: v for k, v in final.items() if k in TERMS}
print(f"\n{'='*76}\n[동의서에 실제 등장하는 용어와의 교집합]  {len(hit)}개\n{'='*76}")
for ko in sorted(hit):
    e = hit[ko]
    print(f"  {ko:<12} EN {e.get('en',{}).get('term','—')[:26]:<28} "
          f"ZH {e.get('zh',{}).get('term','—')[:10]:<12} "
          f"VI {e.get('vi',{}).get('term','—')[:20]}")

# C/D 후보 — 확정하지 않음
cand = {}
for ko, per in G["C"].items():
    if ko in final: continue
    e = {}
    for code, m in per.items():
        if not m: continue
        top = sorted(m.items(), key=lambda x: -x[1])[:3]
        e[code] = [{"term": t, "count": c, "grade": "C"} for t, c in top]
    if e: cand[ko] = e
print(f"\n{'='*76}\n[C등급 후보 — 확정 아님, 검토 필요]  {len(cand)}개\n{'='*76}")
for ko in list(sorted(cand))[:12]:
    e = cand[ko]
    vi = ", ".join(x["term"] for x in e.get("vi", [])[:2])
    en = ", ".join(x["term"] for x in e.get("en", [])[:2])
    print(f"  {ko:<10} EN후보: {en[:38]:<40} VI후보: {vi[:34]}")

json.dump({"확정_AB": final, "후보_C": cand, "동의서교집합": hit},
          open(OUT, "w"), ensure_ascii=False, indent=1)
with open(TSV, "w", encoding="utf-8") as f:
    f.write("한국어\t영어\t중국어\t베트남어\t등급\n")
    for ko in sorted(final):
        e = final[ko]
        g = "".join(sorted({v["grade"] for v in e.values()}))
        f.write(f"{ko}\t{e.get('en',{}).get('term','')}\t{e.get('zh',{}).get('term','')}"
                f"\t{e.get('vi',{}).get('term','')}\t{g}\n")
print(f"\n[저장] {OUT}\n[저장] {TSV}")
