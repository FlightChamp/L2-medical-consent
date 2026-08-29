#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kofih_extract.py v2 — KOFIH 대역 문장쌍 추출 (좌표 기반)

v1의 실패:
  블록을 읽은 순서대로 「외국어 다음이 한국어」로 짝지었더니
  한 칸씩 밀리고 목차·제목이 섞였다. 디자인 레이아웃이라
  블록 순서가 읽기 순서와 다르기 때문이다.

v2의 방법:
  · 페이지 안에서 외국어 문단과 한국어 문단의 좌표를 각각 구한다.
  · 한국어 문단은 대응 외국어 문단 바로 아래에 오므로,
    x가 겹치고 y가 아래쪽에서 가장 가까운 것을 짝으로 삼는다.
  · 한 줄에 두 언어가 섞인 줄(목차·제목)은 제외한다.
  · 본문 글자 크기 범위를 벗어난 블록(표지 대형 제목)도 제외한다.
"""
import os, re, json, glob, argparse
try:
    import pymupdf as fitz
except ImportError:
    import fitz

BASE = os.path.expanduser("~/shared/data/kofih_sel")
OUT  = os.path.expanduser("~/이윤우/kofih_pairs.json")
KO = re.compile(r"[가-힣]")
LANGRE = {
 "en": re.compile(r"[A-Za-z]"),
 "zh": re.compile(r"[\u4e00-\u9fff]"),
 "vi": re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"),
}
MAXSIZE = 13.0        # 본문 글자 크기 상한 (표지·대제목 제외)

ap = argparse.ArgumentParser()
ap.add_argument("--vol", default="02_os")
ap.add_argument("--all", action="store_true")
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()

def clean(t):
    t = t.replace("\u0001", " ").replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip(" •·▶●■\u25aa-–—▪")

def blocks_of(page, code):
    """(언어, 텍스트, x0, y0, x1, 평균글자크기) 목록"""
    out = []
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        if "lines" not in blk: continue
        txt, sizes = "", []
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                txt += sp["text"]; sizes.append(sp["size"])
            txt += "\n"
        t = clean(txt)
        if len(t) < 18 or not sizes: continue
        avg = sum(sizes) / len(sizes)
        if avg > MAXSIZE: continue                 # 표지·대제목 제외
        ko = len(KO.findall(t)); fo = len(LANGRE[code].findall(t))
        # 한 덩어리에 두 언어가 섞이면 목차·제목 → 제외
        if ko >= 4 and fo >= 6: continue
        if ko >= 6 and fo < 3:   L = "ko"
        elif fo >= 10 and ko < 2: L = code
        else: continue
        x0, y0, x1, y1 = blk["bbox"]
        out.append((L, t, x0, y0, x1, avg))
    return out

def extract(path, code, debug=False):
    doc = fitz.open(path); pairs = []
    for i in range(doc.page_count):
        bs = blocks_of(doc[i], code)
        fos = [b for b in bs if b[0] == code]
        kos = [b for b in bs if b[0] == "ko"]
        used = set()
        for f in fos:
            best, bestd = None, 1e9
            for j, k in enumerate(kos):
                if j in used: continue
                # x 구간이 겹치고 한국어가 아래쪽
                overlap = min(f[4], k[4]) - max(f[2], k[2])
                dy = k[3] - f[3]
                if overlap < 40 or dy <= 0 or dy > 320: continue
                if dy < bestd: best, bestd = j, dy
            if best is not None:
                used.add(best)
                pairs.append({"fo": f[1], "ko": kos[best][1], "page": i + 1})
        if debug and i in (5, 6, 7):
            print(f"    [p{i+1}] 외국어 {len(fos)} · 한국어 {len(kos)} → 매칭 {len(used)}")
    doc.close()
    seen, uniq = set(), []
    for p in pairs:
        k = p["ko"][:60]
        if k in seen: continue
        seen.add(k); uniq.append(p)
    return uniq

vols = sorted({m.group(1) for f in glob.glob(os.path.join(BASE, "*.pdf"))
               if (m := re.search(r"_(\d\d_[a-z]+)\.pdf$", f))})
targets = vols if args.all else [args.vol]
print(f"[편 목록] {vols}\n[처리 대상] {targets}\n")

RESULT = {}
for vol in targets:
    RESULT[vol] = {}
    print("=" * 78); print(f"[{vol}]")
    for code in ("en", "zh", "vi"):
        f = os.path.join(BASE, f"kofih_{code}_{vol}.pdf")
        if not os.path.exists(f): print(f"  {code}: 파일 없음"); continue
        pairs = extract(f, code, args.debug)
        RESULT[vol][code] = pairs
        kchar = sum(len(p["ko"]) for p in pairs)
        print(f"  {code}: 문장쌍 {len(pairs):>4}개 / 한국어 {kchar:,}자")
        for p in pairs[:2]:
            print(f"      KO : {p['ko'][:80]}")
            print(f"      {code.upper()} : {p['fo'][:80]}")
    print()

json.dump(RESULT, open(OUT, "w"), ensure_ascii=False, indent=1)
tot = sum(len(v) for d in RESULT.values() for v in d.values())
print(f"[저장] {OUT}   총 {tot}개 문장쌍")

print(f"\n{'='*78}\n[검증] 핵심 용어 대응\n{'='*78}")
CHECK = ["갑상선", "골절", "관절", "인대", "통증", "염증"]
for term in CHECK:
    print(f"\n── {term}")
    found = False
    for vol, per in RESULT.items():
        for code, pairs in per.items():
            for p in pairs:
                if term in p["ko"].replace(" ", ""):
                    print(f"   [{code}] KO: {p['ko'][:88]}")
                    print(f"        FO: {p['fo'][:88]}")
                    found = True; break
            if found: break
        if found: break
    if not found: print("   (없음)")
