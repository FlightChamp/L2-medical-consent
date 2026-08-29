#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_docs.py v2 — 동의서 PDF 점검 (형식 판별 + 키워드 가시성 판정)"""
import os, sys, json, glob, datetime, traceback

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        sys.exit("[ERROR] PyMuPDF 미설치 → pip install pymupdf")

DOCS_DIR = os.path.expanduser("~/이윤우/docs")
OUT_DIR  = os.path.join(DOCS_DIR, "_check")
KEYWORDS = ["갑상선", "나비모양"]
DPI_CROP, DPI_PAGE = 300, 150
DARK_LEVEL, VISIBLE_TH, BLANK_TH = 200, 0.020, 0.005
os.makedirs(OUT_DIR, exist_ok=True)

def ink_ratio(page, rect):
    r = fitz.Rect(rect)
    if r.is_empty or r.width <= 0 or r.height <= 0: return 0.0
    pix = page.get_pixmap(dpi=DPI_CROP, clip=r, colorspace=fitz.csGRAY, alpha=False)
    d = pix.samples
    return (sum(1 for b in d if b < DARK_LEVEL) / len(d)) if d else 0.0

def span_info(page, rect, kw):
    try: dd = page.get_text("dict")
    except Exception: return {}
    for blk in dd.get("blocks", []):
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                if kw in sp.get("text", "") and fitz.Rect(sp["bbox"]).intersects(rect):
                    c = sp.get("color", 0)
                    return {"font": sp.get("font"), "size": round(sp.get("size", 0), 1),
                            "color_rgb": [(c >> 16) & 255, (c >> 8) & 255, c & 255],
                            "text": sp.get("text", "")[:120]}
    return {}

report = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
          "docs_dir": DOCS_DIR, "files": [], "hits": [], "errors": []}
pdfs = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))

print("=" * 84)
print(f"[1] 파일 집계 — {DOCS_DIR}   /   총 {len(pdfs)}개")
print("=" * 84)
print(f"{'파일명':<16}{'형식':>10}{'PDF?':>7}{'쪽':>4}{'문자수':>9}{'폼필드':>7}{'레이어':>7}{'크기(KB)':>10}")
print("-" * 84)

for path in pdfs:
    base = os.path.basename(path); stem = os.path.splitext(base)[0]
    try:
        doc = fitz.open(path)
        fmt = (doc.metadata or {}).get("format", "?")
        is_pdf = doc.is_pdf
        chars = widgets = 0
        for pg in doc:
            chars += len(pg.get_text())
            try: widgets += sum(1 for _ in pg.widgets())
            except Exception: pass
        try: ocgs = len(doc.get_ocgs() or {})
        except Exception: ocgs = 0
        kb = os.path.getsize(path) / 1024
        print(f"{base:<16}{str(fmt)[:10]:>10}{('Y' if is_pdf else 'N'):>7}"
              f"{doc.page_count:>4}{chars:>9}{widgets:>7}{ocgs:>7}{kb:>10.1f}")
        report["files"].append({"file": base, "format": fmt, "is_pdf": is_pdf,
                                "pages": doc.page_count, "chars": chars,
                                "widgets": widgets, "ocgs": ocgs, "size_kb": round(kb, 1)})

        # 박스 그리기용 문서 (비-PDF는 PDF로 변환)
        draw_doc, converted = doc, False
        if not is_pdf:
            try:
                draw_doc = fitz.open("pdf", doc.convert_to_pdf()); converted = True
            except Exception:
                draw_doc = None

        for pno in range(doc.page_count):
            try:
                page = doc[pno]; found = []; idxs = []
                for kw in KEYWORDS:
                    try: rects = page.search_for(kw)
                    except Exception: rects = []
                    for i, r in enumerate(rects):
                        ratio = ink_ratio(page, r)
                        verdict = ("보임" if ratio >= VISIBLE_TH else
                                   "안 보임(숨은 텍스트 의심)" if ratio < BLANK_TH else
                                   "애매(PNG 확인)")
                        line = fitz.Rect(page.rect.x0 + 5, r.y0 - 8,
                                         page.rect.x1 - 5, r.y1 + 8) & page.rect
                        cname = f"{stem}_p{pno+1}_{kw}_{i}_crop.png"
                        page.get_pixmap(dpi=DPI_CROP, clip=line).save(os.path.join(OUT_DIR, cname))
                        idxs.append(len(report["hits"]))
                        report["hits"].append({"file": base, "page": pno + 1, "keyword": kw,
                            "bbox": [round(v, 1) for v in (r.x0, r.y0, r.x1, r.y1)],
                            "ink_ratio": round(ratio, 4), "verdict": verdict,
                            "crop_png": cname, "converted_for_draw": converted,
                            **span_info(page, r, kw)})
                        found.append(r)
                if found:
                    fname = f"{stem}_p{pno+1}_full.png"
                    try:
                        if draw_doc is not None:
                            dp = draw_doc[pno]
                            for r in found: dp.draw_rect(r, color=(1, 0, 0), width=1.2)
                            dp.get_pixmap(dpi=DPI_PAGE).save(os.path.join(OUT_DIR, fname))
                        else:
                            page.get_pixmap(dpi=DPI_PAGE).save(os.path.join(OUT_DIR, fname))
                        for k in idxs: report["hits"][k]["page_png"] = fname
                    except Exception as e:
                        report["errors"].append(f"{base} p{pno+1} 렌더 실패: {e}")
            except Exception as e:
                report["errors"].append(f"{base} p{pno+1}: {e}")
        doc.close()
    except Exception as e:
        print(f"{base:<16}  ** 처리 실패: {e}")
        report["errors"].append(f"{base}: {e}\n{traceback.format_exc(limit=2)}")

print("=" * 84)
print(f"[2] 키워드 검사 — {', '.join(KEYWORDS)}")
print("=" * 84)
if not report["hits"]:
    print("키워드 미발견")
for h in report["hits"]:
    print(f"- {h['file']} p.{h['page']}  '{h['keyword']}'  ink={h['ink_ratio']:.4f}  → {h['verdict']}")
    print(f"    bbox={h['bbox']}  폰트={h.get('font','?')} 크기={h.get('size','?')} 색RGB={h.get('color_rgb','?')}")
    if h.get("text"): print(f"    문맥: {h['text']}")
    print(f"    크롭: _check/{h['crop_png']}")
if report["errors"]:
    print("\n[경고]")
    for e in report["errors"]: print(" -", e.splitlines()[0])

with open(os.path.join(OUT_DIR, "check_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n[저장] " + os.path.join(OUT_DIR, "check_report.json"))
print("[저장] PNG → " + OUT_DIR)
