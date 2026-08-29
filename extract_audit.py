#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_audit.py — ① 추출 단계 진단 (기존 txt 를 건드리지 않는다)
==================================================================
문제:
    현재 docs/*.txt 에 문서당 평균 11.6곳의 정보 유실이 있다.
    `골절된 의 정복 후` 처럼 명사가 빠지고 조사만 남은 자국이다.
    빈칸(서식상 비워둔 자리)과 달리 원문에 있던 정보가 사라진 것이라
    파이프라인 하류에서 복구가 불가능하다.

원인은 셋 중 하나이며 처방이 완전히 다르다:
    (A) PDF 에 텍스트 레이어가 없음        → OCR 필요 (범위가 커진다)
    (B) 추출기의 레이아웃 오독              → 추출 방식 교체
    (C) 후처리 코드 문제                    → 코드 수정 (가장 쉽다)

    (C) 일 가능성이 낮지 않다. `골절된`과 `의`가 살아있고 그 사이만 빠진 것은
    표 셀 경계나 폰트 전환 지점에서 조각이 유실된 형태로 보인다.

방법:
    같은 PDF 를 네 방식으로 추출해 현재 txt 와 대조한다.
      text    페이지 단순 텍스트 (현재 preflight.py 방식)
      blocks  레이아웃 블록 단위
      words   단어 단위. 좌표로 재조립하므로 유실이 가장 적다
      plumber pdfplumber. 표 인식 방식이 다르다

    각 방식에서 명사누락·빈괄호가 얼마나 줄어드는지 본다.

안전장치:
    기존 docs/*.txt 를 수정하지 않는다. 재추출 결과는 docs/_reextract/ 에 둔다.
    지금까지의 모든 실험 기준선이 유지된다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python extract_audit.py                    # 진단만
    python extract_audit.py --show doc1        # 특정 문서 방식별 비교
    python extract_audit.py --save             # 재추출 결과 파일로 저장
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

DOCDIR = os.path.expanduser("~/이윤우/docs")
REDIR = os.path.join(DOCDIR, "_reextract")

METHODS = ("text", "blocks", "words", "plumber")


# ===========================================================================
# 손실 탐지 (blank_guard.py 와 동일 규칙)
# ===========================================================================

LOSS = {
    "명사누락": re.compile(r"\s(을|를|이|가|의|은|는|에|와|과)\s"),
    "빈괄호": re.compile(r"[（(\[〔]\s*[)）\]〕]"),
    "수치누락": re.compile(
        r"약\s+(?=정도|가량|이내|이상|이하|동안|쯤)"
        r"|(?<![\d])\s(?=(주|일|개월|시간|분|년|%|cc|ml|mg|회)\s*(정도|가량|이내|동안))"),
    "고립조사": re.compile(r"(^|\s)(으로|에서|에게|부터|까지|이나|하여)(\s|$)"),
    "깨진문자": re.compile(r"[ᄀ-ᇿ]|[\ufffd]"),
}


def clean(t) -> str:
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


def count_loss(text: str) -> Dict[str, int]:
    return {k: len(p.findall(text)) for k, p in LOSS.items()}


def hangul(t: str) -> int:
    return len(re.findall(r"[가-힣]", t))


# ===========================================================================
# 추출 방식들
# ===========================================================================

class Extractor:
    def __init__(self):
        self.fitz = None
        self.plumber = None
        try:
            import pymupdf
            self.fitz = pymupdf
        except ImportError:
            try:
                import fitz
                self.fitz = fitz
            except ImportError:
                pass
        try:
            import pdfplumber
            self.plumber = pdfplumber
        except ImportError:
            pass
        if not self.fitz:
            sys.exit("[FATAL] PyMuPDF 필요: pip install pymupdf")
        print(f"[EXTRACT] PyMuPDF 사용 가능 / "
              f"pdfplumber {'사용 가능' if self.plumber else '미설치'}")

    def text(self, path: str) -> str:
        doc = self.fitz.open(path)
        out = "\n".join(p.get_text() for p in doc)
        doc.close()
        return out

    def blocks(self, path: str) -> str:
        doc = self.fitz.open(path)
        parts = []
        for pg in doc:
            bs = sorted(pg.get_text("blocks"), key=lambda b: (round(b[1]), b[0]))
            for b in bs:
                if len(b) > 4 and isinstance(b[4], str) and b[4].strip():
                    parts.append(b[4].strip())
        doc.close()
        return "\n".join(parts)

    def words(self, path: str, ytol: float = 3.0) -> str:
        """단어를 좌표로 재조립한다. 같은 줄이면 x 순으로 붙인다."""
        doc = self.fitz.open(path)
        parts = []
        for pg in doc:
            ws = pg.get_text("words")   # (x0,y0,x1,y1,word,block,line,wordno)
            if not ws:
                continue
            ws.sort(key=lambda w: (round(w[1] / ytol), w[0]))
            line_y, buf = None, []
            for w in ws:
                y = round(w[1] / ytol)
                if line_y is None or y == line_y:
                    buf.append(w[4])
                else:
                    parts.append(" ".join(buf))
                    buf = [w[4]]
                line_y = y
            if buf:
                parts.append(" ".join(buf))
        doc.close()
        return "\n".join(parts)

    def plumber_text(self, path: str) -> Optional[str]:
        if not self.plumber:
            return None
        parts = []
        with self.plumber.open(path) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                if t.strip():
                    parts.append(t)
        return "\n".join(parts)

    def run(self, path: str, method: str) -> Optional[str]:
        try:
            if method == "text":
                return self.text(path)
            if method == "blocks":
                return self.blocks(path)
            if method == "words":
                return self.words(path)
            if method == "plumber":
                return self.plumber_text(path)
        except Exception as e:
            print(f"    ({method} 실패: {type(e).__name__}: {e})")
        return None


# ===========================================================================

def pdf_info(ex: Extractor, path: str) -> dict:
    """텍스트 레이어 유무 판정 — OCR 필요 여부를 가른다."""
    doc = ex.fitz.open(path)
    n_pages = len(doc)
    chars = sum(len(p.get_text()) for p in doc)
    n_img = sum(len(p.get_images(full=True)) for p in doc)
    doc.close()
    return {"pages": n_pages, "chars": chars, "images": n_img,
            "chars_per_page": chars / max(n_pages, 1)}


def audit(a):
    ex = Extractor()
    pdfs = sorted(glob.glob(os.path.join(a.docdir, "*.pdf")))
    if not pdfs:
        sys.exit(f"[FATAL] {a.docdir}/*.pdf 없음")

    print("\n" + "=" * 100)
    print("1. PDF 상태 — 텍스트 레이어가 있는가")
    print("=" * 100)
    print(f"  {'문서':<8}{'쪽':>4}{'글자수':>9}{'쪽당':>8}{'이미지':>7}"
          f"{'txt존재':>9}  판정")
    no_text = []
    for p in pdfs:
        stem = os.path.splitext(os.path.basename(p))[0]
        info = pdf_info(ex, p)
        has_txt = os.path.exists(os.path.join(a.docdir, f"{stem}.txt"))
        if info["chars_per_page"] < 100:
            verdict = "텍스트 레이어 없음 → OCR 필요"
            no_text.append(stem)
        elif info["chars_per_page"] < 400:
            verdict = "텍스트 빈약"
        else:
            verdict = "정상"
        print(f"  {stem:<8}{info['pages']:>4}{info['chars']:>9}"
              f"{info['chars_per_page']:>8.0f}{info['images']:>7}"
              f"{'있음' if has_txt else '없음':>9}  {verdict}")
    if no_text:
        print(f"\n  [!] 텍스트 레이어 없는 문서 {len(no_text)}건: "
              f"{', '.join(no_text)} — 추출기 교체로는 해결되지 않습니다")

    print("\n" + "=" * 100)
    print("2. 현재 txt 의 손실 (기준선)")
    print("=" * 100)
    base = {}
    print(f"  {'문서':<8}{'글자':>8}{'한글':>8}"
          + "".join(f"{k:>9}" for k in LOSS))
    tot = Counter()
    for p in sorted(glob.glob(os.path.join(a.docdir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("syn"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            t = clean(f.read())
        c = count_loss(t)
        base[stem] = {"text": t, "loss": c}
        for k, v in c.items():
            tot[k] += v
        print(f"  {stem:<8}{len(t):>8}{hangul(t):>8}"
              + "".join(f"{c[k]:>9}" for k in LOSS))
    print(f"  {'합계':<8}{'':>8}{'':>8}" + "".join(f"{tot[k]:>9}" for k in LOSS))

    print("\n" + "=" * 100)
    print("3. 추출 방식별 비교 — 어느 방식이 손실이 적은가")
    print("=" * 100)
    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for p in pdfs:
        stem = os.path.splitext(os.path.basename(p))[0]
        for m in METHODS:
            t = ex.run(p, m)
            if t is None:
                continue
            t = clean(t)
            results[stem][m] = {"text": t, "chars": len(t),
                                "hangul": hangul(t), "loss": count_loss(t)}
        print(f"  {stem} 완료", end="  ", flush=True)
    print()

    print(f"\n  {'방식':<10}{'문서':>5}{'평균글자':>10}{'평균한글':>10}"
          + "".join(f"{k:>9}" for k in LOSS))
    method_tot = {}
    for m in ("현재txt",) + METHODS:
        if m == "현재txt":
            vals = list(base.values())
            if not vals:
                continue
            ch = sum(len(v["text"]) for v in vals) / len(vals)
            hg = sum(hangul(v["text"]) for v in vals) / len(vals)
            ls = {k: sum(v["loss"][k] for v in vals) for k in LOSS}
            n = len(vals)
        else:
            vals = [d[m] for d in results.values() if m in d]
            if not vals:
                continue
            ch = sum(v["chars"] for v in vals) / len(vals)
            hg = sum(v["hangul"] for v in vals) / len(vals)
            ls = {k: sum(v["loss"][k] for v in vals) for k in LOSS}
            n = len(vals)
        method_tot[m] = ls
        print(f"  {m:<10}{n:>5}{ch:>10.0f}{hg:>10.0f}"
              + "".join(f"{ls[k]:>9}" for k in LOSS))

    print("\n  현재txt 대비 변화 (음수가 개선)")
    cur = method_tot.get("현재txt")
    if cur:
        for m in METHODS:
            if m not in method_tot:
                continue
            d = {k: method_tot[m][k] - cur[k] for k in LOSS}
            print(f"    {m:<10}" + "".join(f"{d[k]:>+9}" for k in LOSS))

    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    if no_text:
        print(f"  (A) 텍스트 레이어 없음 {len(no_text)}건 — OCR 필요")
    if cur:
        best = min((m for m in METHODS if m in method_tot),
                   key=lambda m: method_tot[m]["명사누락"], default=None)
        if best:
            diff = method_tot[best]["명사누락"] - cur["명사누락"]
            print(f"  명사누락 최소 방식: {best} ({method_tot[best]['명사누락']}건, "
                  f"현재 대비 {diff:+d})")
            if diff <= -20:
                print("  (B) 추출 방식 교체로 유의미한 개선이 가능합니다.")
            elif diff >= -5:
                print("  (B) 방식 교체로는 거의 개선되지 않습니다.")
                print("      → 손실이 PDF 자체 또는 후처리 코드에 있을 가능성.")
                print("        4번 블록의 실제 문장 비교로 확인하세요.")
    print("=" * 100)

    if a.save:
        os.makedirs(REDIR, exist_ok=True)
        for stem, d in results.items():
            for m, v in d.items():
                with open(os.path.join(REDIR, f"{stem}__{m}.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(v["text"])
        print(f"\n[SAVE] {REDIR}/ 에 재추출 결과 저장 "
              f"(기존 docs/*.txt 는 그대로)")

    with open("extract_audit_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {k: v["loss"] for k, v in base.items()},
            "methods": {s: {m: v["loss"] for m, v in d.items()}
                        for s, d in results.items()},
        }, f, ensure_ascii=False, indent=2)
    print("[SAVE] extract_audit_report.json")


def show(a):
    """한 문서에서 손실 지점을 방식별로 나란히 본다."""
    ex = Extractor()
    stem = a.show
    pdf = os.path.join(a.docdir, f"{stem}.pdf")
    txt = os.path.join(a.docdir, f"{stem}.txt")
    if not os.path.exists(pdf):
        sys.exit(f"{pdf} 없음")

    cur = ""
    if os.path.exists(txt):
        with open(txt, encoding="utf-8", errors="replace") as f:
            cur = clean(f.read())

    outs = {"현재txt": cur}
    for m in METHODS:
        t = ex.run(pdf, m)
        if t:
            outs[m] = clean(t)

    # 현재 txt 에서 명사누락이 일어난 지점 주변을 뽑는다
    print("=" * 100)
    print(f"{stem} — 손실 지점 방식별 대조")
    print("=" * 100)
    spots = [m.start() for m in LOSS["명사누락"].finditer(cur)][:a.n]
    if not spots:
        print("  현재 txt 에 명사누락 없음")
    for i, pos in enumerate(spots, 1):
        frag = cur[max(0, pos - 25):pos + 30].replace("\n", " ")
        key = re.sub(r"\s+", "", cur[max(0, pos - 12):pos])[-6:]
        print(f"\n  [{i}] 현재txt: …{frag}…")
        if not key:
            continue
        for m in METHODS:
            if m not in outs:
                continue
            flat = re.sub(r"\s+", "", outs[m])
            j = flat.find(key)
            if j >= 0:
                print(f"      {m:<8}: …{flat[max(0, j - 10):j + 40]}…")
            else:
                print(f"      {m:<8}: (대응 지점 못 찾음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--show", default=None)
    ap.add_argument("--n", type=int, default=8, help="--show 시 볼 손실 지점 수")
    ap.add_argument("--save", action="store_true",
                    help="재추출 결과를 docs/_reextract/ 에 저장")
    a = ap.parse_args()
    if a.show:
        show(a)
    else:
        audit(a)


if __name__ == "__main__":
    main()
