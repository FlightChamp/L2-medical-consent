#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inventory.py — 보유 자료 전수 조사 (GPU 불필요, 수 초)"""
import os, re, json, glob, subprocess

HOME   = os.path.expanduser("~/이윤우")
DOCS   = os.path.join(HOME, "docs")
SHARED = os.path.expanduser("~/shared")

def kb(p):
    try: return os.path.getsize(p)/1024
    except: return 0

print("=" * 78)
print("[1] docs/ 원본 자료")
print("=" * 78)
if os.path.isdir(DOCS):
    ext = {}
    for f in sorted(os.listdir(DOCS)):
        p = os.path.join(DOCS, f)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            print(f"  📁 {f}/  ({n}개 파일)")
            continue
        e = os.path.splitext(f)[1].lower() or "(없음)"
        ext.setdefault(e, []).append(f)
    for e, fs in sorted(ext.items()):
        print(f"  {e:<8} {len(fs):>3}개  →  {', '.join(sorted(fs)[:16])}"
              + (" …" if len(fs) > 16 else ""))
else:
    print("  docs/ 없음")

print("\n" + "=" * 78)
print("[2] 결과 파일(JSON) — 재계산 가능 여부가 핵심")
print("=" * 78)
for p in sorted(glob.glob(os.path.join(HOME, "*.json"))):
    f = os.path.basename(p)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  ❌ {f:<28} 파싱 실패: {e}"); continue
    if not isinstance(d, dict):
        print(f"  •  {f:<28} (리스트/기타, {len(d)}항목, {kb(p):.0f}KB)"); continue
    keys = list(d.keys())
    print(f"  •  {f:<28} {len(keys)}개 항목 ({kb(p):.0f}KB)")
    print(f"     항목: {', '.join(keys[:14])}{' …' if len(keys)>14 else ''}")
    first = d[keys[0]] if keys else None
    if isinstance(first, dict):
        has_src = "원문" in first
        has_out = "변환문" in first
        mark = "✅ 재계산 가능" if (has_src and has_out) else "⚠ 원문/변환문 없음 → 재계산 불가"
        print(f"     필드: {', '.join(list(first.keys())[:12])}")
        print(f"     {mark}")

print("\n" + "=" * 78)
print("[3] 스크립트 — 어떤 모델·데이터를 쓰는지")
print("=" * 78)
MODELPAT = re.compile(r'["\']([\w\-.]+/[\w\-.]+)["\']')
for p in sorted(glob.glob(os.path.join(HOME, "*.py"))):
    f = os.path.basename(p)
    src = open(p, encoding="utf-8", errors="replace").read()
    n = src.count("\n") + 1
    models = sorted({m for m in MODELPAT.findall(src)
                     if "/" in m and not m.startswith(("~", ".", "/"))
                     and not m.endswith((".py", ".json", ".txt", ".md"))})
    inline = "있음" if re.search(r'"""[^"]{300,}', src) else "-"
    print(f"  {f:<20} {n:>4}줄  모델:{', '.join(models) if models else '-':<40} 인라인장문:{inline}")

print("\n" + "=" * 78)
print("[4] 로그 파일")
print("=" * 78)
for p in sorted(glob.glob(os.path.join(HOME, "*log*.txt")) +
                glob.glob(os.path.join(HOME, "*.log"))):
    f = os.path.basename(p)
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
        docs = re.findall(r"^### (\w+)$", txt, re.M)
        summ = len(re.findall(r">> 사실", txt))
    except Exception:
        docs, summ = [], 0
    print(f"  {f:<28} {kb(p):>7.0f}KB  문서 {len(docs)}개 {docs[:8]}  요약줄 {summ}")

print("\n" + "=" * 78)
print("[5] 공용/기타 폴더")
print("=" * 78)
for d in [SHARED, os.path.expanduser("~")]:
    if os.path.isdir(d):
        try:
            items = sorted(os.listdir(d))[:20]
            print(f"  {d}: {', '.join(items)}")
        except Exception as e:
            print(f"  {d}: 접근 불가 ({e})")

print("\n" + "=" * 78)
print("[6] 용량")
print("=" * 78)
for cmd in [f"du -sh {HOME} 2>/dev/null",
            f"du -sh {DOCS} 2>/dev/null",
            "df -h ~ | tail -1"]:
    try:
        print("  " + subprocess.check_output(cmd, shell=True, text=True).strip())
    except Exception:
        pass
