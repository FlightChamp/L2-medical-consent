#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenizer_audit.py — SciBERT 한국어 토크나이저 진단
====================================================
질문:
    SciBERT tokenizer 가 한국어 의료 문서를 표현할 수 있는가?

측정:
    · 전체 token 수
    · [UNK] token 수와 비율
    · 한글 구간이 정상 tokenization 되는지
    · 대표 문장의 실제 token 결과

비교 대상:
    allenai/scibert_scivocab_uncased      영어 과학 문헌용
    jhgan/ko-sroberta-multitask           한국어 문장 임베딩용
    snuh/hari-q3-8b                       현재 생성 모델 (참고)

결론 표기 원칙:
    [UNK] 비율이 높게 나오더라도 "SciBERT 가 나쁜 모델"이라고 하지 않는다.
    정확한 결론은 "English scientific text 용 SciBERT tokenizer 는
    본 한국어 동의서 조건에서 직접 사용하기 어렵다" 이다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python experiment_scibert/tokenizer_audit.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
# 기존 스크립트는 상위 폴더(~/이윤우)에 있다. 같은 폴더에 두는 경우도 지원한다.
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections
except ImportError as e:
    sys.exit(f"[FATAL] prompt_ladder.py 가 상위 폴더에 있어야 합니다: {e}")

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
OUTDIR = os.path.join(HERE, "outputs")

MODELS = [
    ("allenai/scibert_scivocab_uncased", "SciBERT (영어 과학문헌)"),
    ("jhgan/ko-sroberta-multitask", "ko-sroberta (한국어 STS)"),
    ("snuh/hari-q3-8b", "HARI (현재 생성 모델)"),
]

SAMPLES = [
    "갑상선은 목 앞부분에 위치한 나비모양의 기관입니다.",
    "감염, 출혈, 혈전색전증, 신경손상 등의 가능성이 있습니다.",
    "수술 후 1~2주간 무거운 물건을 드는 것을 피하십시오.",
]


def load_docs(docdir: str) -> Dict[str, str]:
    out = {}
    for p in sorted(glob.glob(os.path.join(docdir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("syn"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            secs = extract_sections(f.read())
        if secs:
            out[stem] = " ".join(b for _, b in secs)
    return out


def audit(model_id: str, label: str, docs: Dict[str, str]) -> dict:
    from transformers import AutoTokenizer
    print(f"\n{'='*82}")
    print(f"{label}")
    print(f"  {model_id}")
    print("=" * 82)
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  로딩 실패: {type(e).__name__}: {str(e)[:120]}")
        return {"model": model_id, "label": label, "error": str(e)[:200]}

    unk = getattr(tok, "unk_token", None)
    vocab = getattr(tok, "vocab_size", None)
    print(f"  vocab_size={vocab}  unk_token={unk!r}")

    rows = []
    print(f"\n  {'문서':<8}{'글자':>7}{'토큰':>8}{'[UNK]':>8}{'UNK비율':>9}"
          f"{'글자/토큰':>10}")
    for doc, text in docs.items():
        ids = tok.encode(text, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        n_unk = sum(1 for t in toks if unk and t == unk)
        ratio = 100 * n_unk / max(len(toks), 1)
        rows.append({"doc": doc, "chars": len(text), "tokens": len(toks),
                     "unk": n_unk, "unk_ratio": round(ratio, 1)})
        print(f"  {doc:<8}{len(text):>7}{len(toks):>8}{n_unk:>8}"
              f"{ratio:>8.1f}%{len(text)/max(len(toks),1):>10.2f}")

    tot_tok = sum(r["tokens"] for r in rows)
    tot_unk = sum(r["unk"] for r in rows)
    tot_ratio = 100 * tot_unk / max(tot_tok, 1)
    print(f"  {'합계':<8}{sum(r['chars'] for r in rows):>7}{tot_tok:>8}"
          f"{tot_unk:>8}{tot_ratio:>8.1f}%")

    print(f"\n  대표 문장 토큰화")
    samples = []
    for s in SAMPLES:
        t = tok.tokenize(s)
        n_unk_s = sum(1 for x in t if unk and x == unk)
        samples.append({"text": s, "n_tokens": len(t), "n_unk": n_unk_s,
                        "tokens": t[:14]})
        print(f"    입력  : {s}")
        print(f"    토큰  : {len(t)}개, [UNK] {n_unk_s}개")
        print(f"    앞부분: {t[:14]}")
        print()

    return {"model": model_id, "label": label, "vocab_size": vocab,
            "unk_token": unk, "per_doc": rows,
            "total_tokens": tot_tok, "total_unk": tot_unk,
            "unk_ratio": round(tot_ratio, 2), "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "tokenizer_audit.json"))
    a = ap.parse_args()

    docs = load_docs(a.docdir)
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")
    print("=" * 82)
    print("SciBERT 한국어 토크나이저 진단")
    print("=" * 82)
    print(f"  문서 {len(docs)}건 (extract_sections 전처리 후)")

    results = [audit(m, l, docs) for m, l in MODELS]

    print("\n" + "=" * 82)
    print("요약")
    print("=" * 82)
    print(f"  {'모델':<36}{'vocab':>9}{'토큰':>9}{'[UNK]':>9}{'UNK비율':>10}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<36}{'로딩 실패':>37}")
            continue
        print(f"  {r['label']:<36}{r['vocab_size']:>9}{r['total_tokens']:>9}"
              f"{r['total_unk']:>9}{r['unk_ratio']:>9.1f}%")

    sci = next((r for r in results
                if r["model"].startswith("allenai") and "error" not in r), None)
    print("\n" + "=" * 82)
    print("판정")
    print("=" * 82)
    if sci is None:
        print("  SciBERT 로딩에 실패해 판정할 수 없습니다.")
    elif sci["unk_ratio"] >= 50:
        print(f"  SciBERT [UNK] 비율 {sci['unk_ratio']:.1f}%")
        print("  → English scientific text 용 SciBERT tokenizer 는")
        print("     본 한국어 동의서 조건에서 직접 사용하기 어렵습니다.")
        print("     (모델 자체의 품질 문제가 아니라 어휘 적용 범위의 문제입니다)")
    elif sci["unk_ratio"] >= 10:
        print(f"  SciBERT [UNK] 비율 {sci['unk_ratio']:.1f}% — 상당한 손실이 있습니다.")
    else:
        print(f"  SciBERT [UNK] 비율 {sci['unk_ratio']:.1f}% — 예상보다 낮습니다.")
        print("     한국어 처리 가능성을 추가로 확인할 필요가 있습니다.")
    print("=" * 82)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"n_docs": len(docs), "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}")


if __name__ == "__main__":
    main()
