#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_verify2.py — ⑥ 번역 검증 v2 (서술문/서식항목 분리)
=============================================================
v1 에서 드러난 문제:
    MIN 점수가 전 언어 24~37 로 낮게 나왔다.
    그런데 번역문 자체는 정상이었다.
      중국어: 甲状腺是位于颈部前方的蝴蝶状器官  (원문 정확히 반영)
      용어정확도 86~93%, 한글잔존 0

    원인은 문서 유형이었다. 동의서 내용의 **41%만 서술문이고 59%는 서식 항목**이다.
      doc1 36% / doc4 36% / doc10 47% / doc12 45%  (합계 119/287)

    NLI 는 명제 간 함의를 판정한다.
    `신장질환 (부종 등)` 같은 명사구에는 참·거짓을 물을 수 없으므로
    함의 판정이 성립하지 않고 모델은 neutral 로 답한다.
    즉 v1 의 24~37 은 번역 품질이 아니라 지표를 잘못 적용한 결과다.

v2 의 처리:
    서술문(41%)  교차 NLI 양방향 — 환각(FWD)·누락(REV)
    서식항목(59%) 항목 대응 검사 — 용어·수치가 번역문에 있는가

    두 유형은 검증 방법이 다르므로 점수도 따로 보고한다.
    하나의 종합 점수로 합치면 다시 같은 오류를 반복하게 된다.

주의 (v1 과 동일):
    · 교차 NLI 는 용어 오역을 못 잡는다 (AI Hub 검증 AUROC 0.72~0.82, 탐지율 7~14%).
      NLI 점수가 높다 = 환각·누락이 적다 이지, 번역이 정확하다는 뜻이 아니다.
    · 베트남어는 한-베 병렬 말뭉치가 없어 용어 사전이 없다.
      환각·누락·수치만 검증되며 오역 여부는 확인되지 않는다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python translate_verify2.py                  # outputs_translate/ 의 번역문 재채점
    python translate_verify2.py --docs doc1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HOME = os.path.expanduser("~/이윤우")
TRANSDIR = os.path.join(HOME, "outputs_translate")

LANGS = {"en": "영어", "zh": "중국어", "ja": "일본어", "vi": "베트남어"}
NOISE = {
    "en": {"cancer", "surgery", "disease", "treatment", "hospital", "doctor", "health"},
    "zh": {"手术", "患者", "治疗", "医院", "进行", "的患者", "在胆囊"},
    "ja": {"手術", "患者", "治療", "病院", "新型", "コロナウイルス", "血液"},
    "vi": set(),
}


# ===========================================================================
# 서술문 / 서식항목 판별 (test_predicate.py 로 23개 사례 검증 완료)
# ===========================================================================

KO_PRED = re.compile(r"(다|요|음|임|함|까|죠)[.\s]*$")
KO_FORM = re.compile(r"(무\s*유|유\s*무|미상|해당\s*없음|[:：]\s*$|^\s*[가-힣]{1,6}\s*$)")

FO_END = {
    "en": re.compile(r"[.!?]\s*$"),
    "vi": re.compile(r"[.!?]\s*$"),
    "zh": re.compile(r"[。！？；]\s*$"),
    "ja": re.compile(r"[。！？]\s*$"),
}
FO_MINLEN = {"en": 30, "vi": 30, "zh": 12, "ja": 14}


def ko_is_pred(s: str) -> bool:
    s = s.strip()
    if len(s) < 12:
        return False
    if KO_FORM.search(s) and not KO_PRED.search(s):
        return False
    return bool(KO_PRED.search(s))


def fo_is_pred(s: str, lang: str) -> bool:
    s = s.strip()
    if len(s) < FO_MINLEN.get(lang, 25):
        return False
    return bool(FO_END[lang].search(s))


# ===========================================================================
# 전처리
# ===========================================================================

def clean(t) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    KO = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def ko(cls, t: str) -> List[str]:
        out = []
        for p in cls.KO.split(str(t)):
            s = clean(p or "")
            if s and not cls.NUM_PREFIX.match(s) and len(s) >= 6:
                out.append(s)
        return out

    @classmethod
    def foreign(cls, t: str, lang: str) -> List[str]:
        if lang in ("zh", "ja"):
            parts = re.split(r"(?<=[。！？；])\s*|\n+", str(t))
            lo = 5
        else:
            parts = re.split(r"(?<=[.!?])\s+|\n+", str(t))
            lo = 10
        return [s for s in (clean(p) for p in parts if p) if len(s) >= lo]


def build_chunks(sents: Sequence[str], span: int = 3) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for k in range(1, span + 1):
        for i in range(0, len(sents) - k + 1):
            c = " ".join(sents[i:i + k])
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out or ["(빈 문서)"]


def ngrams(t: str, n: int) -> Set[str]:
    s = re.sub(r"\s+", "", t)
    return {s[i:i + n] for i in range(len(s) - n + 1)} or {s}


def topk(hyp: str, chunks: Sequence[str], k: int) -> List[str]:
    hg = ngrams(hyp, 2)
    sc = []
    for i, c in enumerate(chunks):
        g = ngrams(c, 2)
        sc.append((len(hg & g) / max(len(hg | g), 1), i))
    sc.sort(key=lambda x: -x[0])
    return [chunks[i] for _, i in sc[:k]]


NUM_ANY = re.compile(r"(?<![\d.])(\d{1,4})(?![\d.])")
KOCH = re.compile(r"[가-힣]")


def numbers_of(t: str) -> Set[str]:
    return {m.group(1) for m in NUM_ANY.finditer(str(t))}


def load_terms() -> Dict[str, Dict[str, str]]:
    p = os.path.join(HOME, "termdict_verified.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, Dict[str, str]] = {}
    for t, per in d.items():
        e = {}
        for code in ("en", "ja", "zh"):
            for item in per.get(code, []):
                c = item[0] if isinstance(item, (list, tuple)) else item
                if c in NOISE.get(code, set()):
                    continue
                e[code] = c
                break
        if e:
            out[t] = e
    return out


# ===========================================================================

class NLI:
    def __init__(self, gpu: int, batch: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.bs = batch
        self.dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[NLI] {NLI_MODEL} → {self.dev}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL).to(self.dev).eval()
        id2 = {int(k): str(v).lower()
               for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))

    def rate(self, hyps: List[str], chunks: List[str], k: int,
             tau: float) -> float:
        torch = self.torch
        if not hyps or not chunks:
            return float("nan")
        prem, hy, owner = [], [], []
        for i, h in enumerate(hyps):
            # 교차 언어에서는 어휘 겹침 검색이 작동하지 않는다.
            # 한국어와 영어는 공유 문자가 0이라 Jaccard 가 전부 0.000 이 되고
            # topk 가 사실상 앞 k개를 집는다. 따라서 전체 청크와 비교한다.
            cand = chunks if k <= 0 else topk(h, chunks, k)
            for c in cand:
                prem.append(c)
                hy.append(h)
                owner.append(i)
        sc: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hy[i:i + self.bs],
                           truncation=True, padding=True, max_length=256,
                           return_tensors="pt")
            enc = {kk: v.to(self.dev) for kk, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            sc.extend(p[:, self.i_ent].tolist())
        best: Dict[int, float] = {}
        for o, s in zip(owner, sc):
            best[o] = max(best.get(o, -1.0), s)
        return 100 * sum(1 for i in range(len(hyps))
                         if best.get(i, 0.0) >= tau) / len(hyps)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transdir", default=TRANSDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="translate_verify2")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.transdir, "*__*__*.json")))
    if not files:
        sys.exit(f"[FATAL] {a.transdir}/*.json 없음 — translate_verify.py 를 먼저 실행하세요")
    terms = load_terms()
    print(f"[LOAD] 번역문 {len(files)}건 / 용어사전 {len(terms)}개 (영·중·일)")
    nli = NLI(a.gpu, a.batch)

    rows = []
    print(f"\n  {'문서':<7}{'경로':<4}{'언어':<4}"
          f"{'서술비':>7}{'FWD':>7}{'REV':>7}{'MIN':>7}"
          f"{'항목대응':>9}{'수치':>7}{'용어':>7}")
    for p in files:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        doc, route, lang = r["doc"], r["route"], r["lang"]
        if a.docs and doc not in a.docs:
            continue
        if lang not in a.langs:
            continue
        src, out = r["src"], r["out"]

        ko_all = Splitter.ko(src)
        fo_all = Splitter.foreign(out, lang)
        ko_pred = [s for s in ko_all if ko_is_pred(s)]
        ko_form = [s for s in ko_all if not ko_is_pred(s)]
        fo_pred = [s for s in fo_all if fo_is_pred(s, lang)]
        pred_ratio = 100 * len(ko_pred) / max(len(ko_all), 1)

        # 1) 서술문 — 교차 NLI 양방향
        if ko_pred and fo_pred:
            fwd = nli.rate(fo_pred, build_chunks(ko_pred), a.k, a.tau)
            rev = nli.rate(ko_pred, build_chunks(fo_pred), a.k, a.tau)
            mn = min(fwd, rev)
        else:
            fwd = rev = mn = float("nan")

        # 2) 서식 항목 — 용어·수치가 번역문에 대응되는가
        #    항목 안의 사전 용어가 번역문에 있으면 대응된 것으로 본다.
        matched = checked = 0
        for item in ko_form:
            present = [t for t in terms if t in item]
            if not present:
                continue
            checked += 1
            if any((terms[t].get(lang) or "").lower() in out.lower()
                   for t in present if terms[t].get(lang)):
                matched += 1
        item_match = 100 * matched / checked if checked else None

        # 3) 수치·용어 (문서 전체 기준)
        ns, no = numbers_of(src), numbers_of(out)
        num_keep = 100 * len(ns & no) / len(ns) if ns else None
        present = [t for t in terms if t in src]
        hit = miss = 0
        for t in present:
            c = terms[t].get(lang)
            if not c:
                continue
            if c.lower() in out.lower():
                hit += 1
            else:
                miss += 1
        term_acc = 100 * hit / (hit + miss) if hit + miss else None

        row = {"doc": doc, "route": route, "lang": lang,
               "n_all": len(ko_all), "n_pred": len(ko_pred),
               "pred_ratio": round(pred_ratio, 1),
               "fwd": round(fwd, 1) if fwd == fwd else None,
               "rev": round(rev, 1) if rev == rev else None,
               "min": round(mn, 1) if mn == mn else None,
               "item_match": round(item_match, 1) if item_match is not None else None,
               "n_items": checked,
               "num_keep": round(num_keep, 1) if num_keep is not None else None,
               "term_acc": round(term_acc, 1) if term_acc is not None else None,
               "ko_left": len(KOCH.findall(out))}
        rows.append(row)

        def f(x):
            return f"{x:.1f}" if x is not None else "—"
        print(f"  {doc:<7}{route:<4}{lang:<4}{pred_ratio:>6.0f}%"
              f"{f(row['fwd']):>7}{f(row['rev']):>7}{f(row['min']):>7}"
              f"{f(row['item_match']):>9}{f(row['num_keep']):>7}"
              f"{f(row['term_acc']):>7}", flush=True)

    # -- 요약 ---------------------------------------------------------------
    def avg(key, **filt):
        v = [r[key] for r in rows
             if all(r[k] == x for k, x in filt.items()) and r.get(key) is not None]
        return st.fmean(v) if v else float("nan")

    routes = sorted({r["route"] for r in rows})
    print("\n" + "=" * 96)
    print("언어별 요약 — 서술문과 서식항목을 분리해 본다")
    print("=" * 96)
    for route in routes:
        label = "A 원문→번역" if route == "A" else "B 변환문→번역"
        print(f"\n  [{label}]")
        print(f"    {'언어':<8}{'n':>4}{'서술비':>8}"
              f"{'FWD':>8}{'REV':>8}{'MIN':>8}{'항목대응':>10}"
              f"{'수치보존':>10}{'용어정확':>10}")
        for lang in a.langs:
            sel = [r for r in rows if r["route"] == route and r["lang"] == lang]
            if not sel:
                continue
            ta = avg("term_acc", route=route, lang=lang)
            print(f"    {LANGS[lang]:<8}{len(sel):>4}"
                  f"{avg('pred_ratio', route=route, lang=lang):>7.0f}%"
                  f"{avg('fwd', route=route, lang=lang):>8.1f}"
                  f"{avg('rev', route=route, lang=lang):>8.1f}"
                  f"{avg('min', route=route, lang=lang):>8.1f}"
                  f"{avg('item_match', route=route, lang=lang):>10.1f}"
                  f"{avg('num_keep', route=route, lang=lang):>10.1f}"
                  + (f"{ta:>10.1f}" if ta == ta else f"{'사전없음':>10}"))

    if len(routes) > 1:
        print("\n" + "=" * 96)
        print("경로 비교 — 평이화 후 번역이 나은가")
        print("=" * 96)
        print(f"  {'언어':<8}{'A MIN':>10}{'B MIN':>10}{'차이':>9}"
              f"{'A 용어':>10}{'B 용어':>10}{'차이':>9}")
        for lang in a.langs:
            am, bm = avg("min", route="A", lang=lang), avg("min", route="B", lang=lang)
            at, bt = avg("term_acc", route="A", lang=lang), avg("term_acc", route="B", lang=lang)
            line = f"  {LANGS[lang]:<8}{am:>10.1f}{bm:>10.1f}{bm-am:>+9.1f}"
            line += (f"{at:>10.1f}{bt:>10.1f}{bt-at:>+9.1f}"
                     if at == at else f"{'—':>10}{'—':>10}{'—':>9}")
            print(line)

    print("\n" + "=" * 96)
    print("해석")
    print("=" * 96)
    pr = avg("pred_ratio")
    print(f"  · 동의서 내용의 {pr:.0f}%만 서술문입니다. 나머지는 서식 항목입니다.")
    print("    NLI 는 서술문에만 적용되며, 서식 항목은 용어 대응으로 검증합니다.")
    print("  · v1 은 두 유형을 섞어 채점해 MIN 24~37 이 나왔습니다. 지표 오적용이었습니다.")
    print("  · 교차 NLI 는 용어 오역을 못 잡습니다 (AUROC 0.72~0.82, 탐지율 7~14%).")
    print("    MIN 이 높다 = 환각·누락이 적다 이지, 번역이 정확하다는 뜻이 아닙니다.")
    print("  · 베트남어는 용어 사전이 없어 오역 검증이 불가합니다.")
    print("=" * 96)

    with open(f"{a.out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_metrics.json")


if __name__ == "__main__":
    main()
