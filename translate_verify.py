#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_verify.py — ⑥ 번역 검증 (양방향 교차 NLI)
====================================================
배경:
    CometKiwi 를 미채택한 뒤 ⑥ 번역 검증 자리가 비어 있었다.
    (미채택 근거: 갑상선→식도 오역에 -0.006, 장문에서 0.19 편차)

    오늘 AI Hub 병렬 말뭉치(사람 검수)로 교차 NLI 가 성립함을 확인했다.
      · 기본 판별력  MISMATCH AUROC 영 0.912 / 중 0.867 / 일 0.773
      · 누락 탐지는 방향을 뒤집어야 잡힌다
          원문→번역 0.488 / 0.516 / 0.550  (무작위)
          번역→원문 0.907 / 0.915 / 0.889  (세 언어 재현)
      · 용어 오역은 교차 조건에서도 못 잡는다 (AUROC 0.72~0.82, 탐지율 7~14%)

    검증된 이 방법을 우리 번역 결과에 적용한다.

구성:
    FWD  premise 한국어 → hypothesis 번역문   환각·추가 탐지
    REV  premise 번역문 → hypothesis 한국어   누락 탐지
    MIN  둘 중 낮은 값                        운영값
    용어 대조는 termdict_verified.json (영·중·일). 베트남어는 사전 없음.

베트남어에 대한 주의:
    한-베 병렬 말뭉치가 없어 용어 사전이 없다.
    환각·누락·수치는 교차 NLI 로 검증되지만 **오역은 검증되지 않는다.**
    갑상선을 Thực quản(식도)으로 잘못 번역해도 통과할 수 있다.
    따라서 베트남어의 높은 NLI 점수는 '환각과 누락이 적다'는 뜻이지
    '번역이 정확하다'는 뜻이 아니다. 보고 시 반드시 구분할 것.

경로 비교 (--mode both):
    A  원문 → 번역          (기존 translate_eval.json 과 같은 경로)
    B  원문 → 변환문 → 번역  (파이프라인 설계 순서)
    교수님 지적 "한국어에서 먼저 평이화한 뒤 번역하는 순서가 낫지 않냐"에
    데이터로 답하기 위한 비교다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python translate_verify.py --docs doc1              # 점검
    python translate_verify.py                          # 원문 경로만 (약 8분)
    python translate_verify.py --mode both              # 경로 비교 (약 25분)
    python translate_verify.py --eval-only              # 재생성 없이 검증만
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

GEN_MODEL = "snuh/hari-q3-8b"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HOME = os.path.expanduser("~/이윤우")
OUTDIR = os.path.join(HOME, "outputs_translate")

# 기존 translate_eval.py 와 동일한 정의에 vi 추가
LANGS = {
    "en": ("English", "영어"),
    "zh": ("Chinese", "중국어"),
    "ja": ("Japanese", "일본어"),
    "vi": ("Vietnamese", "베트남어"),
}
NOISE = {
    "en": {"cancer", "surgery", "disease", "treatment", "hospital", "doctor", "health"},
    "zh": {"手术", "患者", "治疗", "医院", "进行", "的患者", "在胆囊"},
    "ja": {"手術", "患者", "治療", "病院", "新型", "コロナウイルス", "血液"},
    "vi": set(),
}

TRANS_PROMPT = ("Translate the following Korean surgical consent form into {eng}. "
                "Translate every sentence faithfully. Do not add explanations.\n\n{text}")


# ===========================================================================
# 유틸
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
            if s and not cls.NUM_PREFIX.match(s) and len(s) >= 10:
                out.append(s)
        return out

    @classmethod
    def foreign(cls, t: str, lang: str) -> List[str]:
        """중·일은 구두점이 다르므로 분리 규칙을 달리한다."""
        if lang in ("zh", "ja"):
            parts = re.split(r"(?<=[。！？；])\s*|\n+", str(t))
            lo = 8
        else:
            parts = re.split(r"(?<=[.!?])\s+|\n+", str(t))
            lo = 15
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


NUM_UNIT = re.compile(r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|회|명|번|세)")
NUM_ANY = re.compile(r"(?<![\d.])(\d{1,4})(?![\d.])")
KOCH = re.compile(r"[가-힣]")


def numbers_of(t: str) -> Set[str]:
    return {m.group(1) for m in NUM_ANY.finditer(str(t))}


# ===========================================================================
# 용어 사전
# ===========================================================================

def load_terms() -> Dict[str, Dict[str, str]]:
    p = os.path.join(HOME, "termdict_verified.json")
    if not os.path.exists(p):
        print("  [경고] termdict_verified.json 없음 — 용어 대조 생략")
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
    print(f"  용어 사전 {len(out)}개 (영·중·일). 베트남어는 대응어 없음")
    return out


# ===========================================================================
# 모델
# ===========================================================================

class Generator:
    def __init__(self, gpu: int, cap: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.cap = cap
        print(f"[GEN] {GEN_MODEL} → cuda:{gpu}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                GEN_MODEL, dtype=torch.bfloat16, device_map={"": gpu})
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                GEN_MODEL, torch_dtype=torch.bfloat16, device_map={"": gpu})
        self.model.eval()

    def run(self, prompt: str) -> str:
        torch = self.torch
        msgs = [{"role": "user", "content": prompt}]
        try:
            t = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            t = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(t, return_tensors="pt").to(self.model.device)
        n = inp.input_ids.shape[1]
        with torch.no_grad():
            o = self.model.generate(**inp, max_new_tokens=self.cap,
                                    do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(o[0][n:], skip_special_tokens=True).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        if "</think>" in raw:
            raw = raw.split("</think>")[-1]
        return raw.strip()

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


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
        print(f"[NLI] entail index = {self.i_ent}")

    def rate(self, hyps: List[str], chunks: List[str], k: int,
             tau: float) -> float:
        torch = self.torch
        if not hyps or not chunks:
            return float("nan")
        prem, hy, owner = [], [], []
        for i, h in enumerate(hyps):
            for c in topk(h, chunks, k):
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
# 소스 로딩
# ===========================================================================

def load_sources(mode: str) -> Dict[str, Dict[str, str]]:
    """{doc: {'A': 원문, 'B': 변환문}}"""
    rep: Dict[str, dict] = {}
    for f in ("reportA.json", "reportB.json", "reportC.json"):
        p = os.path.join(HOME, f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                rep.update(json.load(fh))
    out: Dict[str, Dict[str, str]] = {}
    for doc, v in rep.items():
        if not isinstance(v, dict) or "원문" not in v:
            continue
        out[doc] = {"A": clean(v["원문"])}

    if mode == "both":
        # 오늘 재생성한 통짜 변환문을 우선 사용 (13문서 동일 조건)
        n_new = 0
        for doc in list(out):
            p = os.path.join(HOME, "outputs", f"{doc}__whole.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    r = json.load(fh)
                out[doc]["B"] = clean(" ".join(s["out"] for s in r["sections"]))
                n_new += 1
            elif "변환문" in rep[doc]:
                out[doc]["B"] = clean(rep[doc]["변환문"])
        print(f"  변환문: outputs/ 에서 {n_new}건 로드")
    return out


def path_of(doc: str, route: str, lang: str) -> str:
    return os.path.join(OUTDIR, f"{doc}__{route}__{lang}.json")


# ===========================================================================

def stage_generate(srcs: Dict[str, Dict[str, str]], routes: List[str],
                   langs: List[str], a):
    todo = [(d, r, l) for d in sorted(srcs) for r in routes for l in langs
            if r in srcs[d] and (a.force or not os.path.exists(path_of(d, r, l)))]
    if not todo:
        print("[GEN] 생성할 것이 없습니다 (--force 로 재생성)")
        return
    print(f"[GEN] 번역 {len(todo)}건")
    gen = Generator(a.gpu, a.max_new)
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    for n, (doc, route, lang) in enumerate(todo, 1):
        src = srcs[doc][route]
        eng = LANGS[lang][0]
        out = gen.run(TRANS_PROMPT.format(eng=eng, text=src))
        with open(path_of(doc, route, lang), "w", encoding="utf-8") as f:
            json.dump({"doc": doc, "route": route, "lang": lang,
                       "model": GEN_MODEL, "src": src, "out": out},
                      f, ensure_ascii=False, indent=2)
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {doc}/{route}/{lang}: "
              f"{len(src)}자 → {len(out)}자  "
              f"[{el:.0f}s, 남은 예상 {el/n*(len(todo)-n):.0f}s]", flush=True)
    gen.free()


def stage_eval(srcs, routes, langs, a) -> List[dict]:
    terms = load_terms()
    nli = NLI(a.gpu, a.batch)
    rows = []
    print(f"\n  {'문서':<7}{'경로':<4}{'언어':<4}{'FWD':>7}{'REV':>7}{'MIN':>7}"
          f"{'수치':>7}{'용어':>7}{'한글잔존':>9}")
    for doc in sorted(srcs):
        for route in routes:
            for lang in langs:
                p = path_of(doc, route, lang)
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    r = json.load(f)
                src, out = r["src"], r["out"]
                ko_s = Splitter.ko(src)
                fo_s = Splitter.foreign(out, lang)
                if not ko_s or not fo_s:
                    continue
                fwd = nli.rate(fo_s, build_chunks(ko_s), a.k, a.tau)
                rev = nli.rate(ko_s, build_chunks(fo_s), a.k, a.tau)

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

                row = {
                    "doc": doc, "route": route, "lang": lang,
                    "src_chars": len(src), "out_chars": len(out),
                    "fwd": round(fwd, 1), "rev": round(rev, 1),
                    "min": round(min(fwd, rev), 1),
                    "num_keep": round(num_keep, 1) if num_keep is not None else None,
                    "n_nums": len(ns),
                    "term_acc": round(term_acc, 1) if term_acc is not None else None,
                    "n_terms": hit + miss,
                    "ko_left": len(KOCH.findall(out)),
                }
                rows.append(row)
                ta = f"{term_acc:.1f}" if term_acc is not None else "—"
                nk = f"{num_keep:.1f}" if num_keep is not None else "—"
                print(f"  {doc:<7}{route:<4}{lang:<4}{fwd:>7.1f}{rev:>7.1f}"
                      f"{row['min']:>7.1f}{nk:>7}{ta:>7}{row['ko_left']:>9}",
                      flush=True)
    return rows


def report(rows: List[dict], routes: List[str], langs: List[str]):
    if not rows:
        print("결과 없음")
        return
    import statistics as st

    def avg(key, **filt):
        v = [r[key] for r in rows
             if all(r[k] == x for k, x in filt.items()) and r.get(key) is not None]
        return st.fmean(v) if v else float("nan")

    print("\n" + "=" * 92)
    print("언어별 요약")
    print("=" * 92)
    for route in routes:
        if not any(r["route"] == route for r in rows):
            continue
        label = "A 원문→번역" if route == "A" else "B 변환문→번역"
        print(f"\n  [{label}]")
        print(f"    {'언어':<8}{'n':>4}{'FWD':>9}{'REV':>9}{'MIN':>9}"
              f"{'수치보존':>10}{'용어정확':>10}{'한글잔존':>10}")
        for lang in langs:
            sel = [r for r in rows if r["route"] == route and r["lang"] == lang]
            if not sel:
                continue
            ta = avg("term_acc", route=route, lang=lang)
            print(f"    {LANGS[lang][1]:<8}{len(sel):>4}"
                  f"{avg('fwd', route=route, lang=lang):>9.1f}"
                  f"{avg('rev', route=route, lang=lang):>9.1f}"
                  f"{avg('min', route=route, lang=lang):>9.1f}"
                  f"{avg('num_keep', route=route, lang=lang):>10.1f}"
                  + (f"{ta:>10.1f}" if ta == ta else f"{'사전없음':>10}")
                  + f"{avg('ko_left', route=route, lang=lang):>10.1f}")

    if len(routes) > 1 and all(any(r["route"] == x for r in rows) for x in routes):
        print("\n" + "=" * 92)
        print("경로 비교 — 평이화 후 번역이 나은가")
        print("=" * 92)
        print(f"  {'언어':<8}" + "".join(f"{k:>12}" for k in
                                        ("A MIN", "B MIN", "차이", "A 용어", "B 용어"))) 
        for lang in langs:
            a_m, b_m = avg("min", route="A", lang=lang), avg("min", route="B", lang=lang)
            a_t, b_t = avg("term_acc", route="A", lang=lang), avg("term_acc", route="B", lang=lang)
            print(f"  {LANGS[lang][1]:<8}{a_m:>12.1f}{b_m:>12.1f}{b_m-a_m:>+12.1f}"
                  + (f"{a_t:>12.1f}{b_t:>12.1f}" if a_t == a_t else f"{'—':>12}{'—':>12}"))
        print("\n  B가 높으면 '평이화 후 번역'이 유리하다는 근거가 됩니다.")

    print("\n" + "=" * 92)
    print("해석 시 주의")
    print("=" * 92)
    print("  · FWD(원문→번역)는 환각·추가를, REV(번역→원문)는 누락을 잡습니다.")
    print("    AI Hub 검증에서 누락은 FWD 로 탐지 불가(AUROC 0.49~0.55)였습니다.")
    print("  · 교차 NLI 는 용어 오역을 못 잡습니다 (AUROC 0.72~0.82, 탐지율 7~14%).")
    print("    따라서 MIN 이 높다 = 환각·누락이 적다 이지, 번역이 정확하다는 뜻이 아닙니다.")
    print("  · 베트남어는 한-베 병렬 말뭉치가 없어 용어 대조가 불가능합니다.")
    print("    환각·누락·수치만 검증된 상태이며, 오역 여부는 확인되지 않았습니다.")
    print("  · 한글잔존이 크면 번역이 중간에 멈췄거나 실패한 것입니다.")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--mode", choices=["A", "both"], default="A")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="translate_verify")
    a = ap.parse_args()

    routes = ["A"] if a.mode == "A" else ["A", "B"]
    srcs = load_sources(a.mode)
    if a.docs:
        srcs = {d: v for d, v in srcs.items() if d in a.docs}
    if not srcs:
        sys.exit("[FATAL] report*.json 에서 원문을 찾지 못했습니다")
    print(f"[PLAN] 문서 {len(srcs)}건 × 경로 {len(routes)} × 언어 {len(a.langs)} = "
          f"{len(srcs)*len(routes)*len(a.langs)} 번역")

    os.makedirs(OUTDIR, exist_ok=True)
    if not a.eval_only:
        stage_generate(srcs, routes, a.langs, a)
    rows = stage_eval(srcs, routes, a.langs, a)
    report(rows, routes, a.langs)

    with open(f"{a.out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_metrics.json / 번역문은 {OUTDIR}/")


if __name__ == "__main__":
    main()
