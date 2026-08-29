#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosslingual_nli2.py — 언어 교차 NLI v2
=======================================
v1 에서 드러난 세 가지를 고친다.

  (1) 용어사전 파싱 실패
      termdict_verified.json 의 값은 [용어, 점수] 쌍의 리스트다.
      v1 파서는 문자열만 처리해 30개 사전이 통째로 0개가 됐다.
      점수 하한도 건다 — 갑상선의 en 후보에 'cancer'(0.474)가 섞여 있는데
      갑상선암 기사가 많아 공기 빈도가 높았을 뿐 대역어가 아니다.

  (2) 말뭉치가 첫 파일만 로드됨
      파일당 4,000행이라 첫 파일에서 표본이 다 찼다.
      실제로 쓴 것은 학교 가정통신문·자동차 뉴스였다.
      의료 파일을 우선하고 여러 파일에서 고르게 뽑는다.

  (3) 누락(TRUNCATE) 탐지 실패 — 설계 결함
      번역을 잘라내도 남은 조각은 원문에서 여전히 참이다.
      NLI 는 hypothesis 가 참인지를 묻지 완전한지를 묻지 않는다.
      방향을 뒤집으면 잡힌다.

양방향 채점:
    FWD  premise 한국어 원문 → hypothesis 번역문   … 환각·수치 변조
    REV  premise 번역문      → hypothesis 한국어 원문 … 누락
    MIN  둘 중 낮은 값                              … 실제 운영값

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python crosslingual_nli2.py
    python crosslingual_nli2.py --n 400 --min-term-score 0.6
    python crosslingual_nli2.py --langs zh --no-prefer-medical
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DATA_ROOT = "/home/hufs/shared/data"
CORPUS_DIRS = {"en": "corpus_enko", "zh": "corpus_zhko", "ja": "corpus_jako"}
LANG_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "vi": "베트남어"}

KINDS = ("MISMATCH", "NUM_CHANGE", "TRUNCATE", "TERM_SWAP")
DIRS = ("FWD", "REV", "MIN")


# ===========================================================================
# 유틸
# ===========================================================================

def clean(t) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(t))
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", t).strip()


def hangul_ratio(t: str) -> float:
    if not t:
        return 0.0
    ko = sum(1 for c in t if "가" <= c <= "힣")
    letters = sum(1 for c in t if c.isalpha() or "\u4e00" <= c <= "\u9fff")
    return ko / max(letters, 1)


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = [-x for x in pos], [-x for x in neg]
    allv = sorted(p + n)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = sum(ranks[x] for x in p)
    return (r - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def tau_for_spec(scores: Sequence[float], target: float) -> float:
    xs = sorted(scores)
    k = max(0, min(int(round((1.0 - target) * len(xs))), len(xs) - 1))
    return xs[k]


# ===========================================================================
# 용어사전 (구조 수정)
# ===========================================================================

def load_termdict(path: str, lang: str, min_score: float) -> Dict[str, str]:
    """termdict_verified.json → {한국어: 대상언어 최상위 용어}

    값 형태: {"갑상선": {"en": [["thyroid", 0.763], ["cancer", 0.474]], ...}}
    점수 하한 미만은 버린다. 공기 빈도만 높은 잡음을 걸러내기 위함이다.
    """
    if not os.path.exists(path):
        print(f"  [경고] {path} 없음 — TERM_SWAP 건너뜀")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [경고] 사전 읽기 실패 {e}")
        return {}
    out: Dict[str, str] = {}
    dropped = 0
    for k, v in (d.items() if isinstance(d, dict) else []):
        if not isinstance(v, dict):
            continue
        lst = v.get(lang)
        if not isinstance(lst, list) or not lst:
            continue
        best = None
        for item in lst:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    t, s = str(item[0]), float(item[1])
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, str):
                t, s = item, 1.0
            else:
                continue
            if s < min_score:
                dropped += 1
                continue
            if best is None or s > best[1]:
                best = (t, s)
        if best:
            out[k] = best[0]
    print(f"  용어사전 {len(out)}개 확보 (점수 {min_score} 미만 후보 {dropped}개 제외)")
    return out


# ===========================================================================
# 말뭉치 로딩
# ===========================================================================

class CorpusLoader:
    def __init__(self, root: str, prefer_medical: bool = True):
        self.root = root
        self.prefer_medical = prefer_medical

    def files(self, subdir: str) -> List[str]:
        base = os.path.join(self.root, subdir)
        if not os.path.isdir(base):
            return []
        out = []
        for ext in ("json", "jsonl", "csv", "tsv"):
            out.extend(glob.glob(os.path.join(base, "**", f"*.{ext}"),
                                 recursive=True))
        # 의료 파일을 앞으로
        def key(p):
            n = os.path.basename(p).lower()
            return (0 if (self.prefer_medical and "medical" in n) else 1, n)
        return sorted(out, key=key)

    def _records(self, path: str, limit: int) -> List[dict]:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".jsonl":
                out = []
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if len(out) >= limit:
                            break
                        if line.strip():
                            out.append(json.loads(line))
                return out
            if ext == ".json":
                with open(path, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [d for d in data[:limit] if isinstance(d, dict)]
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            return v[:limit]
                return []
            delim = "\t" if ext == ".tsv" else ","
            with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
                return [r for _, r in zip(range(limit),
                                          csv.DictReader(f, delimiter=delim))]
        except Exception as e:
            print(f"    (읽기 실패 {os.path.basename(path)}: {type(e).__name__})")
        return []

    @staticmethod
    def _pick_columns(recs: List[dict]) -> Optional[Tuple[str, str]]:
        if not recs:
            return None
        stats = {}
        for k in recs[0]:
            vals = [str(r.get(k, "")) for r in recs[:200] if r.get(k)]
            vals = [v for v in vals if len(v) >= 8]
            if len(vals) < 5:
                continue
            stats[k] = (sum(hangul_ratio(v) for v in vals) / len(vals),
                        sum(len(v) for v in vals) / len(vals))
        ko = [k for k, (h, _) in stats.items() if h >= 0.4]
        fo = [k for k, (h, _) in stats.items() if h < 0.1]
        if not ko or not fo:
            return None
        ko.sort(key=lambda k: -stats[k][1])
        fo.sort(key=lambda k: -stats[k][1])
        return (ko[0], fo[0])

    def load(self, lang: str, n: int, terms: Sequence[str],
             rng: random.Random) -> List[Tuple[str, str, str]]:
        """(한국어, 외국어, 출처파일). 사전 용어가 든 문장을 우선한다."""
        sub = CORPUS_DIRS.get(lang)
        if not sub:
            return []
        with_term: List[Tuple[str, str, str]] = []
        without: List[Tuple[str, str, str]] = []
        files = self.files(sub)
        print(f"  파일 {len(files)}개 탐색")
        for path in files:
            name = os.path.basename(path)
            recs = self._records(path, 4000)
            cols = self._pick_columns(recs)
            if not cols:
                continue
            kc, fc = cols
            got_w = got_n = 0
            for r in recs:
                ko, fo = clean(r.get(kc, "")), clean(r.get(fc, ""))
                if not (15 <= len(ko) <= 250 and 10 <= len(fo) <= 400):
                    continue
                if hangul_ratio(ko) < 0.4 or hangul_ratio(fo) >= 0.1:
                    continue
                if fo.startswith("http"):
                    continue
                if any(t in ko for t in terms):
                    with_term.append((ko, fo, name))
                    got_w += 1
                else:
                    without.append((ko, fo, name))
                    got_n += 1
            if got_w or got_n:
                print(f"    {name[:46]:<48} 용어포함 {got_w:>4} / 기타 {got_n:>5}")
            if len(with_term) >= n and len(without) >= n:
                break
        rng.shuffle(with_term)
        rng.shuffle(without)
        # 용어 포함 문장을 최대 절반까지 우선 배치
        half = min(len(with_term), max(n // 2, n - len(without)))
        picked = with_term[:half] + without[: n - half]
        rng.shuffle(picked)
        srcs = Counter(s for _, _, s in picked)
        print(f"  → 표본 {len(picked)}건 (용어포함 {half}건) / 출처 {dict(srcs)}")
        return picked


# ===========================================================================
# 훼손
# ===========================================================================

class Corrupter:
    NUM = re.compile(r"(?<![\d.])(\d{1,4})")

    def __init__(self, lang: str, termdict: Dict[str, str], rng: random.Random):
        self.lang = lang
        self.rng = rng
        self.pairs = termdict
        self.values = sorted({v for v in termdict.values() if len(v) >= 2})

    def mismatch(self, fo: str, others: Sequence[str]) -> Optional[str]:
        for _ in range(10):
            c = self.rng.choice(others)
            if c != fo:
                return c
        return None

    def num_change(self, fo: str) -> Optional[str]:
        m = self.NUM.search(fo)
        if not m:
            return None
        return fo[:m.start()] + str(int(m.group(1)) * 3 + 1) + fo[m.end():]

    def truncate(self, fo: str) -> Optional[str]:
        if len(fo) < 40:
            return None
        cut = fo[: int(len(fo) * 0.55)].rstrip()
        return cut if len(cut) >= 15 else None

    def term_swap(self, ko: str, fo: str) -> Optional[Tuple[str, str]]:
        """한국어에 있는 용어의 대역어를, 원문에 없는 다른 용어로 바꾼다."""
        cands = [(k, v) for k, v in self.pairs.items()
                 if k in ko and v.lower() in fo.lower()]
        if not cands:
            return None
        src_ko, src = max(cands, key=lambda x: len(x[1]))
        alts = [v for v in self.values
                if v.lower() != src.lower() and v.lower() not in fo.lower()]
        if not alts:
            return None
        dst = self.rng.choice(alts)
        i = fo.lower().find(src.lower())
        return (fo[:i] + dst + fo[i + len(src):], f"{src_ko}: {src}→{dst}")


# ===========================================================================
# 채점
# ===========================================================================

class Scorer:
    def __init__(self, batch_size: int = 64, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.bs, self.ml = batch_size, max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] device={self.device} model={MODEL_ID}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()
        id2 = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, v in id2.items() if v.startswith("entail"))
        print(f"[INIT] id2label={id2} -> entail={self.i_ent}")

    def entail(self, prem: List[str], hyp: List[str]) -> List[float]:
        torch = self.torch
        out: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hyp[i:i + self.bs],
                           truncation=True, padding=True, max_length=self.ml,
                           return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(p[:, self.i_ent].tolist())
        return out


@dataclass
class Row:
    lang: str
    kind: str
    detail: str
    ko: str
    fo: str
    fwd: float = 0.0
    rev: float = 0.0

    @property
    def mn(self) -> float:
        return min(self.fwd, self.rev)

    def score(self, d: str) -> float:
        return {"FWD": self.fwd, "REV": self.rev, "MIN": self.mn}[d]


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "ja"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--termdict", default="termdict_verified.json")
    ap.add_argument("--min-term-score", type=float, default=0.5)
    ap.add_argument("--target-spec", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--no-prefer-medical", action="store_true")
    ap.add_argument("--out", default="xnli2")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    loader = CorpusLoader(a.root, prefer_medical=not a.no_prefer_medical)
    scorer = Scorer(a.batch_size)
    rows: List[Row] = []
    meta = {}

    for lang in a.langs:
        print(f"\n[{LANG_NAMES[lang]}]")
        td = load_termdict(a.termdict, lang, a.min_term_score)
        data = loader.load(lang, a.n, list(td.keys()), rng)
        if not data:
            print("  사용 가능한 쌍 없음 — 건너뜀")
            continue
        corr = Corrupter(lang, td, rng)
        others = [fo for _, fo, _ in data]

        batch: List[Row] = []
        for ko, fo, _src in data:
            batch.append(Row(lang, "CORRECT", "-", ko, fo))
            m = corr.mismatch(fo, others)
            if m:
                batch.append(Row(lang, "MISMATCH", "-", ko, m))
            nc = corr.num_change(fo)
            if nc:
                batch.append(Row(lang, "NUM_CHANGE", "-", ko, nc))
            tr = corr.truncate(fo)
            if tr:
                batch.append(Row(lang, "TRUNCATE", "-", ko, tr))
            ts = corr.term_swap(ko, fo)
            if ts:
                batch.append(Row(lang, "TERM_SWAP", ts[1], ko, ts[0]))

        print(f"  채점 {len(batch)}건 × 양방향")
        fwd = scorer.entail([r.ko for r in batch], [r.fo for r in batch])
        rev = scorer.entail([r.fo for r in batch], [r.ko for r in batch])
        for r, f, v in zip(batch, fwd, rev):
            r.fwd, r.rev = f, v
        rows.extend(batch)
        meta[lang] = {"termdict": len(td), "n": len(data)}

    # -- 집계 ---------------------------------------------------------------
    summary = {}
    for lang in meta:
        sel = [r for r in rows if r.lang == lang]
        pos = [r for r in sel if r.kind == "CORRECT"]
        if not pos:
            continue
        taus = {d: tau_for_spec([r.score(d) for r in pos], a.target_spec)
                for d in DIRS}
        print("\n" + "=" * 78)
        print(f"[{LANG_NAMES[lang]}]  정답쌍 {len(pos)}건  "
              f"정상 통과율 {a.target_spec:.0%} 통일")
        print("=" * 78)
        print("  정답쌍 평균 entailment  " + "  ".join(
            f"{d} {sum(r.score(d) for r in pos)/len(pos):.3f}" for d in DIRS))
        print(f"  보정 임계값             " + "  ".join(
            f"{d} {taus[d]:.3f}" for d in DIRS))
        print()
        print(f"  {'훼손 유형':<14}{'n':>5}" + "".join(f"{d:>22}" for d in DIRS))
        print(f"  {'':<14}{'':>5}" + "".join(f"{'탐지율   AUROC':>22}" for d in DIRS))
        summary[lang] = {"taus": {d: round(taus[d], 4) for d in DIRS},
                         "n_correct": len(pos), "by_kind": {}}
        for kind in KINDS:
            k_rows = [r for r in sel if r.kind == kind]
            if not k_rows:
                print(f"  {kind:<14}{0:>5}   (표본 없음)")
                continue
            line = f"  {kind:<14}{len(k_rows):>5}"
            summary[lang]["by_kind"][kind] = {"n": len(k_rows)}
            for d in DIRS:
                det = sum(1 for r in k_rows if r.score(d) < taus[d]) / len(k_rows)
                au = auroc([r.score(d) for r in k_rows],
                           [r.score(d) for r in pos])
                summary[lang]["by_kind"][kind][d] = {
                    "detect": round(det, 4), "auroc": round(au, 4)}
                line += f"{det:>14.1%}{au:>8.3f}"
            print(line)
            lo, hi = wilson(sum(1 for r in k_rows if r.mn < taus["MIN"]),
                            len(k_rows))
            print(f"  {'':<19}MIN 95% CI [{lo:.0%}, {hi:.0%}]"
                  + ("   [표본 부족]" if len(k_rows) < 30 else ""))

    # -- 판정 ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("판정")
    print("=" * 78)
    for lang, s in summary.items():
        print(f"\n[{LANG_NAMES[lang]}]")
        bk = s["by_kind"]
        tr = bk.get("TRUNCATE")
        if tr:
            f, r = tr["FWD"]["auroc"], tr["REV"]["auroc"]
            print(f"  누락 탐지  FWD {f:.3f} → REV {r:.3f}  ({r-f:+.3f})")
            if r - f > 0.15:
                print("    → 역방향이 누락을 잡습니다. 양방향 채점이 필요합니다.")
            elif r < 0.7:
                print("    → 역방향으로도 누락을 못 잡습니다. 별도 모듈이 필요합니다.")
            else:
                print("    → 방향 전환 효과가 뚜렷하지 않습니다.")
        ts = bk.get("TERM_SWAP")
        if ts:
            best = max(DIRS, key=lambda d: ts[d]["auroc"])
            print(f"  용어 오역  최고 {best} AUROC {ts[best]['auroc']:.3f} "
                  f"(n={ts['n']})")
            if ts[best]["auroc"] < 0.75:
                print("    → 교차 조건에서도 용어 오역은 못 잡습니다."
                      " 한국어 실험과 같은 결론입니다.")
            else:
                print("    → 용어 오역까지 잡힙니다. 표본을 늘려 재확인하세요.")
        else:
            print("  용어 오역  표본 없음 — 사전과 말뭉치의 겹침을 확인하세요.")
        mm = bk.get("MISMATCH")
        if mm:
            print(f"  기본 판별  MIN AUROC {mm['MIN']['auroc']:.3f}")

    print("=" * 78)
    with open(f"{a.out}_report.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    with open(f"{a.out}_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "kind", "detail", "fwd", "rev", "min", "ko", "fo"])
        for r in rows:
            w.writerow([r.lang, r.kind, r.detail, round(r.fwd, 4),
                        round(r.rev, 4), round(r.mn, 4), r.ko[:300], r.fo[:300]])
    print(f"\n[SAVE] {a.out}_report.json / {a.out}_cases.csv")


if __name__ == "__main__":
    main()
