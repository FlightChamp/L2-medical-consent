#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_sections3.py — 섹션분할 vs 통짜 변환 비교 + 변환문 저장 + 통합 검증
=========================================================================
v1 → v2 → v3 수정 이력:
    v2  enable_thinking=False 추가 (hari 는 Qwen3 기반 추론 모델이라
        <think> 영어 사고과정을 먼저 출력한다. v1 은 이게 통째로
        '변환문'으로 저장돼 분량 +1726% 같은 허수를 만들었다)
    v2  시스템/유저 2단 구조 + 기존 real_run.py 프롬프트로 교체
    v3  메타 발화·마크다운 정제 추가 (이 파일)

왜 v3 가 필요한가:
    모델이 "물론입니다. 아래는 원문의 내용을 유지하면서…" 같은 여는 말과
    **굵게**, ---, ## 같은 마크다운을 붙인다.
    이 메타는 섹션마다 붙으므로 섹션분할(섹션 10개 → 인사말 10번)이
    통짜(1번)보다 10배 오염된다. 두 방식 비교가 구조적으로 왜곡된다.
    실제로 v2 에서 근거율이 section 61.9 / whole 77.1 로 갈렸는데
    상당 부분이 이 편향으로 의심된다.

정제 원칙 (test_cleaner.py 로 9개 사례 검증 완료):
    · 여는 말은 '쓰기 행위'를 가리키는 표현과 함께 나올 때만 제거한다.
      '다음은 수술의 위험성입니다' 같은 본문을 지우지 않기 위해서다.
    · '안내드립니다' 처럼 본문에도 흔한 표현은 건드리지 않는다.
    · 마크다운은 기호만 벗기고 글자는 남긴다.
    · 정제 결과가 원본의 15% 미만이면 되돌린다 (사고과정 제거는 유지).
    · 원문 출력(out_raw)도 함께 저장해 언제든 재검증할 수 있게 한다.

측정 지표:
    용어보존   기존 real_run.py 정의 그대로 (상위 20개 용어의 잔존율)
    수치보존   단위 붙은 숫자의 보존율 — 규칙 기반, 재현·설명 확실
               (기존 '사실보존'은 LLM 사실단위 방식이라 프롬프트 없이 재현 불가)
    근거율     변환문 문장이 원문에 근거하는가 (NLI, k=5, 정방향)
    커버리지   원문 문장이 변환문에 남아 있는가 (NLI, k=5, 역방향)
    복사유사도 원문-변환문 문자 3-gram Jaccard. 0.90 이상이면 사실상 복사
    평이화     전문용어·한자어·명사형·문장길이
    메타제거   정제로 걷어낸 글자 수 (오염 규모 자체가 결과다)

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python rerun_sections3.py --docs doc1        # 먼저 1건으로 점검
    python rerun_sections3.py                    # 전체 (약 50분)
    python rerun_sections3.py --eval-only        # 재생성 없이 검증만
    python rerun_sections3.py --show doc1        # 정제 전후 비교 출력
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
DOCDIR = os.path.expanduser("~/이윤우/docs")
OUTDIR = os.path.expanduser("~/이윤우/outputs")

# ── 기존 real_run.py 의 프롬프트를 그대로 사용 ──
SYS_PROMPT = ("당신은 한국어 의료 어시스턴트입니다. "
              "원문에 없는 내용은 절대 추가하지 마세요.")
USER_PROMPT = ("다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
               "수술 이름, 모든 수치, 발생 조건, 시간·기간 정보는 하나도 빠짐없이 "
               "유지해 주세요.\n\n{text}")


# ===========================================================================
# 출력 정제 (test_cleaner.py 검증 완료)
# ===========================================================================

class Cleaner:
    THINK = re.compile(r"<think>.*?</think>", re.S)

    # 여는 말: '지금부터 다시 쓰겠다'는 취지. 쓰기 행위를 가리키는 말과
    # 함께 나올 때만 잡는다.
    META_OPEN = [
        r"^물론입니다", r"^알겠습니다", r"^네[,.]?\s*$",
        r"아래(는|와\s*같이).{0,40}(작성|정리|변환|다시\s*(썼|쓴|쓰))",
        r"다음은.{0,40}(쉬운|다시\s*(쓴|작성)|변환|바꾼)",
        r"원문(의|에).{0,30}(유지|바탕).{0,20}(작성|다시)",
        r"^다시\s*(정리|작성)(해|한)",
        r"원문에\s*있는\s*모든",
    ]
    # 닫는 말: 사용자 응대 표현
    META_CLOSE = [
        r"필요하시?면.{0,20}(말씀|알려|문의|주세요)",
        r"도움이\s*되(었|시|길|기를)",
        r"추가.{0,10}(궁금|질문|문의)",
        r"더\s*(자세한|궁금).{0,20}(설명|안내|점).{0,20}(필요|원하|있으)",
        r"드릴\s*수\s*있",
        r"말씀해\s*주세요",
        r"참고하시(기|어)",
    ]
    OPEN_RE = [re.compile(p) for p in META_OPEN]
    CLOSE_RE = [re.compile(p) for p in META_CLOSE]

    RULE_LINE = re.compile(r"^\s*([-*=_]{3,}|─{3,})\s*$")
    HEADING = re.compile(r"^\s*#{1,6}\s*")
    BOLD = re.compile(r"\*\*(.+?)\*\*")
    ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    BULLET = re.compile(r"^\s*[-*•▪◦]\s+")
    QUOTE = re.compile(r"^\s*>\s?")
    CODE = re.compile(r"`+")

    @classmethod
    def clean(cls, raw: str) -> Tuple[str, int, List[str]]:
        """(정제문, 제거한 글자 수, 사유 목록)"""
        reasons: List[str] = []
        t = cls.THINK.sub("", raw or "")
        if "</think>" in t:
            t = t.split("</think>")[-1]
            reasons.append("think")
        base = t.strip()
        before = len(base)

        lines = [l.rstrip() for l in t.split("\n")]

        # 여는 말 뒤에 구분선이 오는 흔한 패턴은 구분선까지 잘라낸다
        for i, l in enumerate(lines[:6]):
            if cls.RULE_LINE.match(l):
                head = " ".join(lines[:i])
                if any(p.search(head) for p in cls.OPEN_RE):
                    lines = lines[i + 1:]
                    reasons.append("preamble+rule")
                break

        while lines:
            first = lines[0].strip()
            if not first or cls.RULE_LINE.match(first):
                lines.pop(0)
                continue
            if any(p.search(first) for p in cls.OPEN_RE) and len(lines) > 1:
                lines.pop(0)
                reasons.append("open")
                continue
            break

        while lines:
            last = lines[-1].strip()
            if not last or cls.RULE_LINE.match(last):
                lines.pop()
                continue
            if any(p.search(last) for p in cls.CLOSE_RE) and len(lines) > 1:
                lines.pop()
                reasons.append("close")
                continue
            break

        out = []
        for l in lines:
            if cls.RULE_LINE.match(l):
                continue
            l = cls.HEADING.sub("", l)
            l = cls.BOLD.sub(r"\1", l)
            l = cls.ITALIC.sub(r"\1", l)
            l = cls.BULLET.sub("", l)
            l = cls.QUOTE.sub("", l)
            l = cls.CODE.sub("", l)
            if l.strip():
                out.append(l.strip())

        res = "\n".join(out).strip()
        # 과잉 삭제 방어. 되돌리더라도 사고과정 제거는 유지한다.
        if not res or (before > 100 and len(res) < 0.15 * before):
            return (base, 0, reasons + ["fallback"])
        return (res, before - len(res), reasons)


# ===========================================================================
# 텍스트 처리 (기존 real_run.py 규칙 유지)
# ===========================================================================

SEC = re.compile(r"^\s*(\d+)\s*[.．]\s*(.+)$")
DROP = re.compile(
    r"유\s*무|□|■|☐|☑|병록\s*번호|등록\s*번호|성명|생년월일|성별\s*/?\s*나이|진료과|주치의|"
    r"시행\s*예정|병동|병실|집도의|참여\s*의료진|전문의|전문\s*과목|서명|보호자|"
    r"^\s*\(?\s*[좌우]\s*\)?\s*$|^\s*년\s*월\s*일|환자의\s*현재\s*상태")


def extract_sections(raw: str) -> List[Tuple[str, str]]:
    lines = [re.sub(r"\s+", " ", l).strip() for l in raw.split("\n")]
    secs: List[Tuple[str, str]] = []
    cur: Optional[str] = None
    buf: List[str] = []
    for l in lines:
        if not l:
            continue
        m = SEC.match(l)
        if m:
            if cur and buf:
                secs.append((cur, " ".join(buf)))
            cur, buf = f"{m.group(1)}. {m.group(2)}", []
            continue
        if cur is None or DROP.search(l):
            continue
        if len(re.findall(r"[가-힣]", l)) < 5:
            continue
        buf.append(l)
    if cur and buf:
        secs.append((cur, " ".join(buf)))
    return secs


class Norm:
    CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @classmethod
    def clean(cls, t) -> str:
        t = unicodedata.normalize("NFKC", str(t))
        t = cls.CTRL.sub(" ", t)
        return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls.BOUND.split(str(t)):
            s = Norm.clean(p or "")
            if s and not cls.NUM_PREFIX.match(s) and len(s) >= 10:
                out.append(s)
        return out


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


def jaccard3(a: str, b: str) -> float:
    ga, gb = ngrams(a, 3), ngrams(b, 3)
    return len(ga & gb) / max(len(ga | gb), 1)


def topk(hyp: str, chunks: Sequence[str], k: int) -> List[str]:
    hg = ngrams(hyp, 2)
    scored = []
    for i, c in enumerate(chunks):
        g = ngrams(c, 2)
        scored.append((len(hg & g) / max(len(hg | g), 1), i))
    scored.sort(key=lambda x: -x[0])
    return [chunks[i] for _, i in scored[:k]]


# ===========================================================================
# 규칙 기반 지표
# ===========================================================================

STOP = set("""환자 수술 시술 검사 치료 경우 가능 필요 발생 대한 위해 통해 이후 이전 다음 아래 관련
설명 동의 내용 방법 사항 결과 상태 정도 이상 이하 미만 대해 위한 등의 또는 그리고 하지만 있습니다
없습니다 합니다 됩니다 입니다 병원 의사 의료진 서명 보호자 성명 날짜 기록 확인 이해 질문 답변
본인 가족 담당 예정 시행 실시 있으며 하며 그러나 따라서 대체방법 주의사항""".split())


def extract_terms(src: str, k: int = 20) -> List[str]:
    """기존 real_run.py 정의 그대로 (상위 20개, 길이 우선)."""
    c: Dict[str, int] = {}
    for w in re.findall(r"[가-힣]{3,10}", src):
        if w in STOP:
            continue
        c[w] = c.get(w, 0) + 1
    return sorted(c, key=lambda w: (-len(w), -c[w]))[:k]


NUM_UNIT = re.compile(
    r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번|세)")


def numeric_units(t: str) -> List[str]:
    return [f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(t)]


MED_TERMS = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양", "감염",
    "출혈", "혈전", "색전증", "마취", "수혈", "봉합", "절제", "이식", "배액관",
    "합병증", "후유증", "부작용", "재발", "불유합", "성대마비", "저칼슘혈증",
]
HANJA_SUFFIX = re.compile(
    r"[가-힣]{1,4}(증|염|술|양성|성|적|화|법|부위|부|경|관|제|액|압|통|"
    r"기능|장애|손상|절제|봉합|주입|투여)(?=[\s,.)]|$)")
NOMINAL = re.compile(r"[가-힣]+(함|됨|임|음)(?=[\s,.)]|$)")


def readability(t: str) -> Dict[str, float]:
    sents = Splitter.split(t)
    if not sents:
        return {"sent_len": 0.0, "term": 0.0, "hanja": 0.0, "nominal": 0.0}
    words = sum(len(s.split()) for s in sents) or 1
    j = " ".join(sents)
    return {
        "sent_len": words / len(sents),
        "term": 100 * sum(j.count(x) for x in MED_TERMS) / words,
        "hanja": 100 * len(HANJA_SUFFIX.findall(j)) / words,
        "nominal": 100 * len(NOMINAL.findall(j)) / words,
    }


# ===========================================================================
# 생성
# ===========================================================================

class Generator:
    def __init__(self, gpu: int, user_prompt: str, cap: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.user_prompt = user_prompt
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

    def run(self, text: str) -> str:
        """정제 전 원문 출력을 그대로 반환한다. 정제는 호출부에서 한다."""
        torch = self.torch
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": self.user_prompt.format(text=text)}]
        # enable_thinking=False 가 핵심.
        # 없으면 <think> 사고과정(영어)이 그대로 출력된다.
        try:
            t = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            t = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(t, return_tensors="pt").to(self.model.device)
        n_in = inp.input_ids.shape[1]
        with torch.no_grad():
            o = self.model.generate(**inp, max_new_tokens=self.cap,
                                    do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0][n_in:], skip_special_tokens=True).strip()

    def free(self):
        del self.model
        self.torch.cuda.empty_cache()


# ===========================================================================
# NLI 검증
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
        print(f"[NLI] id2label={id2} entail={self.i_ent}")

    def entail(self, prem: List[str], hyp: List[str]) -> List[float]:
        torch = self.torch
        out: List[float] = []
        for i in range(0, len(prem), self.bs):
            enc = self.tok(prem[i:i + self.bs], hyp[i:i + self.bs],
                           truncation=True, padding=True, max_length=256,
                           return_tensors="pt")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits, dim=-1)
            out.extend(p[:, self.i_ent].tolist())
        return out

    def rate(self, hyps: List[str], chunks: List[str], k: int,
             tau: float) -> float:
        """hyps 중 chunks 에 의해 함의되는 비율(%)."""
        if not hyps or not chunks:
            return float("nan")
        prem, hy, owner = [], [], []
        for i, h in enumerate(hyps):
            for c in topk(h, chunks, k):
                prem.append(c)
                hy.append(h)
                owner.append(i)
        sc = self.entail(prem, hy)
        best: Dict[int, float] = {}
        for o, s in zip(owner, sc):
            best[o] = max(best.get(o, -1.0), s)
        ok = sum(1 for i in range(len(hyps)) if best.get(i, 0.0) >= tau)
        return 100 * ok / len(hyps)


# ===========================================================================
# 파이프라인
# ===========================================================================

def gen_path(doc: str, mode: str) -> str:
    return os.path.join(OUTDIR, f"{doc}__{mode}.json")


def stage_generate(docs: List[str], modes: List[str], a) -> None:
    todo = [(d, m) for d in docs for m in modes
            if a.force or not os.path.exists(gen_path(d, m))]
    if not todo:
        print("[GEN] 생성할 것이 없습니다 (--force 로 재생성)")
        return
    gen = Generator(a.gpu, a.user_prompt, a.max_new)
    t0 = time.time()
    for n, (doc, mode) in enumerate(todo, 1):
        with open(os.path.join(a.docdir, f"{doc}.txt"),
                  encoding="utf-8", errors="replace") as f:
            raw_doc = f.read()
        secs = extract_sections(raw_doc)
        if not secs:
            print(f"  [{doc}] 섹션 추출 실패 — 건너뜀")
            continue
        rec = {"doc": doc, "mode": mode, "model": GEN_MODEL,
               "system": SYS_PROMPT, "user": a.user_prompt, "sections": []}
        bodies = ([(t, b) for t, b in secs] if mode == "section"
                  else [("(전체)", " ".join(b for _, b in secs))])
        for title, body in bodies:
            raw_out = gen.run(body)
            cleaned, removed, reasons = Cleaner.clean(raw_out)
            rec["sections"].append({
                "title": title, "src": body,
                "out": cleaned, "out_raw": raw_out,
                "meta_removed": removed, "meta_reasons": reasons})
        os.makedirs(OUTDIR, exist_ok=True)
        with open(gen_path(doc, mode), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        el = time.time() - t0
        si = sum(len(s["src"]) for s in rec["sections"])
        so = sum(len(s["out"]) for s in rec["sections"])
        mr = sum(s["meta_removed"] for s in rec["sections"])
        print(f"  [{n}/{len(todo)}] {doc}/{mode}: 섹션 {len(rec['sections'])} "
              f"/ {si}자 → {so}자 ({100 * (so - si) / max(si, 1):+.0f}%) "
              f"메타제거 {mr}자  "
              f"[{el:.0f}s, 남은 예상 {el / n * (len(todo) - n):.0f}s]",
              flush=True)
    gen.free()


def stage_eval(docs: List[str], modes: List[str], a) -> List[dict]:
    nli = NLI(a.gpu, a.batch)
    rows: List[dict] = []
    print(f"\n  {'문서':<7}{'방식':<9}{'용어':>7}{'수치':>7}{'근거':>7}"
          f"{'커버':>7}{'유사':>7}{'메타':>7}")
    for doc in docs:
        for mode in modes:
            p = gen_path(doc, mode)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            src = " ".join(s["src"] for s in rec["sections"])
            out = " ".join(s["out"] for s in rec["sections"])
            meta = sum(s.get("meta_removed", 0) for s in rec["sections"])

            terms = extract_terms(src, a.terms_k)
            t_keep = 100 * sum(1 for t in terms if t in out) / max(len(terms), 1)
            nums = sorted(set(numeric_units(src)))
            n_keep = (100 * sum(1 for x in nums if x in out) / len(nums)
                      if nums else None)

            s_sents, o_sents = Splitter.split(src), Splitter.split(out)
            ground = nli.rate(o_sents, build_chunks(s_sents), a.k, a.tau)
            cover = nli.rate(s_sents, build_chunks(o_sents), a.k, a.tau)
            rs, ro = readability(src), readability(out)

            row = {
                "doc": doc, "mode": mode,
                "src_chars": len(src), "out_chars": len(out),
                "meta_removed": meta,
                "delta_chars": 100 * (len(out) - len(src)) / max(len(src), 1),
                "term_keep": round(t_keep, 1),
                "num_keep": round(n_keep, 1) if n_keep is not None else None,
                "n_nums": len(nums),
                "ground": round(ground, 1), "cover": round(cover, 1),
                "copy_sim": round(jaccard3(src, out), 3),
                **{f"src_{k}": round(v, 2) for k, v in rs.items()},
                **{f"out_{k}": round(v, 2) for k, v in ro.items()},
            }
            rows.append(row)
            nk = f"{n_keep:.1f}" if n_keep is not None else "—"
            print(f"  {doc:<7}{mode:<9}{t_keep:>7.1f}{nk:>7}"
                  f"{ground:>7.1f}{cover:>7.1f}{row['copy_sim']:>7.2f}"
                  f"{meta:>7}", flush=True)
    return rows


def report(rows: List[dict]):
    if not rows:
        print("결과 없음")
        return
    print("\n" + "=" * 100)
    print("문서별 결과")
    print("=" * 100)
    print(f"  {'문서':<7}{'방식':<9}{'원문자':>7}{'변환자':>7}{'증감':>7}"
          f"{'용어':>7}{'수치':>7}{'근거율':>8}{'커버':>7}{'유사도':>7}{'메타':>7}")
    for r in sorted(rows, key=lambda x: (x["doc"], x["mode"])):
        nk = f"{r['num_keep']:.1f}" if r["num_keep"] is not None else "—"
        print(f"  {r['doc']:<7}{r['mode']:<9}{r['src_chars']:>7}"
              f"{r['out_chars']:>7}{r['delta_chars']:>+6.0f}%"
              f"{r['term_keep']:>7.1f}{nk:>7}{r['ground']:>8.1f}"
              f"{r['cover']:>7.1f}{r['copy_sim']:>7.2f}{r['meta_removed']:>7}")

    modes = sorted({r["mode"] for r in rows})
    if len(modes) < 2:
        return
    print("\n" + "=" * 100)
    print("방식 비교")
    print("=" * 100)
    keys = [("delta_chars", "분량 증감%"), ("term_keep", "용어보존"),
            ("num_keep", "수치보존"), ("ground", "근거율"),
            ("cover", "커버리지"), ("copy_sim", "복사유사도"),
            ("meta_removed", "메타제거자수"),
            ("out_term", "전문용어비율"), ("out_hanja", "한자어비율"),
            ("out_sent_len", "평균어절")]
    print(f"  {'지표':<14}" + "".join(f"{m:>12}" for m in modes) + f"{'차이':>10}")
    for k, label in keys:
        vals = {}
        for m in modes:
            xs = [r[k] for r in rows if r["mode"] == m and r.get(k) is not None]
            vals[m] = sum(xs) / len(xs) if xs else float("nan")
        diff = vals[modes[0]] - vals[modes[1]]
        print(f"  {label:<14}" + "".join(f"{vals[m]:>12.2f}" for m in modes)
              + f"{diff:>+10.2f}")

    print("\n  길이별 분량 증감 (원문 2,000자 경계)")
    for m in modes:
        short = [r["delta_chars"] for r in rows
                 if r["mode"] == m and r["src_chars"] < 2000]
        long_ = [r["delta_chars"] for r in rows
                 if r["mode"] == m and r["src_chars"] >= 2000]
        s = sum(short) / len(short) if short else float("nan")
        l = sum(long_) / len(long_) if long_ else float("nan")
        gap = abs(s - l) if short and long_ else float("nan")
        print(f"    {m:<10} 짧은 문서 {s:>+7.1f}%({len(short)}건)  "
              f"긴 문서 {l:>+7.1f}%({len(long_)}건)  격차 {gap:>6.1f}%p")
    print("  → 격차가 줄면 섹션 분할이 길이 의존성을 완화한 것입니다.")

    copies = [r for r in rows if r["copy_sim"] >= 0.90]
    if copies:
        print(f"\n  [!] 복사 의심 {len(copies)}건: "
              + ", ".join(f"{r['doc']}/{r['mode']}" for r in copies))
    print("=" * 100)


def show(doc: str, modes: List[str]):
    """정제 전후를 눈으로 확인한다."""
    for mode in modes:
        p = gen_path(doc, mode)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        print("=" * 78)
        print(f"{doc} / {mode}")
        print("=" * 78)
        for s in rec["sections"][:3]:
            print(f"\n── {s['title']}  (메타제거 {s.get('meta_removed', 0)}자, "
                  f"사유 {s.get('meta_reasons') or '없음'})")
            print(f"  원문   : {s['src'][:110]}")
            print(f"  정제전 : {s.get('out_raw', '')[:180].replace(chr(10), ' ⏎ ')}")
            print(f"  정제후 : {s['out'][:180].replace(chr(10), ' ⏎ ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=["section", "whole"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--terms-k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--prompt-file", default=None,
                    help="유저 프롬프트 교체 ({text} 자리표시자 필요)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--show", default=None, help="정제 전후만 출력하고 종료")
    ap.add_argument("--out", default="rerun_sections3")
    a = ap.parse_args()

    if a.show:
        show(a.show, a.modes)
        return

    a.user_prompt = USER_PROMPT
    if a.prompt_file:
        with open(a.prompt_file, encoding="utf-8") as f:
            a.user_prompt = f.read()
        print(f"[PROMPT] {a.prompt_file} 사용")
    else:
        print("[PROMPT] 기존 real_run.py 프롬프트 사용")

    if a.docs:
        docs = a.docs
    else:
        docs = sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(a.docdir, "*.txt"))
                      if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")
    print(f"[PLAN] 문서 {len(docs)}건 × 방식 {len(a.modes)}종 = "
          f"{len(docs) * len(a.modes)} 실행")

    os.makedirs(OUTDIR, exist_ok=True)
    if not a.eval_only:
        stage_generate(docs, a.modes, a)
    rows = stage_eval(docs, a.modes, a)
    report(rows)

    with open(f"{a.out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_metrics.json")
    print(f"       변환문은 {OUTDIR}/ 에 정제 전(out_raw)·후(out) 모두 저장됩니다")


if __name__ == "__main__":
    main()
