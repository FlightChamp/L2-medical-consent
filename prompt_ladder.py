#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_ladder.py — ③ 변환 프롬프트 누적 비교
=============================================
설계:
    오늘 확인된 실패 양상을 각각 겨냥해 지시를 하나씩 쌓는다.
    한 번에 다 넣으면 무엇이 효과였는지 분리되지 않는다.

      P0  현재 프롬프트 (기준선)
      P1  + 용어 보존          ← 용어보존 39.6% 대응
      P2  + 창작 금지·빈칸 유지 ← 맹검 판정 24건 중 창작 11건 대응
      P3  + 분량 유지          ← 긴 문서 -28.2% 압축 대응

판정 기준 (실험 전에 못박는다. 나중에 유리하게 해석하지 않기 위함):
    · 복사유사도 0.6 이상  → 실패. '요약하지 말라'가 복사를 유발한 경우
      (hari 기준선 0.30, Medical-Llama 복사 사례 0.81~0.86)
    · [미기재] 남발        → 실패. 원문 빈칸 수보다 많이 찍히면 오작동
    · 근거율이 P0 미만     → 그 조건 기각

    지시를 늘리면 좋아 보이는 지표(용어보존 등)가 오르기 쉬우므로,
    위 세 가지를 함께 보지 않으면 잘못된 결론에 이른다.
    Medical-Llama 가 원문을 복사하고도 용어보존 100% 를 기록한 것이 그 예다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python prompt_ladder.py --docs doc1        # 먼저 1건으로 점검
    python prompt_ladder.py                    # 전체 (약 30분)
    python prompt_ladder.py --eval-only
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
OUTDIR = os.path.expanduser("~/이윤우/outputs_prompt")

SYS_PROMPT = ("당신은 한국어 의료 어시스턴트입니다. "
              "원문에 없는 내용은 절대 추가하지 마세요.")

BASE = ("다음 수술 동의서를 환자가 이해하기 쉬운 말로 다시 써 주세요. "
        "수술 이름, 모든 수치, 발생 조건, 시간·기간 정보는 하나도 빠짐없이 "
        "유지해 주세요.")

ADD_TERM = ("\n의학 용어는 지우거나 다른 말로 바꾸지 마세요. "
            "용어를 그대로 쓴 뒤 괄호 안에 쉬운 설명을 덧붙이세요. "
            "예: 불유합(뼈가 제대로 붙지 않는 것)")

ADD_NOCREATE = ("\n원문에 없는 사실·수치·조건을 새로 만들지 마세요. "
                "원문이 비어 있는 자리는 채우지 말고 [미기재]로 표시하세요. "
                "원문에 없는 항목을 새로 만들어 넣지 마세요.")

ADD_KEEP = ("\n요약하지 마세요. 원문의 모든 항목을 빠짐없이 옮기세요. "
            "다만 원문 문장을 그대로 베끼지 말고 쉬운 말로 바꿔 쓰세요.")

PROMPTS = {
    "P0": BASE,
    "P1": BASE + ADD_TERM,
    "P2": BASE + ADD_TERM + ADD_NOCREATE,
    "P3": BASE + ADD_TERM + ADD_NOCREATE + ADD_KEEP,
    "P4": BASE + ADD_NOCREATE + ADD_KEEP,   # 용어 지시만 제외
}


# ===========================================================================
# 텍스트 처리 (rerun_sections3.py 와 동일 기준)
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


def clean(t) -> str:
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", t).strip()


class Splitter:
    NUM_PREFIX = re.compile(r"^\s*(\d+|[가-힣])\s*[.)]\s*$")
    BOUND = re.compile(r"(?<=[다요음임함])\.\s+|(?<=[.!?])\s+|\n+")

    @classmethod
    def split(cls, t: str) -> List[str]:
        out = []
        for p in cls.BOUND.split(str(t)):
            s = clean(p or "")
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
    sc = []
    for i, c in enumerate(chunks):
        g = ngrams(c, 2)
        sc.append((len(hg & g) / max(len(hg | g), 1), i))
    sc.sort(key=lambda x: -x[0])
    return [chunks[i] for _, i in sc[:k]]


# ===========================================================================
# 지표
# ===========================================================================

STOP = set("""환자 수술 시술 검사 치료 경우 가능 필요 발생 대한 위해 통해 이후 이전 다음 아래 관련
설명 동의 내용 방법 사항 결과 상태 정도 이상 이하 미만 대해 위한 등의 또는 그리고 하지만 있습니다
없습니다 합니다 됩니다 입니다 병원 의사 의료진 서명 보호자 성명 날짜 기록 확인 이해 질문 답변
본인 가족 담당 예정 시행 실시 있으며 하며 그러나 따라서 대체방법 주의사항""".split())


def extract_terms(src: str, k: int = 20) -> List[str]:
    c: Dict[str, int] = {}
    for w in re.findall(r"[가-힣]{3,10}", src):
        if w in STOP:
            continue
        c[w] = c.get(w, 0) + 1
    return sorted(c, key=lambda w: (-len(w), -c[w]))[:k]


NUM_UNIT = re.compile(
    r"(?<![\d.])(\d{1,4})\s*(주|일|개월|시간|분|년|%|퍼센트|cc|ml|mg|회|명|번|세)")
BLANK_MARK = re.compile(r"\[미기재\]|\[\s*\]|\[시간\]|\[수술\s*이름\]")
ORIG_BLANK = re.compile(
    r"[（(\[〔]\s*[)）\]〕]|[_＿]{2,}"
    r"|약\s+(?=정도|가량|이내|이상|이하|동안|쯤)")


def numeric_units(t: str) -> List[str]:
    return [f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(t)]


def numeric_halluc(src: str, out: str) -> List[str]:
    s = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(src)}
    o = {f"{m.group(1)}{m.group(2)}" for m in NUM_UNIT.finditer(out)}
    return sorted(o - s)


MED_TERMS = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "골절", "탈구", "종양", "낭종", "궤양", "협착", "파열", "농양", "감염",
    "출혈", "혈전", "색전증", "마취", "수혈", "봉합", "절제", "이식", "배액관",
    "합병증", "후유증", "부작용", "재발", "불유합", "성대마비", "저칼슘혈증",
]
HANJA = re.compile(
    r"[가-힣]{1,4}(증|염|술|양성|성|적|화|법|부위|부|경|관|제|액|압|통|"
    r"기능|장애|손상|절제|봉합|주입|투여)(?=[\s,.)]|$)")


def readability(t: str) -> Dict[str, float]:
    s = Splitter.split(t)
    if not s:
        return {"sent_len": 0.0, "term": 0.0, "hanja": 0.0}
    w = sum(len(x.split()) for x in s) or 1
    j = " ".join(s)
    return {"sent_len": w / len(s),
            "term": 100 * sum(j.count(x) for x in MED_TERMS) / w,
            "hanja": 100 * len(HANJA.findall(j)) / w}


# ===========================================================================
# 출력 정제 (rerun_sections3.py 의 Cleaner 와 동일)
# ===========================================================================

class Cleaner:
    THINK = re.compile(r"<think>.*?</think>", re.S)
    OPEN_RE = [re.compile(p) for p in (
        r"^물론입니다", r"^알겠습니다", r"^네[,.]?\s*$",
        r"아래(는|와\s*같이).{0,40}(작성|정리|변환|다시\s*(썼|쓴|쓰))",
        r"다음은.{0,40}(쉬운|다시\s*(쓴|작성)|변환|바꾼)",
        r"원문(의|에).{0,30}(유지|바탕).{0,20}(작성|다시)",
        r"^다시\s*(정리|작성)(해|한)", r"원문에\s*있는\s*모든")]
    CLOSE_RE = [re.compile(p) for p in (
        r"필요하시?면.{0,20}(말씀|알려|문의|주세요)",
        r"도움이\s*되(었|시|길|기를)", r"추가.{0,10}(궁금|질문|문의)",
        r"더\s*(자세한|궁금).{0,20}(설명|안내|점).{0,20}(필요|원하|있으)",
        r"드릴\s*수\s*있", r"말씀해\s*주세요", r"참고하시(기|어)")]
    RULE = re.compile(r"^\s*([-*=_]{3,}|─{3,})\s*$")
    HEAD = re.compile(r"^\s*#{1,6}\s*")
    BOLD = re.compile(r"\*\*(.+?)\*\*")
    ITAL = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    BULL = re.compile(r"^\s*[-*•▪◦]\s+")
    QUOT = re.compile(r"^\s*>\s?")
    CODE = re.compile(r"`+")

    @classmethod
    def clean(cls, raw: str) -> Tuple[str, int]:
        t = cls.THINK.sub("", raw or "")
        if "</think>" in t:
            t = t.split("</think>")[-1]
        base = t.strip()
        before = len(base)
        lines = [l.rstrip() for l in t.split("\n")]
        for i, l in enumerate(lines[:6]):
            if cls.RULE.match(l):
                if any(p.search(" ".join(lines[:i])) for p in cls.OPEN_RE):
                    lines = lines[i + 1:]
                break
        while lines:
            f = lines[0].strip()
            if not f or cls.RULE.match(f):
                lines.pop(0)
                continue
            if any(p.search(f) for p in cls.OPEN_RE) and len(lines) > 1:
                lines.pop(0)
                continue
            break
        while lines:
            l = lines[-1].strip()
            if not l or cls.RULE.match(l):
                lines.pop()
                continue
            if any(p.search(l) for p in cls.CLOSE_RE) and len(lines) > 1:
                lines.pop()
                continue
            break
        out = []
        for l in lines:
            if cls.RULE.match(l):
                continue
            for pat, rep in ((cls.HEAD, ""), (cls.BULL, ""),
                             (cls.QUOT, ""), (cls.CODE, "")):
                l = pat.sub(rep, l)
            l = cls.BOLD.sub(r"\1", l)
            l = cls.ITAL.sub(r"\1", l)
            if l.strip():
                out.append(l.strip())
        res = "\n".join(out).strip()
        if not res or (before > 100 and len(res) < 0.15 * before):
            return (base, 0)
        return (res, before - len(res))


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

    def run(self, user_prompt: str, text: str) -> str:
        torch = self.torch
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"{user_prompt}\n\n{text}"}]
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
        return self.tok.decode(o[0][n:], skip_special_tokens=True).strip()

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

def path_of(doc: str, cond: str) -> str:
    return os.path.join(OUTDIR, f"{doc}__{cond}.json")


def stage_generate(docs: List[str], conds: List[str], a):
    todo = [(d, c) for d in docs for c in conds
            if a.force or not os.path.exists(path_of(d, c))]
    if not todo:
        print("[GEN] 생성할 것이 없습니다 (--force 로 재생성)")
        return
    gen = Generator(a.gpu, a.max_new)
    t0 = time.time()
    for n, (doc, cond) in enumerate(todo, 1):
        with open(os.path.join(a.docdir, f"{doc}.txt"),
                  encoding="utf-8", errors="replace") as f:
            secs = extract_sections(f.read())
        if not secs:
            print(f"  [{doc}] 섹션 추출 실패")
            continue
        body = " ".join(b for _, b in secs)
        raw = gen.run(PROMPTS[cond], body)
        out, removed = Cleaner.clean(raw)
        os.makedirs(OUTDIR, exist_ok=True)
        with open(path_of(doc, cond), "w", encoding="utf-8") as f:
            json.dump({"doc": doc, "cond": cond, "model": GEN_MODEL,
                       "prompt": PROMPTS[cond], "src": body,
                       "out": out, "out_raw": raw,
                       "meta_removed": removed}, f, ensure_ascii=False, indent=2)
        el = time.time() - t0
        print(f"  [{n}/{len(todo)}] {doc}/{cond}: {len(body)}자 → {len(out)}자 "
              f"({100*(len(out)-len(body))/max(len(body),1):+.0f}%) "
              f"[{el:.0f}s, 남은 예상 {el/n*(len(todo)-n):.0f}s]", flush=True)
    gen.free()


def stage_eval(docs: List[str], conds: List[str], a) -> List[dict]:
    nli = NLI(a.gpu, a.batch)
    rows = []
    for doc in docs:
        for cond in conds:
            p = path_of(doc, cond)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            src, out = r["src"], r["out"]
            terms = extract_terms(src)
            nums = sorted(set(numeric_units(src)))
            ss, os_ = Splitter.split(src), Splitter.split(out)
            rs, ro = readability(src), readability(out)
            rows.append({
                "doc": doc, "cond": cond,
                "src_chars": len(src), "out_chars": len(out),
                "delta": 100 * (len(out) - len(src)) / max(len(src), 1),
                "term_keep": round(100 * sum(1 for t in terms if t in out)
                                   / max(len(terms), 1), 1),
                "num_keep": round(100 * sum(1 for x in nums if x in out)
                                  / len(nums), 1) if nums else None,
                "n_halluc": len(numeric_halluc(src, out)),
                "ground": round(nli.rate(os_, build_chunks(ss), a.k, a.tau), 1),
                "cover": round(nli.rate(ss, build_chunks(os_), a.k, a.tau), 1),
                "copy_sim": round(jaccard3(src, out), 3),
                "blank_mark": len(BLANK_MARK.findall(out)),
                "orig_blank": len(ORIG_BLANK.findall(src)),
                "out_term": round(ro["term"], 2),
                "out_hanja": round(ro["hanja"], 2),
                "out_sent_len": round(ro["sent_len"], 2),
            })
            print(f"  {doc:<7}{cond:<5} 용어{rows[-1]['term_keep']:>6.1f} "
                  f"근거{rows[-1]['ground']:>6.1f} 커버{rows[-1]['cover']:>6.1f} "
                  f"복사{rows[-1]['copy_sim']:>6.2f} "
                  f"미기재{rows[-1]['blank_mark']:>3}/{rows[-1]['orig_blank']}",
                  flush=True)
    return rows


def report(rows: List[dict], conds: List[str]):
    if not rows:
        print("결과 없음")
        return
    import statistics as st

    def avg(c, k):
        v = [r[k] for r in rows if r["cond"] == c and r.get(k) is not None]
        return st.fmean(v) if v else float("nan")

    print("\n" + "=" * 100)
    print("조건별 평균")
    print("=" * 100)
    keys = [("delta", "분량 증감%"), ("term_keep", "용어보존"),
            ("num_keep", "수치보존"), ("n_halluc", "수치환각"),
            ("ground", "근거율"), ("cover", "커버리지"),
            ("copy_sim", "복사유사도"), ("blank_mark", "[미기재] 수"),
            ("out_term", "전문용어비율"), ("out_hanja", "한자어비율"),
            ("out_sent_len", "평균어절")]
    print(f"  {'지표':<14}" + "".join(f"{c:>11}" for c in conds)
          + f"{'P3-P0':>10}")
    for k, lab in keys:
        vals = [avg(c, k) for c in conds]
        d = vals[-1] - vals[0]
        print(f"  {lab:<14}" + "".join(f"{v:>11.2f}" for v in vals)
              + f"{d:>+10.2f}")

    ob = st.fmean(r["orig_blank"] for r in rows) if rows else 0
    print(f"\n  참고: 원문 빈칸 평균 {ob:.1f}개")

    print("\n" + "=" * 100)
    print("판정 — 사전에 정한 기준")
    print("=" * 100)
    base_g = avg(conds[0], "ground")
    for c in conds:
        fails = []
        cs = avg(c, "copy_sim")
        if cs >= 0.6:
            fails.append(f"복사유사도 {cs:.2f} ≥ 0.6")
        bm, obm = avg(c, "blank_mark"), avg(c, "orig_blank")
        if bm > obm * 2 and bm > 3:
            fails.append(f"[미기재] {bm:.1f}개 > 원문 빈칸 {obm:.1f}개의 2배")
        g = avg(c, "ground")
        if c != conds[0] and g < base_g:
            fails.append(f"근거율 {g:.1f} < P0 {base_g:.1f}")
        mark = "채택 가능" if not fails else "기각: " + " / ".join(fails)
        print(f"  {c}  {mark}")

    ok = [c for c in conds
          if avg(c, "copy_sim") < 0.6
          and not (avg(c, "blank_mark") > avg(c, "orig_blank") * 2
                   and avg(c, "blank_mark") > 3)
          and (c == conds[0] or avg(c, "ground") >= base_g)]
    if len(ok) > 1:
        best = max(ok[1:], key=lambda c: avg(c, "term_keep"))
        print(f"\n  용어보존 기준 최선: {best} "
              f"({avg(best,'term_keep'):.1f}, P0 대비 "
              f"{avg(best,'term_keep')-avg(conds[0],'term_keep'):+.1f})")
    else:
        print("\n  기준을 통과한 개선 조건이 없습니다. P0 유지.")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--conds", nargs="+", default=list(PROMPTS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="prompt_ladder")
    a = ap.parse_args()

    docs = a.docs or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.docdir, "*.txt"))
        if not os.path.basename(p).startswith("syn"))
    if not docs:
        sys.exit(f"[FATAL] {a.docdir}/*.txt 없음")
    print(f"[PLAN] 문서 {len(docs)}건 × 조건 {len(a.conds)}종 = "
          f"{len(docs)*len(a.conds)} 실행")
    for c in a.conds:
        print(f"  {c}: {PROMPTS[c][:70]}...")

    os.makedirs(OUTDIR, exist_ok=True)
    if not a.eval_only:
        stage_generate(docs, a.conds, a)
    rows = stage_eval(docs, a.conds, a)
    report(rows, a.conds)
    with open(f"{a.out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {a.out}_metrics.json / 출력은 {OUTDIR}/")


if __name__ == "__main__":
    main()
