#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protected_span.py — Protected Span v1 (숫자+단위, 빈칸)
========================================================
목적:
    프롬프트로 "숫자를 바꾸지 마세요" 라고 부탁하는 대신, 보호 대상을
    placeholder 로 치환해 모델이 **물리적으로 바꿀 수 없게** 만든다.

v1 보호 대상 (의도적으로 좁게 시작한다):
    NUM    숫자+단위        3주, 1시간, 5%, 2박 3일, 1~2주, 30분
    BLANK  빈칸 / 미기재     (  ), ____, "약 __ 정도", [미기재]

    부정 표현·위험도·법적 조항은 v1 에서 보호하지 않는다.
    보호 범위를 넓히면 문장이 placeholder 로 가득 차 평이화 자체가
    불가능해지므로, 효과가 확인된 뒤 단계적으로 넓힌다.

파이프라인:
    원문 ──mask──> 마스킹문 ──생성──> 출력(placeholder 포함)
                                    │
                              bijection audit
                                    │ pass
                                    ├──restore──> 최종문
                                    │ fail
                                    └──fail-closed──> 원문 그대로 반환 + 사유 기록

bijection audit (fail-closed):
    1. 모든 placeholder 가 출력에 **정확히 1회** 존재해야 한다
    2. 마스킹에 쓰지 않은 placeholder 가 출력에 있으면 실패 (모델이 만들어냄)
    3. placeholder 변형(⟦N1 ⟧, ⟦n1⟧, ⟦N01⟧ 등)이 있으면 실패
    4. 하나라도 어기면 복원하지 않고 실패로 처리한다.
       "일부만 복원" 은 원문보다 위험하므로 허용하지 않는다.

placeholder 형식:
    ⟦N1⟧ ⟦B1⟧
    U+27E6/U+27E7 (MATHEMATICAL WHITE SQUARE BRACKET). 한국어 텍스트에
    자연 출현하지 않고, 토크나이저가 통째로 다루는 경우가 많다.

자체 점검:
    python protected_span.py            # 단위 테스트 + 실제 문서 마스킹 미리보기
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

L, R = "⟦", "⟧"
PH_RE = re.compile(r"⟦([A-Z]+)(\d+)⟧")
# 변형 탐지 — 정상 형태가 아닌 유사 패턴
PH_LOOSE = re.compile(r"[⟦\[［]\s*([A-Za-z]+)\s*0*(\d+)\s*[⟧\]］]")

# ── 보호 패턴 (순서 중요: 긴 것부터) ─────────────────────────────────
UNIT = r"(?:주|일|개월|달|시간|분|초|년|%|퍼센트|회|번|명|세|박|cc|ml|mg|g|kg|cm|mm)"
NUM_PATTERNS: List[Tuple[str, str]] = [
    # 2박 3일
    ("NUM", rf"\d{{1,4}}\s*박\s*\d{{1,4}}\s*일"),
    # 1~2주, 5-7일, 3∼4개월
    ("NUM", rf"\d{{1,4}}\s*[~∼\-–]\s*\d{{1,4}}\s*{UNIT}"),
    # 약 3주, 3주
    ("NUM", rf"\d{{1,4}}\s*{UNIT}"),
]
# phase1_eval.py 의 ORIG_BLANK 와 같은 대상을 잡도록 맞췄다.
# 특히 마지막 패턴이 중요하다 — 원문의 "약 ___ 정도 소요됩니다" 는
# PDF 추출 시 숫자가 사라져 "약 정도 소요됩니다" 로 붙어 나온다.
BLANK_PATTERNS: List[Tuple[str, str]] = [
    ("BLANK", r"\[\s*미기재\s*\]"),
    ("BLANK", r"[（(\[〔]\s{2,}[)）\]〕]"),      # 공백만 있는 괄호
    ("BLANK", r"[（(\[〔]\s*[)）\]〕]"),          # 완전히 빈 괄호
    ("BLANK", r"[_＿]{2,}"),
    ("BLANK", r"\.{4,}"),
    # 수치가 빠진 "약 ~ 정도/가량/이내" 자리
    ("BLANK", r"약\s+(?=정도|가량|이내|이상|이하|동안|쯤)"),
]
PATTERNS = NUM_PATTERNS + BLANK_PATTERNS


@dataclass
class Masked:
    text: str                                  # 마스킹된 텍스트
    mapping: Dict[str, str] = field(default_factory=dict)   # ⟦N1⟧ -> "3주"
    spans: List[dict] = field(default_factory=list)          # 감사용 원본 정보

    @property
    def n(self) -> int:
        return len(self.mapping)

    def counts(self) -> Dict[str, int]:
        """종류별 개수. 키는 'NUM' / 'BLANK' (placeholder 접두는 N / B)."""
        名 = {"N": "NUM", "B": "BLANK"}
        c: Dict[str, int] = {"NUM": 0, "BLANK": 0}
        for k in self.mapping:
            m = PH_RE.match(k)
            if m:
                key = 名.get(m.group(1), m.group(1))
                c[key] = c.get(key, 0) + 1
        return c


def mask(text: str) -> Masked:
    """보호 대상을 placeholder 로 치환한다. 겹치는 구간은 긴 쪽을 택한다."""
    hits: List[Tuple[int, int, str, str]] = []
    for kind, pat in PATTERNS:
        for m in re.finditer(pat, text):
            hits.append((m.start(), m.end(), kind, m.group(0)))
    # 겹침 제거 — 시작이 빠른 것, 같으면 긴 것 우선
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for s, e, kind, raw in hits:
        if s >= last_end:
            chosen.append((s, e, kind, raw))
            last_end = e

    counters: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    spans: List[dict] = []
    out, prev = [], 0
    for s, e, kind, raw in chosen:
        counters[kind] = counters.get(kind, 0) + 1
        ph = f"{L}{'N' if kind == 'NUM' else 'B'}{counters[kind]}{R}"
        mapping[ph] = raw
        spans.append({"placeholder": ph, "kind": kind,
                      "original": raw, "start": s, "end": e})
        out.append(text[prev:s])
        out.append(ph)
        prev = e
    out.append(text[prev:])
    return Masked("".join(out), mapping, spans)


# ===========================================================================
# bijection audit
# ===========================================================================

@dataclass
class AuditResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    detail: Dict[str, object] = field(default_factory=dict)


def audit(generated: str, m: Masked) -> AuditResult:
    """모든 placeholder 가 정확히 1회 존재하는지 확인한다. fail-closed."""
    reasons: List[str] = []
    expected = set(m.mapping)

    found = PH_RE.findall(generated)
    found_ph = [f"{L}{k}{i}{R}" for k, i in found]
    counts: Dict[str, int] = {}
    for p in found_ph:
        counts[p] = counts.get(p, 0) + 1

    missing = sorted(expected - set(counts))
    unknown = sorted(set(counts) - expected)
    dup = sorted(p for p, c in counts.items() if c > 1)

    # 변형 탐지 — 느슨한 패턴으로 잡히지만 정상 형태가 아닌 것
    variants = []
    for mm in PH_LOOSE.finditer(generated):
        whole = mm.group(0)
        if not PH_RE.fullmatch(whole):
            variants.append(whole)

    if missing:
        reasons.append(f"누락 {len(missing)}개: {missing[:6]}")
    if dup:
        reasons.append(f"중복 {len(dup)}개: {[(p, counts[p]) for p in dup[:6]]}")
    if unknown:
        reasons.append(f"미지 placeholder {len(unknown)}개: {unknown[:6]}")
    if variants:
        reasons.append(f"변형 {len(variants)}개: {variants[:6]}")

    return AuditResult(
        ok=not reasons, reasons=reasons,
        detail={"n_expected": len(expected), "n_found": len(counts),
                "missing": missing, "duplicated": dup,
                "unknown": unknown, "variants": variants[:20]})


def restore(generated: str, m: Masked) -> str:
    """placeholder 를 원본 문자열로 되돌린다. audit 통과 후에만 호출한다."""
    out = generated
    # 번호가 큰 것부터 치환해 ⟦N1⟧ 이 ⟦N11⟧ 의 일부와 섞이는 것을 막는다
    for ph in sorted(m.mapping, key=lambda p: -len(p)):
        out = out.replace(ph, m.mapping[ph])
    return out


def protect_and_restore(generated: str, m: Masked) -> Tuple[str, AuditResult]:
    """(최종문, 감사결과). 실패 시 복원하지 않고 빈 문자열을 돌려준다."""
    r = audit(generated, m)
    return (restore(generated, m) if r.ok else ""), r


# ===========================================================================
# 자체 점검
# ===========================================================================

def _tests() -> int:
    bad = 0

    def chk(name, cond, extra=""):
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  [{'OK  ' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

    print("=" * 84)
    print("1. 마스킹")
    print("=" * 84)
    src = ("수술 시간은 편측의 경우 1 시간, 양측의 경우 2 시간 가량 소요됩니다. "
           "입원 기간은 2박 3일이며 1~2주간 안정이 필요합니다. "
           "감염은 약 5% 에서 보고됩니다. 특이체질(  ) 여부 확인. "
           "수술 예정일: ____년 __월. 약 [미기재] 정도 소요됩니다. "
           "회복에는 약 정도 걸립니다.")
    m = mask(src)
    print(f"  원문   : {src}")
    print(f"  마스킹 : {m.text}")
    print(f"  개수   : {m.counts()}  총 {m.n}")
    for ph, raw in list(m.mapping.items())[:10]:
        print(f"     {ph} = {raw!r}")

    chk("숫자+단위 포착", any(v.strip().startswith("1") and "시간" in v
                          for v in m.mapping.values()))
    chk("2박 3일 한 덩어리", any("박" in v and "일" in v for v in m.mapping.values()))
    chk("범위 표현 1~2주", any("~" in v for v in m.mapping.values()))
    chk("빈 괄호 포착", any(re.fullmatch(r"[（(\[〔]\s*[)）\]〕]", v)
                        for v in m.mapping.values()))
    chk("밑줄 포착", any("_" in v for v in m.mapping.values()))
    chk("[미기재] 포착", any("미기재" in v for v in m.mapping.values()))
    chk("수치 빠진 '약 정도' 포착",
        any(v.strip() == "약" for v in m.mapping.values()))
    chk("원문에 숫자 잔존 없음",
        not re.search(rf"\d\s*{UNIT}", m.text), m.text[:60])

    print("\n" + "=" * 84)
    print("2. bijection audit — 정상")
    print("=" * 84)
    good = m.text.replace("소요됩니다", "걸립니다")
    fin, r = protect_and_restore(good, m)
    chk("정상 출력 통과", r.ok, str(r.reasons))
    chk("복원 후 placeholder 없음", not PH_RE.search(fin))
    chk("복원 값이 원본과 일치", all(v in fin for v in m.mapping.values()))

    print("\n" + "=" * 84)
    print("3. bijection audit — 실패 (fail-closed 확인)")
    print("=" * 84)
    first = list(m.mapping)[0]
    cases = [
        ("placeholder 누락", m.text.replace(first, "", 1)),
        ("placeholder 중복", m.text + f" 그리고 {first} 다시"),
        ("미지 placeholder", m.text + " ⟦N99⟧"),
        ("변형 (소문자)", m.text.replace(first, first.lower(), 1)),
        ("변형 (공백 삽입)", m.text.replace(first, first[0] + " " + first[1:], 1)),
        ("변형 (0 패딩)", m.text.replace(first, first.replace("1", "01"), 1)),
    ]
    for name, g in cases:
        fin2, r2 = protect_and_restore(g, m)
        chk(f"{name} 차단", (not r2.ok) and fin2 == "",
            "; ".join(r2.reasons)[:70])

    print("\n" + "=" * 84)
    print("4. 번호 자릿수 충돌")
    print("=" * 84)
    long_src = " ".join(f"{i}주" for i in range(1, 15))
    m2 = mask(long_src)
    fin3, r3 = protect_and_restore(m2.text, m2)
    chk(f"placeholder {m2.n}개 왕복", r3.ok and fin3 == long_src,
        f"복원={fin3[:40]}")

    print("\n" + "=" * 84)
    print(f"실패 {bad}")
    print("=" * 84)
    return bad


def _preview(docdir: str, docs: Optional[List[str]], n: int):
    from prompt_ladder import extract_sections
    paths = sorted(glob.glob(os.path.join(docdir, "*.txt")))
    names = docs or [os.path.splitext(os.path.basename(p))[0] for p in paths
                     if not os.path.basename(p).startswith("syn")]
    print("\n" + "=" * 96)
    print("실제 문서 마스킹 미리보기")
    print("=" * 96)
    print(f"  {'문서':<8}{'글자':>7}{'NUM':>6}{'BLANK':>7}{'합계':>6}"
          f"{'잔존숫자':>9}  마스킹 후 앞부분")
    tot = {"NUM": 0, "BLANK": 0}
    for name in names[:n]:
        p = os.path.join(docdir, f"{name}.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            secs = extract_sections(f.read())
        if not secs:
            continue
        K = " ".join(b for _, b in secs)
        m = mask(K)
        c = m.counts()
        tot["NUM"] += c.get("NUM", 0)
        tot["BLANK"] += c.get("BLANK", 0)
        left = len(re.findall(rf"\d\s*{UNIT}", m.text))
        print(f"  {name:<8}{len(K):>7}{c.get('NUM',0):>6}{c.get('BLANK',0):>7}"
              f"{m.n:>6}{left:>9}  {m.text[:48]}")
    print(f"  {'합계':<8}{'':>7}{tot['NUM']:>6}{tot['BLANK']:>7}"
          f"{tot['NUM']+tot['BLANK']:>6}")
    print("""
  읽는 법
    잔존숫자가 0 이어야 정상입니다. 0 이 아니면 마스킹 패턴이 놓친 형태가
    있다는 뜻이므로, 해당 문서를 열어 확인한 뒤 패턴을 보강해야 합니다.
    BLANK 가 0 인 문서는 원문에 빈칸이 없는 것입니다(문서마다 다릅니다).""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=os.path.expanduser("~/이윤우/docs"))
    ap.add_argument("--docs", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--tests-only", action="store_true")
    a = ap.parse_args()

    bad = _tests()
    if not a.tests_only:
        try:
            _preview(a.docdir, a.docs, a.n)
        except Exception as e:
            print(f"\n  (문서 미리보기 생략: {type(e).__name__}: {e})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
