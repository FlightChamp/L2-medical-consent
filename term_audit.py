#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
term_audit.py — 용어 실태 감사 (읽기 전용)
============================================
아무 파일도 수정하지 않는다. 읽고, 집계하고, 검토용 CSV 를 새로 쓴다.

[지표 명칭 고정]
    prompt_ladder.extract_terms() 는 의료용어 추출기가 아니다.
    "임의의 3~10자 한글 단어 중 긴 것 상위 20개" 의 문자열 잔존율이다.
    앞으로 이 값은 **Top-20 surface lexical retention** 으로만 부른다.
    terminology integrity 와 같은 것으로 쓰지 않는다.

[candidate pool]
    simplicity_metrics.MED_TERMS
    ∪ 의료 접미사 규칙 (절제술/치환술/곤란/마비/색전증/유합/유착 등)
    ∪ 해부·기관명 목록
    ∪ nli_sensitivity2.Perturber.TERM_GROUPS  (TERM_SWAP 실험에 쓰인 용어)

    이 pool 은 **자동 후보**일 뿐이며 gold extractor 가 아니다.
    term_candidates.csv 를 사람이 검토해 valid 열을 채운 뒤 확정한다.

[term occurrence 인식]
    조사 결합형을 정상 출현으로 인식한다.
        갑상선이 / 갑상선을 / 갑상선의 / 갑상선으로 / 담낭절제술은
    방법:
      1) 후보를 길이 내림차순으로 정렬 (longest-match 우선)
      2) 앞은 한글이 아니어야 한다 (담낭절제술 안의 '담낭' 방지)
      3) 뒤는 (a) 한글이 아니거나 (b) 알려진 조사·어미로 시작해야 한다
      4) 이미 선택된 구간과 겹치면 버린다 (overlap suppression)

[HARI term status — 5분류]
    exact_retained          용어가 출력에 그대로 있다
    retained_explained      용어가 있고 바로 뒤에 (설명) 이 붙어 있다
    omitted                 용어가 없고, 원문에 없던 다른 의료용어도 없다
    wrongly_substituted     용어가 없고, 원문에 없던 다른 의료용어가 출력에 있다
    paraphrased_no_term     용어가 없고, 그 용어의 문맥이 남아 있는 것으로 보인다

    뒤의 세 분류는 **자동 판정이 불가능하다.** 규칙으로 후보만 나누고
    term_status_review.csv 에 담아 사람이 확정하게 한다.
    단순 문자열 미존재를 모두 omission 으로 세지 않는다.

사용법:
    cd ~/이윤우 && source .venv/bin/activate
    python term_audit.py
    python term_audit.py --min-docs 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from prompt_ladder import extract_sections, extract_terms, STOP
except ImportError as e:
    sys.exit("[FATAL] prompt_ladder.py 필요: " + str(e))

MED_TERMS_SM: List[str] = []
try:
    from simplicity_metrics import MED_TERMS as MED_TERMS_SM
except Exception:
    pass

TERM_GROUPS: Dict[str, List[str]] = {}
try:
    from nli_sensitivity2 import Perturber
    TERM_GROUPS = dict(getattr(Perturber, "TERM_GROUPS", {}) or {})
except Exception:
    pass

HOME = os.path.expanduser("~/이윤우")
DOCDIR = os.path.join(HOME, "docs")
HARI_P0 = os.path.join(HOME, "outputs_prompt", "{doc}__P0.json")

DICT_CANDIDATES = ["termdict_verified.json", "termdict_raw.json",
                   "kofih_termdict.json"]

# ── 의료 접미사 규칙 ────────────────────────────────────────────────
MED_SUFFIX = re.compile(
    r"[가-힣]{1,9}(?:절제술|재건술|치환술|성형술|삽입술|고정술|삽입|생검|내시경|"
    r"곤란|마비|협착|파열|폐색|색전증|혈증|염증|괴사|유착|누출|유출|출혈|감염|"
    r"합병증|후유증|부작용|장애|손상|골절|탈구|종양|낭종|궤양|농양|결석|"
    r"경색증|경색|허혈|부종|혈전|유합|천공|열상|봉합|배액|질환|위축|"
    r"마취|수혈|투여|주입|절개|이식|검사|촬영|판독)$")

ANATOMY = [
    "갑상선", "담낭", "전립선", "방광", "신장", "유방", "척추", "관절", "인대",
    "혈관", "신경", "고관절", "슬관절", "견관절", "인공관절", "십자인대",
    "담도", "담관", "복강", "흉곽", "늑막", "흉관", "기도", "성대", "고환",
    "서혜부", "견봉", "반월상연골", "회전근개", "치핵", "치루", "항문",
    "무릎", "어깨", "대장", "소장", "담석", "결석", "농양", "천공", "누공",
]

# 조사·어미 — 용어 뒤에 붙어도 같은 용어 출현으로 본다
JOSA = (
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "에게", "에는",
    "으로", "로", "와", "과", "도", "만", "부터", "까지", "라", "이라",
    "이나", "나", "이며", "며", "이고", "고", "인", "이란", "란",
    "처럼", "보다", "조차", "마저", "밖에", "이라고", "라고",
    "술", "시", "후", "전", "중", "및", "또는",
)
JOSA_RE = re.compile("^(?:" + "|".join(sorted(JOSA, key=len, reverse=True)) + ")")

HANGUL = re.compile(r"[가-힣]")
# "용어(설명)" — 괄호 안 한글 4자 이상
EXPLAIN_AFTER = r"\s*[（(]\s*([^)）]{4,60})\s*[)）]"


def load_K(doc: str, docdir: str) -> Optional[str]:
    p = os.path.join(docdir, doc + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        secs = extract_sections(f.read())
    return " ".join(b for _, b in secs) if secs else None


def load_hari(doc: str) -> Optional[str]:
    p = HARI_P0.format(doc=doc)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("out")


# ===========================================================================
# candidate pool
# ===========================================================================

def build_pool() -> Dict[str, Set[str]]:
    """용어 → 출처 집합. 출처를 기록해 검토 시 판단 근거로 쓴다."""
    pool: Dict[str, Set[str]] = defaultdict(set)
    for t in MED_TERMS_SM:
        pool[t].add("MED_TERMS")
    for a in ANATOMY:
        pool[a].add("anatomy")
    for cat, terms in TERM_GROUPS.items():
        for t in terms:
            pool[t].add("TERM_SWAP:" + cat)
    return pool


def suffix_hits(text: str) -> Set[str]:
    out: Set[str] = set()
    for w in re.findall(r"[가-힣]{3,12}", text):
        if w in STOP:
            continue
        if MED_SUFFIX.fullmatch(w):
            out.add(w)
    return out


# ===========================================================================
# occurrence — 조사 허용, longest-match, overlap suppression
# ===========================================================================

def find_occurrences(text: str, terms: List[str]) -> List[Tuple[int, int, str]]:
    """(start, end, term). 긴 것 우선, 겹치면 버린다."""
    hits: List[Tuple[int, int, str]] = []
    for t in sorted(set(terms), key=len, reverse=True):
        start = 0
        while True:
            i = text.find(t, start)
            if i < 0:
                break
            start = i + 1
            # 앞이 한글이면 더 긴 단어의 일부다
            if i > 0 and HANGUL.match(text[i - 1]):
                continue
            j = i + len(t)
            tail = text[j:j + 6]
            # 뒤가 한글이면 조사·어미일 때만 허용
            if tail and HANGUL.match(tail[0]) and not JOSA_RE.match(tail):
                continue
            hits.append((i, j, t))
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: List[Tuple[int, int, str]] = []
    last = -1
    for s, e, t in hits:
        if s >= last:
            kept.append((s, e, t))
            last = e
    return kept


def has_explanation(text: str, term: str) -> bool:
    return re.search(re.escape(term) + EXPLAIN_AFTER, text) is not None


def explanation_of(text: str, term: str):
    """용어 뒤 괄호 내용. 없으면 None."""
    m = re.search(re.escape(term) + EXPLAIN_AFTER, text)
    return m.group(1).strip() if m else None


def find_variant(term: str, pool_terms) -> str:
    """term 의 변이형을 찾는다. 포함 관계면 변이형으로 본다.
       원문 심근경색증 → 출력 심근경색   (접미 탈락)
       원문 절개      → 출력 피부절개   (접두 부착)

       기준:
         · 1자 용어는 제외한다 ('위' 가 '위험' 과 우연히 겹친다)
         · 상대 용어는 3자 이상이어야 한다
         · pool_terms 는 이미 의료용어 후보만 담고 있으므로
           일반어와의 우연 일치는 발생하지 않는다"""
    if len(term) < 2:
        return ""
    best = ""
    for t in pool_terms:
        if t == term or len(t) < 3:
            continue
        if term in t or t in term:
            if len(t) > len(best):
                best = t
    return best


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdir", default=DOCDIR)
    ap.add_argument("--min-docs", type=int, default=1)
    ap.add_argument("--out", default="term_audit.json")
    ap.add_argument("--csv-cand", default="term_candidates.csv")
    ap.add_argument("--csv-status", default="term_status_review.csv")
    a = ap.parse_args()

    docs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(a.docdir, "*.txt"))
                  if not os.path.basename(p).startswith("syn"))

    print("=" * 98)
    print("용어 실태 감사 (읽기 전용)")
    print("=" * 98)

    # ── 0. smoke test — 일반화하지 않는다 ───────────────────────────
    print("\n0. 접미사 규칙 smoke test")
    print("   ※ 이것은 규칙이 돌아가는지 보는 점검이며 extractor 성능이 아닙니다.")
    print("     13문서 전체 후보를 사람이 검토하기 전에는 gold extractor 로")
    print("     취급하지 않습니다.")
    smoke_ok = ["담낭절제술", "인공관절치환술", "흉관삽입", "연하곤란",
                "불유합", "장유착", "문합부유출", "저칼슘혈증"]
    smoke_ng = ["수술", "환자", "대체방법", "있습니다"]
    hit = sum(1 for t in smoke_ok if MED_SUFFIX.fullmatch(t) and len(t) >= 3)
    fp = sum(1 for t in smoke_ng if MED_SUFFIX.fullmatch(t) and len(t) >= 3)
    print(f"   포착 {hit}/{len(smoke_ok)}, 오탐 {fp}/{len(smoke_ng)} (smoke test 한정)")

    # ── 1. 기존 사전 ───────────────────────────────────────────────
    print("\n1. 기존 용어 사전 파일")
    dicts = {}
    for name in DICT_CANDIDATES:
        p = os.path.join(HOME, name)
        if not os.path.exists(p):
            print(f"   {name:<26}없음")
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            n = len(d) if isinstance(d, (list, dict)) else 0
            print(f"   {name:<26}{n:>6}항목")
            dicts[name] = n
            if isinstance(d, dict) and d:
                k0 = list(d)[0]
                v0 = json.dumps(d[k0], ensure_ascii=False)[:150]
                print(f"      키 예시 {k0!r} → {v0}")
                fields = set()
                for v in list(d.values())[:50]:
                    if isinstance(v, dict):
                        fields |= set(v)
                ko_expl = {"설명", "easy", "easy_explanation", "쉬운설명",
                           "explanation"} & fields
                print(f"      필드: {sorted(fields)[:8]}")
                print(f"      한국어 쉬운 설명 필드: "
                      f"{sorted(ko_expl) if ko_expl else '없음'}")
        except Exception as e:
            print(f"   {name:<26}읽기 실패 {type(e).__name__}")

    pool = build_pool()
    print(f"\n   candidate pool 구성")
    print(f"      MED_TERMS            {len(MED_TERMS_SM):>4}")
    print(f"      anatomy 목록          {len(ANATOMY):>4}")
    print(f"      TERM_SWAP TERM_GROUPS {sum(len(v) for v in TERM_GROUPS.values()):>4}"
          f"  ({len(TERM_GROUPS)}범주)")
    print(f"      합집합 (중복 제거)      {len(pool):>4}")
    print(f"      + 접미사 규칙은 문서에서 동적으로 추출")

    # ── 2. 문서별 후보 ─────────────────────────────────────────────
    Ks: Dict[str, str] = {}
    by_doc: Dict[str, Set[str]] = {}
    freq = Counter()
    docs_of: Dict[str, Set[str]] = defaultdict(set)
    src_of: Dict[str, Set[str]] = defaultdict(set)

    for d in docs:
        K = load_K(d, a.docdir)
        if not K:
            continue
        Ks[d] = K
        cand = set(suffix_hits(K))
        for t in cand:
            src_of[t].add("suffix")
        for t in pool:
            if find_occurrences(K, [t]):
                cand.add(t)
                src_of[t] |= pool[t]
        occ = find_occurrences(K, list(cand))
        present = {t for _, _, t in occ}
        by_doc[d] = present
        for _, _, t in occ:
            freq[t] += 1
        for t in present:
            docs_of[t].add(d)

    allterms = sorted(docs_of, key=lambda t: (-len(docs_of[t]), -freq[t], -len(t)))
    multi = [t for t in allterms if len(docs_of[t]) >= a.min_docs]

    print("\n2. 13문서 candidate 집계")
    print(f"   전체 고유 candidate      {len(allterms)}")
    print(f"   {a.min_docs}개 이상 문서 등장      {len(multi)}")
    print("   문서별: " + ", ".join(f"{d}={len(by_doc.get(d,[]))}" for d in docs))
    print(f"\n   상위 25 (등장 문서 / 출현 횟수 / 출처)")
    print(f"   {'용어':<16}{'문서':>5}{'출현':>6}   출처")
    for t in allterms[:25]:
        print(f"   {t:<16}{len(docs_of[t]):>5}{freq[t]:>6}   "
              f"{','.join(sorted(src_of[t])[:3])}")

    # ── 3. HARI term status (5분류) ────────────────────────────────
    print("\n3. HARI P0 에서의 용어 status")
    print("   exact_retained / retained_explained 는 자동 판정")
    print("   omitted / wrongly_substituted / paraphrased_no_term 은 후보 분류이며")
    print("   term_status_review.csv 에서 사람이 확정해야 합니다.")

    status_rows = []
    tally = Counter()
    per_doc = []
    for d in docs:
        K, H = Ks.get(d), load_hari(d)
        if not K or not H:
            continue
        src_terms = sorted(by_doc[d])
        h_occ = find_occurrences(H, list(pool) + sorted(suffix_hits(H)))
        h_present = {t for _, _, t in h_occ}
        # 출력에만 있고 원문에 없는 의료용어 — 치환 의심의 근거
        new_in_out = sorted(h_present - by_doc[d])

        c = Counter()
        for t in src_terms:
            expl_out = explanation_of(H, t)
            expl_src = explanation_of(K, t)
            variant = ""
            if t in h_present:
                if expl_out:
                    # 원문에 같은 형태의 설명이 있으면 서식에서 온 것이다
                    st = ("explained_in_source" if expl_src
                          else "explained_by_model")
                else:
                    st = "exact_retained"
            else:
                variant = find_variant(t, h_present)
                st = "variant_form" if variant else "absent"
            c[st] += 1
            tally[st] += 1
            status_rows.append({
                "doc": d, "term": t, "auto_status": st,
                "n_docs": len(docs_of[t]), "freq_in_doc": K.count(t),
                "in_output": t in h_present,
                "explanation_in_output": expl_out or "",
                "explanation_in_source": expl_src or "",
                "variant_in_output": variant,
                "new_terms_in_output": ";".join(new_in_out[:5]),
                # absent 인 경우에만 사람이 아래 셋 중 하나로 확정한다
                "manual_status(omitted/paraphrased/substituted)": "",
                "note": "",
            })
        per_doc.append({"doc": d, "n_src": len(src_terms), **dict(c),
                        "new_in_out": new_in_out[:8]})
        print(f"   {d:<8}원문 {len(src_terms):>3}  "
              f"유지 {c['exact_retained']:>3}  "
              f"원문설명 {c['explained_in_source']:>2}  "
              f"모델설명 {c['explained_by_model']:>2}  "
              f"변이형 {c['variant_form']:>2}  "
              f"부재 {c['absent']:>3}"
              f"   출력에만: {', '.join(new_in_out[:3])}")

    n_tot = sum(tally.values())
    print(f"\n   {'status':<26}{'건수':>6}{'비율':>8}   설명")
    desc = {
        "exact_retained": "용어 유지, 설명 없음",
        "explained_in_source": "설명 있으나 원문 서식에서 온 것",
        "explained_by_model": "HARI 가 붙인 설명  ← 개선의 기준선",
        "variant_form": "용어 자체는 없고 포함관계 변이형이 있음",
        "absent": "용어·변이형 모두 없음  ← 사람이 확정 필요",
    }
    for k in ("exact_retained", "explained_in_source", "explained_by_model",
              "variant_form", "absent"):
        v = tally.get(k, 0)
        print(f"   {k:<26}{v:>6}{100*v/max(n_tot,1):>7.1f}%   {desc[k]}")
    print(f"   {'합계':<26}{n_tot:>6}")

    integ = (tally.get("exact_retained", 0) + tally.get("explained_in_source", 0)
             + tally.get("explained_by_model", 0) + tally.get("variant_form", 0))
    print(f"""
   terminology integrity = {100*integ/max(n_tot,1):.1f}%
     (exact + 설명 2종 + 변이형. 용어가 어떤 형태로든 남아 있는 비율)

   absent {tally.get('absent',0)}건은 omitted / paraphrased_no_term /
   wrongly_substituted 중 하나이며 자동 분리가 불가능합니다.
   term_status_review.csv 의 manual_status 열에서 확정하십시오.
   'new_terms_in_output' 열이 판단 근거입니다.

   1차 감사에서 wrongly_substituted 로 셌던 37건 중 상당수는
   심근경색증 → 심근경색 같은 변이형이었고, 이제 variant_form 으로 갑니다.""")

    # ── 4. Top-20 surface lexical retention (별개 지표) ────────────
    print("\n4. Top-20 surface lexical retention  (terminology integrity 아님)")
    sl = []
    for d in docs:
        K, H = Ks.get(d), load_hari(d)
        if not K or not H:
            continue
        ts = extract_terms(K)
        kept = sum(1 for t in ts if t in H)
        sl.append(100 * kept / max(len(ts), 1))
    if sl:
        print(f"   13문서 평균 {sum(sl)/len(sl):.2f}%")
        print("   → prompt_ladder.extract_terms() 의 임의 장단어 상위 20개 기준입니다.")
        print("     의료용어 보존율과 다른 값이며 혼용하지 않습니다.")

    # ── 5. 기존 설명 사례 ──────────────────────────────────────────
    print("\n5. HARI P0 출력에 이미 '용어(설명)' 형태로 있는 사례")
    ex = []
    for d in docs:
        H = load_hari(d)
        if not H:
            continue
        for t in sorted(pool, key=len, reverse=True):
            m = re.search(re.escape(t) + EXPLAIN_AFTER, H)
            if m:
                ex.append((d, t, m.group(1)))
        for t in sorted(suffix_hits(H), key=len, reverse=True):
            m = re.search(re.escape(t) + EXPLAIN_AFTER, H)
            if m and not any(x[0] == d and x[1] == t for x in ex):
                ex.append((d, t, m.group(1)))
    print(f"   총 {len(ex)}건")
    for d, t, e in ex[:15]:
        print(f"     [{d}] {t}({e[:44]})")

    # ── 6. Glossary 대상 산정 ──────────────────────────────────────
    attachable = sorted(
        {t for t in allterms
         if any(r["term"] == t and r["in_output"] for r in status_rows)},
        key=lambda t: (-len(docs_of[t]), -freq[t]))
    # 이미 설명이 붙은 것 중 '원문 서식' 은 개선 대상에서 빼지 않는다.
    # 원문 서식 설명(심장질환(심근경색 등))은 환자용 쉬운 설명이 아니므로
    # Glossary 로 보완할 여지가 있다. 모델이 붙인 설명만 제외한다.
    already = {r["term"] for r in status_rows
               if r["auto_status"] == "explained_by_model"}
    already_src = {r["term"] for r in status_rows
                   if r["auto_status"] == "explained_in_source"}
    need = [t for t in attachable if t not in already]
    need_multi = [t for t in need if len(docs_of[t]) >= 2]

    print("\n6. Glossary 대상 산정")
    print(f"   전체 candidate              {len(allterms)}")
    print(f"   HARI 출력에 남는 용어         {len(attachable)}   ← 적용 가능 범위")
    print(f"   HARI 가 설명을 붙인 용어      {len(already & set(attachable))}"
          f"   ← 제외 (이미 목표 형태)")
    print(f"   원문 서식 설명이 붙은 용어     {len(already_src & set(attachable))}"
          f"   ← 포함 (환자용 설명이 아님)")
    print(f"   설명을 새로 정의할 용어       {len(need)}")
    print(f"     그중 2개 이상 문서 등장     {len(need_multi)}   ← 출처 조사 1차 대상")
    print(f"\n   1차 대상 목록")
    for i, t in enumerate(need_multi[:30], 1):
        print(f"     {i:>2}. {t:<16}{len(docs_of[t])}문서  출현 {freq[t]}")

    # ── CSV ────────────────────────────────────────────────────────
    with open(a.csv_cand, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "n_docs", "n_occurrence", "sources", "docs",
                    "in_hari_output", "already_explained",
                    "valid(1/0)", "category", "note"])
        for t in allterms:
            w.writerow([t, len(docs_of[t]), freq[t],
                        ";".join(sorted(src_of[t])), ";".join(sorted(docs_of[t])),
                        1 if t in attachable else 0,
                        1 if t in already else 0, "", "", ""])

    with open(a.csv_status, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(status_rows[0]) if status_rows
                           else ["doc", "term"])
        w.writeheader()
        for r in status_rows:
            w.writerow(r)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("candidate pool 은 자동 후보이며 gold extractor 가 아니다. "
                     "term_candidates.csv 의 valid 열을 사람이 채운 뒤 확정한다."),
            "pool_sources": {"MED_TERMS": len(MED_TERMS_SM),
                             "anatomy": len(ANATOMY),
                             "TERM_GROUPS": sum(len(v) for v in TERM_GROUPS.values())},
            "existing_dicts": dicts,
            "n_candidates": len(allterms),
            "n_attachable": len(attachable),
            "n_need_definition": len(need),
            "n_need_multi_doc": len(need_multi),
            "status_tally": dict(tally),
            "top20_surface_lexical_retention": (sum(sl)/len(sl) if sl else None),
            "per_doc": per_doc,
            "terms": [{"term": t, "n_docs": len(docs_of[t]), "freq": freq[t],
                       "sources": sorted(src_of[t]), "docs": sorted(docs_of[t]),
                       "attachable": t in attachable,
                       "already_explained": t in already} for t in allterms],
            "existing_explanations": [{"doc": d, "term": t, "explanation": e}
                                      for d, t, e in ex],
            "priority_need": need_multi,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {a.out}")
    print(f"[SAVE] {a.csv_cand}      ← valid 열을 채워 후보를 확정하십시오")
    print(f"[SAVE] {a.csv_status}   ← manual_status 열에 5분류를 확정하십시오")
    print("""
결과 보는 법
   · 3절의 exact_retained / retained_explained 만 자동 판정입니다.
     omitted? / wrongly_substituted? 는 '?' 가 붙은 후보 분류입니다.
   · '출력에만 있는 용어' 열이 치환 의심의 근거입니다. 비어 있으면
     그 문서의 미존재 용어는 omission 또는 paraphrase 쪽입니다.
   · 4절의 Top-20 surface lexical retention 은 의료용어 지표가 아닙니다.
     보고서에서 terminology integrity 와 섞어 쓰지 마십시오.
   · 6절의 '적용 가능 범위' 가 이 실험의 상한입니다. 이 값이 작으면
     Glossary 로 얻을 수 있는 최대치도 작습니다.""")
    print("=" * 98)


if __name__ == "__main__":
    main()
