#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_schema.py — 파이프라인 공통 데이터 스키마
====================================================
목적:
    ①~⑥ 단계가 주고받을 데이터 형식을 하나로 고정한다.
    실행기와 UI 가 모두 이 형식에 의존하므로, 이것이 정해져야
    두 작업을 동시에 시작할 수 있다.

설계 결정 (2026-09-19 확정):
    1. 최종 산출은 단일 파일 results/{doc_id}.json
       중간 캐시는 .cache/{doc_id}/{stage}.json 에 따로 둔다.
       → UI 는 파일 하나만 읽으면 되고, 실패 지점부터 재개할 수 있다.
    2. source.text 를 포함한다.
       → 이 JSON 은 의료문서 파생물이므로 results/ 와 .cache/ 를
         .gitignore 에 넣는다. UI 는 로컬에서만 동작한다.
    3. cache invalidation 을 위해 각 단계에 input_hash 와 stage_version 을 둔다.
       스키마 자체에는 schema_version, 문서 전체에는 status 를 둔다.

캐시 무효화 규칙:
    각 단계는 자신의 입력을 해시한 input_hash 와 구현 버전 stage_version 을
    기록한다. 재실행 시 둘 중 하나라도 달라지면 캐시를 버리고 다시 계산한다.

        input_hash     그 단계에 실제로 들어간 텍스트·설정의 sha256 앞 12자
        stage_version  그 단계 구현이 바뀌면 올린다 (예: "s3-v1")

    상류 단계가 다시 계산되면 하류의 input_hash 가 자동으로 달라지므로
    연쇄 무효화가 별도 로직 없이 일어난다.

기존 출력과의 매핑 (실측 확인):
    ③ outputs_prompt/{doc}__P0.json
         doc, cond, model, prompt, src, out, out_raw, meta_removed
    ③-b outputs_glossary/E/{doc}.json
         doc, route, source_generation, render, A_output, E_output,
         n_insertions, insertions, n_skipped, skipped,
         incorrect_insertions, sentence_diagnostic, eligible_occurrences,
         glossary_version, glossary_entries_enabled
    ⑤ outputs_translate/{doc}__{route}__{lang}.json
         doc, route, lang, model, src, out
    ② blank_guard.py
         find_blanks(text), should_convert(text, min_hangul),
         mark_blanks(text), numeric_hallucination(src, out)
    ④ NLI.rate(hyps, chunks, k, tau)

사용:
    from pipeline_schema import (
        new_document, stage_meta, input_hash,
        validate, SCHEMA_VERSION, STAGE_VERSIONS,
    )

    doc = new_document("doc1", "docs/doc1.pdf")
    doc["source"].update(...)
    doc["source"]["meta"] = stage_meta("s1", input_hash("docs/doc1.pdf"))

자체 점검:
    python pipeline_schema.py
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

# 각 단계 구현 버전. 구현이 바뀌면 여기를 올린다 → 캐시가 자동 무효화된다.
STAGE_VERSIONS = {
    "s1_extract": "s1-v1",
    "s2_integrity": "s2-v1",
    "s3_simplify": "s3-v1",
    "s3b_glossary": "s3b-v1.05",   # glossary_v1.json 버전과 맞춘다
    "s4_validate_ko": "s4-v1",
    "s5_translate": "s5-v1",
    "s6_validate_mt": "s6-v1",
}

STAGES = list(STAGE_VERSIONS)
LANGS = ["en", "zh", "ja", "vi"]

# 문서 전체 상태
#   pending    아직 시작 안 함
#   running    처리 중
#   complete   모든 단계 성공
#   partial    일부 단계만 성공 (재개 가능)
#   failed     복구 불가한 오류
DOC_STATUS = ("pending", "running", "complete", "partial", "failed")

# 단계별 상태
#   ok         정상 완료
#   skipped    설계상 건너뜀 (예: 빈칸 없음)
#   cached     캐시 재사용
#   failed     오류
STAGE_STATUS = ("ok", "skipped", "cached", "failed")


# ===========================================================================
# 해시 · 메타
# ===========================================================================

def _sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def input_hash(*parts: Any) -> str:
    """단계 입력의 해시. 텍스트·설정·파일 경로를 모두 받는다.

    파일 경로가 주어지면 파일 내용을 읽어 해시한다. 그 외에는
    JSON 으로 직렬화해 해시한다. 상류가 바뀌면 하류 해시도 달라지므로
    연쇄 무효화가 자동으로 일어난다."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str) and os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read())
        elif isinstance(p, (dict, list)):
            h.update(json.dumps(p, ensure_ascii=False, sort_keys=True).encode())
        else:
            h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:12]


def stage_meta(stage: str, in_hash: str, status: str = "ok",
               error: Optional[str] = None,
               elapsed_sec: Optional[float] = None,
               extra: Optional[dict] = None) -> dict:
    """모든 단계가 공통으로 갖는 메타 블록."""
    if stage not in STAGE_VERSIONS:
        raise ValueError("알 수 없는 단계: " + stage)
    if status not in STAGE_STATUS:
        raise ValueError("알 수 없는 상태: " + status)
    m = {
        "stage": stage,
        "stage_version": STAGE_VERSIONS[stage],
        "input_hash": in_hash,
        "status": status,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        m["error"] = error
    if elapsed_sec is not None:
        m["elapsed_sec"] = round(elapsed_sec, 2)
    if extra:
        m.update(extra)
    return m


def is_cache_valid(cached_meta: Optional[dict], stage: str,
                   in_hash: str) -> bool:
    """캐시를 재사용해도 되는지. 버전과 입력 해시가 모두 같아야 한다."""
    if not cached_meta:
        return False
    return (cached_meta.get("stage_version") == STAGE_VERSIONS.get(stage)
            and cached_meta.get("input_hash") == in_hash
            and cached_meta.get("status") in ("ok", "skipped", "cached"))


# ===========================================================================
# 문서 골격
# ===========================================================================

def new_document(doc_id: str, pdf_path: str) -> Dict[str, Any]:
    """빈 문서 골격. 각 단계가 자기 블록을 채운다."""
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "pipeline": {
            "langs": list(LANGS),
            "stage_versions": dict(STAGE_VERSIONS),
            "completed_stages": [],
            "failed_stages": [],
        },

        # ── ① 추출 ────────────────────────────────────────────────
        "source": {
            "meta": None,
            "pdf_path": pdf_path,
            "extractor": None,          # "pymupdf-text"
            "text": None,               # extract_sections 결과를 이어붙인 K
            "sections": [],             # [{"title","body","hangul_len"}]
            "n_chars": None,
            "n_sections": None,
        },

        # ── ② 무결성 ──────────────────────────────────────────────
        "integrity": {
            "meta": None,
            "blanks": [],               # [{"section_idx","kind","span","marked"}]
            "n_blanks": 0,
            "marked_text": None,        # mark_blanks 적용 결과
            "skipped_sections": [],     # [{"section_idx","title","reason","hangul_len"}]
            "min_hangul": 15,           # should_convert 임계값
        },

        # ── ③ 평이화 ──────────────────────────────────────────────
        "simplified": {
            "meta": None,
            "model": None,              # "snuh/hari-q3-8b"
            "prompt_version": None,     # "P0"
            "gen_config": None,         # {"do_sample":false,"max_new_tokens":4096}
            "text": None,               # 정제 후 (Cleaner.clean)
            "text_raw": None,           # 정제 전
            "meta_removed": 0,          # Cleaner 가 제거한 글자 수
            "n_chars": None,
        },

        # ── ③-b Glossary ──────────────────────────────────────────
        "glossary": {
            "meta": None,
            "version": None,            # "v1.05"
            "render": "inline",         # inline | footnote | none
            "text": None,               # 삽입 후 최종 (= UI 가 보여줄 한국어)
            "n_insertions": 0,
            "insertions": [],           # [{"term","explanation","position",
                                        #   "source","source_url","source_type",
                                        #   "source_verified","explanation_verified"}]
            "n_skipped": 0,
            "skipped": [],              # [{"term","reason"}]  ※ 원문 발췌는 넣지 않는다
            "incorrect_insertions": [],
            "sentence_boundary_preserved": None,
        },

        # ── ④ 한국어 검증 ─────────────────────────────────────────
        "ko_validation": {
            "meta": None,
            "target": None,             # 무엇을 검증했는지: "glossary" | "simplified"
            "nli_model": None,
            "params": None,             # {"k":5,"tau":0.5}
            "groundedness": None,       # 변환문이 원문에 근거하는 비율
            "coverage": None,           # 원문이 변환문에 남은 비율
            "terminology_integrity": None,
            "numeric": {                # 규칙 기반 (Protected Span 에서 전환)
                "preservation": None,
                "n_hallucination": 0,
                "hallucinated": [],
                "missing": [],
            },
            "blank_preservation": None,
            "copy_similarity": None,
            "flags": [],                # [{"kind","sentence_idx","score","text"}]
            "gate": {                   # 실험 전 고정. 결과를 보고 바꾸지 않는다.
                "criteria": {"num_keep_min": 90.0, "n_halluc_max": 0,
                             "ground_min": 70.0},
                "passed": None,
                "failures": [],
            },
        },

        # ── Simplicity (④와 함께 계산, 안전성과 분리해 보고) ───────
        "simplicity": {
            "meta": None,
            "ko2025": {                 # 고승연(2025). 높을수록 쉬움
                "policy_A": None,       # 미등재 어휘 = 중급 이상
                "policy_B": None,       # 미등재 어휘 = 분모 제외
                "level_A": None,        # 쉬움 | 보통 | 어려움
                "level_B": None,
            },
            "mid_vocab_ratio": {"policy_A": None, "policy_B": None},
            "words_per_sent": None,
            "chars_per_sent": None,
            "long_sent_ratio": None,
            "n_sentences": None,
            "length_change_pct": None,
        },

        # ── ⑤ 번역 ────────────────────────────────────────────────
        "translations": {               # lang -> {...}
            lang: {
                "meta": None,
                "model": None,
                "text": None,
                "n_chars": None,
            } for lang in LANGS
        },

        # ── ⑥ 번역 검증 ───────────────────────────────────────────
        "mt_validation": {              # lang -> {...}
            lang: {
                "meta": None,
                "nli_model": None,
                "params": None,         # {"k":0} — 교차 언어는 전체 청크 비교
                "forward": None,        # 한국어 전제 → 번역문 (환각)
                "reverse": None,        # 번역문 전제 → 한국어 (누락)
                "bidirectional_min": None,
                "numeric_preservation": None,
                "terminology_accuracy": None,
                "flags": [],
            } for lang in LANGS
        },

        # ── UI 표시용 요약 ────────────────────────────────────────
        "display": {
            "korean_source": None,      # 원문 (source.text)
            "korean_simple": None,      # 최종 한국어 (glossary.text 또는 simplified.text)
            "warnings": [],             # [{"level","kind","message","lang"}]
            "glossary_tooltips": [],    # [{"term","explanation","source_url"}]
        },
    }


# ===========================================================================
# 검증
# ===========================================================================

REQUIRED_TOP = ("schema_version", "doc_id", "status", "pipeline", "source",
                "integrity", "simplified", "glossary", "ko_validation",
                "simplicity", "translations", "mt_validation", "display")


def validate(doc: Dict[str, Any], strict: bool = False) -> List[str]:
    """스키마 위반을 목록으로 돌려준다. 빈 리스트면 정상.
    strict=True 면 아직 채워지지 않은 필수 값도 오류로 본다."""
    errs: List[str] = []

    for k in REQUIRED_TOP:
        if k not in doc:
            errs.append("최상위 키 누락: " + k)
    if errs:
        return errs

    if doc["schema_version"] != SCHEMA_VERSION:
        errs.append("schema_version 불일치: " + str(doc["schema_version"])
                    + " (기대 " + SCHEMA_VERSION + ")")
    if doc["status"] not in DOC_STATUS:
        errs.append("알 수 없는 status: " + str(doc["status"]))

    for lang in LANGS:
        if lang not in doc["translations"]:
            errs.append("translations 에 " + lang + " 없음")
        if lang not in doc["mt_validation"]:
            errs.append("mt_validation 에 " + lang + " 없음")

    # 단계 메타 검사
    blocks = [("source", doc["source"]), ("integrity", doc["integrity"]),
              ("simplified", doc["simplified"]), ("glossary", doc["glossary"]),
              ("ko_validation", doc["ko_validation"]),
              ("simplicity", doc["simplicity"])]
    blocks += [("translations." + l, doc["translations"][l]) for l in LANGS]
    blocks += [("mt_validation." + l, doc["mt_validation"][l]) for l in LANGS]

    for name, blk in blocks:
        m = blk.get("meta")
        if m is None:
            if strict:
                errs.append(name + ".meta 가 비어 있음")
            continue
        for f in ("stage", "stage_version", "input_hash", "status"):
            if f not in m:
                errs.append(name + ".meta 에 " + f + " 없음")
        if m.get("status") not in STAGE_STATUS:
            errs.append(name + ".meta.status 가 잘못됨: " + str(m.get("status")))

    if strict:
        if not doc["source"].get("text"):
            errs.append("source.text 가 비어 있음")
        if not doc["display"].get("korean_simple"):
            errs.append("display.korean_simple 이 비어 있음")

    # 논리 일관성
    gl = doc["glossary"]
    if gl.get("n_insertions") != len(gl.get("insertions", [])):
        errs.append("glossary.n_insertions 와 insertions 길이가 다름")
    if gl.get("n_skipped") != len(gl.get("skipped", [])):
        errs.append("glossary.n_skipped 와 skipped 길이가 다름")
    if doc["integrity"].get("n_blanks") != len(doc["integrity"].get("blanks", [])):
        errs.append("integrity.n_blanks 와 blanks 길이가 다름")

    return errs


def mark_stage(doc: Dict[str, Any], stage: str, ok: bool = True) -> None:
    """단계 완료를 pipeline 블록에 기록하고 전체 status 를 갱신한다."""
    p = doc["pipeline"]
    tgt = p["completed_stages"] if ok else p["failed_stages"]
    if stage not in tgt:
        tgt.append(stage)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    if p["failed_stages"]:
        doc["status"] = "failed" if not p["completed_stages"] else "partial"
    elif all(s in p["completed_stages"] for s in STAGES):
        doc["status"] = "complete"
    else:
        doc["status"] = "running"


# ===========================================================================
# 경로 규칙
# ===========================================================================

def result_path(out_dir: str, doc_id: str) -> str:
    """최종 단일 파일."""
    return os.path.join(out_dir, doc_id + ".json")


def cache_path(out_dir: str, doc_id: str, stage: str) -> str:
    """단계별 캐시. 재개용이며 최종 산출이 아니다."""
    return os.path.join(out_dir, ".cache", doc_id, stage + ".json")


# ===========================================================================
# 자체 점검
# ===========================================================================

def _demo() -> Dict[str, Any]:
    d = new_document("doc1", "docs/doc1.pdf")
    K = "1. 환자의 현재 상태 신장질환 (부 호흡기질환(기침, 종증) 가래등)"

    h1 = input_hash("docs/doc1.pdf")
    d["source"].update({
        "meta": stage_meta("s1_extract", h1, elapsed_sec=0.4),
        "extractor": "pymupdf-text", "text": K,
        "sections": [{"title": "1. 환자의 현재 상태", "body": K, "hangul_len": 28}],
        "n_chars": len(K), "n_sections": 1,
    })
    mark_stage(d, "s1_extract")

    h2 = input_hash(K, {"min_hangul": 15})
    d["integrity"].update({
        "meta": stage_meta("s2_integrity", h2),
        "blanks": [{"section_idx": 0, "kind": "empty_paren", "span": [12, 14],
                    "marked": "[미기재]"}],
        "n_blanks": 1, "marked_text": K,
    })
    mark_stage(d, "s2_integrity")

    h3 = input_hash(K, "P0", {"do_sample": False, "max_new_tokens": 4096})
    d["simplified"].update({
        "meta": stage_meta("s3_simplify", h3, elapsed_sec=12.3),
        "model": "snuh/hari-q3-8b", "prompt_version": "P0",
        "gen_config": {"do_sample": False, "max_new_tokens": 4096},
        "text": "환자의 현재 상태를 확인합니다.", "meta_removed": 0, "n_chars": 16,
    })
    mark_stage(d, "s3_simplify")

    h3b = input_hash(d["simplified"]["text"], "v1.05", "inline")
    d["glossary"].update({
        "meta": stage_meta("s3b_glossary", h3b),
        "version": "v1.05", "render": "inline",
        "text": "환자의 현재 상태를 확인합니다.",
        "n_insertions": 0, "n_skipped": 0,
        "sentence_boundary_preserved": True,
    })
    mark_stage(d, "s3b_glossary")

    d["display"].update({
        "korean_source": K,
        "korean_simple": d["glossary"]["text"],
        "warnings": [{"level": "warn", "kind": "blank",
                      "message": "원문에 비어 있는 항목이 1곳 있습니다"}],
    })
    return d


if __name__ == "__main__":
    print("=" * 84)
    print("pipeline_schema 자체 점검")
    print("=" * 84)
    print("  schema_version  " + SCHEMA_VERSION)
    print("  단계            " + ", ".join(STAGES))
    print("  언어            " + ", ".join(LANGS))

    print("\n1. 빈 문서 생성")
    empty = new_document("doc0", "docs/doc0.pdf")
    e = validate(empty)
    print("   느슨한 검증  " + ("통과" if not e else "실패 " + str(e)))
    e2 = validate(empty, strict=True)
    print("   엄격한 검증  " + str(len(e2)) + "건 지적 (아직 안 채웠으니 정상)")
    for x in e2[:3]:
        print("      " + x)

    print("\n2. 단계 진행 시뮬레이션")
    d = _demo()
    print("   status            " + d["status"])
    print("   completed_stages  " + ", ".join(d["pipeline"]["completed_stages"]))
    e3 = validate(d)
    print("   검증              " + ("통과" if not e3 else "실패 " + str(e3)))

    print("\n3. 캐시 무효화")
    K = d["source"]["text"]
    h_now = input_hash(K, "P0", {"do_sample": False, "max_new_tokens": 4096})
    meta = d["simplified"]["meta"]
    print("   같은 입력          " + str(is_cache_valid(meta, "s3_simplify", h_now))
          + "  (True 여야 정상 — 캐시 재사용)")
    h_diff = input_hash(K, "P4", {"do_sample": False, "max_new_tokens": 4096})
    print("   프롬프트 변경      " + str(is_cache_valid(meta, "s3_simplify", h_diff))
          + "  (False 여야 정상 — 재계산)")
    bumped = dict(meta, stage_version="s3-v2")
    print("   구현 버전 변경     " + str(is_cache_valid(bumped, "s3_simplify", h_now))
          + "  (False 여야 정상 — 재계산)")

    print("\n4. 연쇄 무효화")
    K2 = K + " 추가된 문장입니다."
    h3_new = input_hash(K2, "P0", {"do_sample": False, "max_new_tokens": 4096})
    print("   상류(K) 변경 시 하류 해시 변화  "
          + str(h_now != h3_new) + "  (True 여야 정상)")

    print("\n5. 논리 일관성 검사")
    bad = _demo()
    bad["glossary"]["n_insertions"] = 3        # insertions 는 비어 있음
    errs = validate(bad)
    print("   불일치 주입 후  " + str(len(errs)) + "건 탐지")
    for x in errs:
        print("      " + x)

    print("\n6. 경로 규칙")
    print("   최종  " + result_path("results", "doc1"))
    print("   캐시  " + cache_path("results", "doc1", "s3_simplify"))

    print("\n7. 스키마 골격")
    sk = {k: (list(v) if isinstance(v, dict) else type(v).__name__)
          for k, v in empty.items()}
    print(json.dumps(sk, ensure_ascii=False, indent=2)[:1400])
    print("=" * 84)
