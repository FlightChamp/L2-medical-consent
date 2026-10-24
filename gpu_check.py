#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_check.py — 실험 서버 GPU 환경 확인 (조회 전용)
====================================================
교수님께 보고할 GPU 정보를 정확히 확인한다. 아무것도 수정하지 않는다.

확인 항목
    1. 서버에 물리적으로 장착된 GPU 수와 모델명, 메모리
    2. 현재 사용 중인 메모리와 점유 프로세스
    3. 우리 실험이 실제로 쓰는 GPU 수 (코드상 설정)
    4. 드라이버·CUDA·PyTorch 버전

주의
    서버가 공용이라면 다른 사용자의 프로세스도 보일 수 있다.
    '장착된 GPU 수' 와 '우리가 쓰는 GPU 수' 는 다를 수 있으므로
    두 값을 나누어 보고한다.

사용법
    cd ~/이윤우 && source .venv/bin/activate
    python gpu_check.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys


def run(cmd: list, timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return "(실행 실패: %s)" % type(e).__name__


def section(t: str) -> None:
    print("\n" + "=" * 86)
    print(t)
    print("=" * 86)


print("=" * 86)
print("실험 서버 GPU 환경 확인")
print("=" * 86)
print("  호스트   " + run(["hostname"]))
print("  계정     " + (os.environ.get("USER") or run(["whoami"])))
print("  작업경로 " + os.getcwd())

# ── 1. 장착 GPU ─────────────────────────────────────────────────────
section("1. 서버에 장착된 GPU")

if not shutil.which("nvidia-smi"):
    print("  nvidia-smi 를 찾을 수 없습니다. GPU 드라이버가 없거나 PATH 문제입니다.")
else:
    q = run(["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"])
    if q.startswith("(") or not q:
        print("  조회 실패: " + q)
    else:
        lines = [l for l in q.split("\n") if l.strip()]
        print("  장착 수량  %d개\n" % len(lines))
        print("  " + "번호".ljust(5) + "모델명".ljust(26)
              + "전체(GB)".rjust(10) + "사용(GB)".rjust(10)
              + "여유(GB)".rjust(10) + "사용률".rjust(8) + "온도".rjust(7))
        tot_all = used_all = 0.0
        for l in lines:
            c = [x.strip() for x in l.split(",")]
            idx, name = c[0], c[1]
            tot, used, free = float(c[2]) / 1024, float(c[3]) / 1024, float(c[4]) / 1024
            util, temp = c[5], c[6]
            tot_all += tot
            used_all += used
            print("  " + idx.ljust(5) + name[:25].ljust(26)
                  + ("%10.1f" % tot) + ("%10.1f" % used) + ("%10.1f" % free)
                  + (util + "%").rjust(8) + (temp + "C").rjust(7))
        print("\n  합계  전체 %.1f GB / 사용 중 %.1f GB / 여유 %.1f GB"
              % (tot_all, used_all, tot_all - used_all))

    # 드라이버·CUDA
    v = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    drv = v.split("\n")[0] if v and not v.startswith("(") else "?"
    smi = run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    print("  드라이버 %s / CUDA %s" % (drv, m.group(1) if m else "?"))

# ── 2. 점유 프로세스 ────────────────────────────────────────────────
section("2. 현재 GPU 를 쓰고 있는 프로세스")
if shutil.which("nvidia-smi"):
    p = run(["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
             "--format=csv,noheader,nounits"])
    lines = [l for l in p.split("\n") if l.strip()] if p and not p.startswith("(") else []
    if not lines:
        print("  실행 중인 프로세스가 없습니다. (GPU 가 비어 있음)")
    else:
        print("  " + "PID".ljust(9) + "메모리(GB)".rjust(11) + "  사용자".ljust(12) + "프로세스")
        for l in lines:
            c = [x.strip() for x in l.split(",")]
            pid, mem, pname = c[1], c[2], c[3] if len(c) > 3 else "?"
            owner = run(["ps", "-o", "user=", "-p", pid]) or "?"
            print("  " + pid.ljust(9) + ("%11.1f" % (float(mem) / 1024))
                  + "  " + owner[:11].ljust(12) + os.path.basename(pname)[:36])
        print("\n  ※ 서버가 공용이면 다른 사용자의 프로세스도 보입니다.")

# ── 3. 우리 실험이 쓰는 GPU ─────────────────────────────────────────
section("3. 우리 실험이 실제로 사용하는 GPU")

cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
print("  CUDA_VISIBLE_DEVICES 환경변수  " + (cvd if cvd is not None else "(설정 없음 — 전체 접근 가능)"))

try:
    import torch
    print("\n  PyTorch        " + torch.__version__)
    print("  CUDA 사용 가능   " + str(torch.cuda.is_available()))
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("  PyTorch 가 보는 GPU 수  %d개" % n)
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        print("    [%d] %s  %.1f GB  compute %d.%d"
              % (i, pr.name, pr.total_memory / 1024 ** 3, pr.major, pr.minor))
except ImportError:
    print("\n  PyTorch 가 설치되어 있지 않습니다. (.venv 를 활성화했는지 확인)")
except Exception as e:
    print("\n  PyTorch 조회 실패: %s %s" % (type(e).__name__, e))

# 코드에서 지정한 GPU
print("\n  스크립트의 GPU 지정")
found = False
for f, label in [("rerun_sections3.py", "③ 한국어 평이화 (HARI)"),
                 ("prompt_ladder.py", "④ 검증 (NLI)"),
                 ("model_compare.py", "모델 비교"),
                 ("translate_verify2.py", "⑥ 번역 검증"),
                 ("eval_A_vs_E.py", "Glossary 평가")]:
    if not os.path.exists(f):
        continue
    with open(f, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    hits = set()
    for m in re.finditer(r'device_map\s*=\s*["\']([^"\']+)["\']', src):
        hits.add("device_map=" + m.group(1))
    for m in re.finditer(r'--gpu["\'][^)]*default\s*=\s*(\d+)', src):
        hits.add("--gpu 기본값 " + m.group(1))
    for m in re.finditer(r'cuda:(\d+)', src):
        hits.add("cuda:" + m.group(1))
    if hits:
        found = True
        print("    %-26s %s" % (label, ", ".join(sorted(hits))))
if not found:
    print("    (해당 패턴을 찾지 못했습니다)")

# ── 4. 요약 ─────────────────────────────────────────────────────────
section("4. 보고용 요약")
n_installed = 0
if shutil.which("nvidia-smi"):
    q = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
    n_installed = len([l for l in q.split("\n") if l.strip()]) if q and not q.startswith("(") else 0
n_torch = 0
try:
    import torch
    n_torch = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    pass

print("  서버에 장착된 GPU            %d개" % n_installed)
print("  우리 실험이 접근 가능한 GPU   %d개" % n_torch)
print("  실험 1회 실행 시 사용하는 수   1개  (단일 GPU 로 순차 실행)")
print("""
  해석
    · 장착 수량과 우리가 쓰는 수량은 다릅니다. 현재 모든 스크립트가
      단일 GPU(cuda:0)에서 순차 실행되도록 작성되어 있습니다.
    · 여러 GPU 로 분산하거나 병렬 실행하도록 구성한 적은 없습니다.
    · 서버가 공용이면 2절의 프로세스 목록에 다른 사용자 작업이 보입니다.""")
print("=" * 86)
