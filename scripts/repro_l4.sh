#!/usr/bin/env bash
# One-command replication of the headline numbers on a second machine.
#
# Why this exists: every absolute number in this repository was measured on a
# laptop RTX 5060 (8 GB, sm_120).  The structural claims -- the weight-byte
# model, the lm_head share, the min-p saving, the KV-stream growth with context
# -- should survive a different card, and the only honest way to show that is to
# rerun the same stages somewhere else.  This script is that rerun, kept in the
# repo so it costs one command instead of an afternoon.
#
# It does not install anything.  Prepare the host first (see
# docs/环境与复现.md): Ubuntu 24.04, CUDA 13 toolchain, python 3.12, then
#   python3.12 -m venv /opt/cloud-inference-framework/.venv
#   /opt/cloud-inference-framework/.venv/bin/pip install vllm==0.31.0
#   # Qwen3-0.6B into /opt/cloud-inference-framework/models/Qwen3-0.6B
#   # this checkout's scripts/ and kernels/ into /opt/cloud-inference-framework/
#
# Usage:
#   bash scripts/repro_l4.sh              # every stage, newest results win
#   bash scripts/repro_l4.sh --dry-run    # print the plan and the estimate
#   STAGES="quant-probe quant-context" bash scripts/repro_l4.sh
#
# Each stage goes through scripts/run_stage.sh, so every run lands in its own
# timestamped directory with a .log and an .exitcode, exactly like the laptop.
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="$project_dir/scripts/run_stage.sh"
dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

# stage | expected minutes on the laptop | why it is here
# Cheapest probes first, the two long paired runs last.  A 24 GB card is not
# necessarily faster at these tiny shapes: the point is a second measurement.
plan=(
    "preflight|1|environment facts and a real forward pass"
    "quant-probe|3|fp8 W8A8 availability and single-operator speed"
    "minp-sweep|6|BLOCK_SIZE / num_warps response of the min-p kernel"
    "minp-compare|6|upstream vs two-launch patch vs launch knobs"
    "quant-context|10|decode interval at 128 / 2048 / 8192 prompt tokens"
    "quant-decode|12|same-process fp16 vs fp8 decode interval"
    "abba-knob|25|paired end-to-end min-p saving at batch 1/2/4/8"
)

if [[ -n "${STAGES:-}" ]]; then
    wanted=" $STAGES "
else
    wanted=""
    for row in "${plan[@]}"; do wanted+=" ${row%%|*} "; done
fi

total=0
selected=()
for row in "${plan[@]}"; do
    stage="${row%%|*}"
    rest="${row#*|}"
    minutes="${rest%%|*}"
    why="${rest#*|}"
    if [[ "$wanted" == *" $stage "* ]]; then
        selected+=("$stage")
        total=$((total + minutes))
        printf '  %-14s %3s min  %s\n' "$stage" "$minutes" "$why"
    fi
done

printf 'stages: %s\n' "${selected[*]}"
printf 'estimated wall clock: about %d min on the laptop\n' "$total"
printf 'results land in /opt/cloud-inference-framework/results/<stage>-<utc>/\n'
if (( dry_run )); then
    exit 0
fi

failed=()
for stage in "${selected[@]}"; do
    printf '\n=== %s ===\n' "$stage"
    if [[ "$stage" == "preflight" ]]; then
        # preflight is a standalone shell script, not a run_stage stage
        if bash "$project_dir/scripts/preflight.sh"; then ok=1; else ok=0; fi
    elif bash "$run" "$stage"; then
        ok=1
    else
        ok=0
    fi
    if (( ok )); then
        printf '=== %s ok ===\n' "$stage"
    else
        # A failed stage is data too: keep the directory and keep going, then
        # report the whole list at the end.
        printf '=== %s FAILED (kept) ===\n' "$stage"
        failed+=("$stage")
    fi
done

printf '\n--- summary ---\n'
if (( ${#failed[@]} )); then
    printf 'failed: %s\n' "${failed[*]}"
    printf 'kept directories: /opt/cloud-inference-framework/results/<stage>-<utc>/\n'
    exit 1
fi
printf 'all %d stages passed\n' "${#selected[@]}"
printf 'compare against docs/第六阶段量化干预与上下文影响-20261018.md\n'
