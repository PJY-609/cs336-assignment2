#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
launch=benchmark_results/h200_compile_experiment_launch
exec 9>"$launch/resume.lock"
flock -n 9 || { echo "Another resume task holds the lock" >&2; exit 1; }
if pgrep -f '^([^ ]*/)?python[^ ]* .*scripts/benchmark_attention.py .*--output .*h200_compile_comparison' >/dev/null; then
  echo "A main comparison runner is already active" >&2
  exit 1
fi
trap 'echo "failed or interrupted $(date -u +%FT%TZ)" > "$launch/resume_status.txt"' ERR HUP INT TERM
echo "$$" > "$launch/resume.pid"
echo "main_and_tiles $(date -u +%FT%TZ)" > "$launch/resume_status.txt"
python scripts/benchmark_attention.py --tile-sensitivity --resume --output benchmark_results/h200_compile_comparison >> benchmark_results/h200_compile_comparison/resume.log 2>&1
echo "batch $(date -u +%FT%TZ)" > "$launch/resume_status.txt"
mkdir -p benchmark_results/h200_batch_compile_comparison
python scripts/benchmark_batch_sizes.py --resume --output benchmark_results/h200_batch_compile_comparison >> benchmark_results/h200_batch_compile_comparison/resume.log 2>&1
echo "complete $(date -u +%FT%TZ)" > "$launch/resume_status.txt"
