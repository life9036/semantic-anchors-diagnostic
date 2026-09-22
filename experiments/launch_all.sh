#!/usr/bin/env bash
# NEW BOX (dual 4090) launcher: smoke both GPUs, then run the two matrix
# partitions concurrently (GPU0=A, GPU1=B), then aggregate gates.
# NOTE: tmux sessions are killed by this platform when the SSH session ends;
# use setsid-detached processes instead (verified 2026-08-31).
set -euo pipefail
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl

echo "=== smoke GPU0 $(date +%T) ==="
CUDA_VISIBLE_DEVICES=0 bash experiments/run_smoke.sh
rm -rf exp/smoke   # fresh dirs for the GPU1 pass
echo "=== smoke GPU1 $(date +%T) ==="
CUDA_VISIBLE_DEVICES=1 bash experiments/run_smoke.sh
rm -rf exp/smoke

# cloned-image hygiene: drop any partial training state carried over from
# the old box (train.py resumes from last_snapshot.pt if present!); keep
# exp/anchor_lib + exp/preflight (validated artifacts).
echo "=== cleaning cloned partial state ==="
rm -rf exp/main exp/main_results exp/smoke
rm -f logs/m1_*.log logs/m2_*.log logs/m3_*.log logs/smoke_*.log \
      logs/setup_smoke.log logs/phase_*.log
mkdir -p exp/main_results

rm -f exp/main/m1_A_DONE.flag exp/main/m1_B_DONE.flag \
      exp/main/m2_A_DONE.flag exp/main/m2_B_DONE.flag \
      exp/main/m3_A_DONE.flag exp/main/m3_B_DONE.flag \
      exp/main/ALL_GATES_DONE.flag exp/main/PARTITION_FAILURE.flag

echo "=== launching partition drivers $(date +%T) ==="
# partition driver: runs m1->m2->m3; on failure marks PARTITION_FAILURE
setsid bash -c '
  cd '$REPO'
  export CUDA_VISIBLE_DEVICES=0 PARTITION=A
  bash experiments/run_matrix.sh m1 && \
  bash experiments/run_matrix.sh m2 && \
  bash experiments/run_matrix.sh m3
  rc=$?
  echo "PARTITION_A_EXIT=$rc $(date +%T)"
  [ $rc -ne 0 ] && touch '$REPO'/exp/main/PARTITION_FAILURE.flag
  exit 0
' </dev/null > "$REPO/logs/phase_A.log" 2>&1 &

setsid bash -c '
  cd '$REPO'
  export CUDA_VISIBLE_DEVICES=1 PARTITION=B
  bash experiments/run_matrix.sh m1 && \
  bash experiments/run_matrix.sh m2 && \
  bash experiments/run_matrix.sh m3
  rc=$?
  echo "PARTITION_B_EXIT=$rc $(date +%T)"
  [ $rc -ne 0 ] && touch '$REPO'/exp/main/PARTITION_FAILURE.flag
  exit 0
' </dev/null > "$REPO/logs/phase_B.log" 2>&1 &

# watcher: aggregate once both partitions finish m3; abort on failure
setsid bash -c '
  cd '$REPO'
  while true; do
    a=exp/main/m3_A_DONE.flag; b=exp/main/m3_B_DONE.flag
    if [ -f $a ] && [ -f $b ]; then
      bash experiments/run_matrix.sh aggregate-m2
      bash experiments/run_matrix.sh aggregate-m34
      touch exp/main/ALL_GATES_DONE.flag
      echo ALL_GATES_DONE $(date +%T)
      break
    fi
    if [ -f exp/main/PARTITION_FAILURE.flag ]; then
      echo PARTITION_FAILURE_ABORT $(date +%T)
      break
    fi
    sleep 300
  done
' </dev/null > "$REPO/logs/phase_watch.log" 2>&1 &

sleep 5
echo "=== launched: partition A (GPU0) / partition B (GPU1) / watcher $(date +%T) ==="
ps aux | grep -E "run_matrix.sh|phase_watch" | grep -v grep | awk "{print \$2, \$11, \$12, \$13}" | head -5
