#!/usr/bin/env bash
# c4_reeval.sh — C4 基线评估重跑（eval_anchors_v2.py Round-3 修复版）
# 不重训练；只重跑 6 个 C4 run 的评估（train16v2 + idrecover + text heldoutv2）。
# Usage (setsid):
#   setsid bash -c 'cd ~/autodl-tmp/genrl && export CUDA_VISIBLE_DEVICES=0 \
#     && bash experiments/c4_reeval.sh' </dev/null >> logs/phase_C4_reeval.log 2>&1 &
set -u
cd ~/autodl-tmp/genrl
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate genrl
export WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=.:${PYTHONPATH:-}

RES=exp/main_results
EVAL="python experiments/eval_anchors_v2.py"
EVAL1="python experiments/eval_anchors.py"
LOGF="logs/phase_C4_reeval.log"
exec >> "$LOGF" 2>&1

ts() { date '+%H:%M:%S'; }
flag() { touch "exp/main/${1}_c4reeval_DONE.flag"; }

run_t16() {  # run_t16 <tag> <mode>
  local tag=$1 mode=$2
  local donef="exp/main/${tag}_c4reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] train16v2 skip (flag) $(ts)"; return 0; fi
  $EVAL train16v2 "exp/main/$tag/last_snapshot.pt" "$RES/${tag}_train16v2.json" \
        --cond-mode "$mode" \
    || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts)"; return 1; }
  $EVAL1 idrecover "$RES/${tag}_train16v2.json.feats.npz" \
                   "$RES/${tag}_idrec_v2.json" \
    || { echo "=== [$tag] idrecover FAILED rc=$? $(ts)"; return 1; }
  flag "$tag"; echo "=== [$tag] train16v2+idrec done $(ts)"
}

run_held() {  # run_held <tag>
  local tag=$1
  local donef="exp/main/${tag}_heldout_c4reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] heldoutv2 skip (flag) $(ts)"; return 0; fi
  $EVAL heldoutv2 "exp/main/$tag/last_snapshot.pt" "$RES/${tag}_heldoutv2.json" \
        --cond-mode text --sig-file "$RES/${tag}_train16v2.json" \
    || { echo "=== [$tag] heldoutv2 FAILED rc=$? $(ts)"; return 1; }
  flag "${tag}_heldout"; echo "=== [$tag] heldoutv2 done $(ts)"
}

dispatch() {  # dispatch "tag:mode"
  IFS=: read -r tag mode <<< "$1"
  run_t16 "$tag" "$mode"
}

export -f run_t16 run_held dispatch ts flag
export RES EVAL EVAL1

WAVE1=(
  "m4base_text_s1:text" "m4base_text_s2:text" "m4base_text_s3:text"
  "m4base_random_s1:random" "m4base_random_s2:random" "m4base_random_s3:random"
)
WAVE2=("m4base_text_s1" "m4base_text_s2" "m4base_text_s3")

echo "### C4 REEVAL round3 start $(ts) (wave1=${#WAVE1[@]} wave2=${#WAVE2[@]})"
printf '%s\n' "${WAVE1[@]}" | xargs -P 6 -I{} bash -c 'dispatch "$1"' _ {}
echo "### C4 REEVAL wave1 done $(ts)"
printf '%s\n' "${WAVE2[@]}" | xargs -P 3 -I{} bash -c 'run_held "$1"' _ {}
echo "### C4 REEVAL ALL DONE $(ts) ###"
echo "PARTITION_C4_EXIT=0"
