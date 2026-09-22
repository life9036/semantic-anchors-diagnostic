#!/usr/bin/env bash
# run_cheetah_c4.sh — M4 fix (post-outcome supplement, 2026-09-10):
# Cheetah representation-isolation baselines, mirroring Walker C4.
#   text   — VLM text features via fixed projection (semantics, no temporal
#            trajectory pooling), 3 seeds, heldout four-arm included
#   random — fixed seeded random unit codes (no semantic content), 3 seeds,
#            no heldout (random has no semantic query path, same as Walker)
# Both: same semantic reward targets S_i, same 50k budget, pure_prob=1.0
# (matched to c5_anchor_off). Seeds 1-3 match c5_anchor_off s1-3 for
# matched-pair analysis (df=2, low precision disclosed).
# Protocol: review-stage/M4_PROTOCOL_REVISION.md (post-outcome, exploratory;
# no new gates — descriptive endpoints).
# Usage (setsid):
#   setsid bash -c 'cd /root/autodl-tmp/genrl && export CUDA_VISIBLE_DEVICES=0 \
#     PARTITION=A && bash experiments/run_cheetah_c4.sh' </dev/null \
#     > logs/cheetah_c4_A.log 2>&1 &
set -u
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=.:${PYTHONPATH:-}
export CHEETAH=1
EXP=exp/main
RES=exp/main_results
LIB=exp/anchor_lib/cheetah_lib.pt
SNAP=models/genrl_cheetah_500k_2.pt
PARTITION="${PARTITION:-A}"
mkdir -p "$EXP" "$RES" logs
LOGF="logs/cheetah_c4_${PARTITION}.log"
exec >> "$LOGF" 2>&1

ts() { date '+%H:%M:%S'; }
flag() { touch "exp/main/${1}_cheetah_DONE.flag"; }

train() {  # train <tag> <mode> <seed>
  local tag=$1 mode=$2 s=$3
  echo "=== [$tag] train start $(ts)"
  python3 train.py task=cheetah_run \
    snapshot_load_dir="$REPO/$SNAP" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" "snapshot_dir=$EXP/$tag" \
    agent=genrl_anchor train_from_data=False \
    "agent.anchor.lib_path=$LIB" \
    agent.anchor.pure_prob=1.0 agent.anchor.cond_mode=$mode seed=$s \
    > "logs/$tag.log" 2>&1 \
    || { echo "=== [$tag] train FAILED rc=$? $(ts)"; return 1; }
  echo "=== [$tag] train done $(ts)"
}

eval_train16() {  # eval_train16 <tag> <mode>
  local tag=$1 mode=$2
  python3 experiments/eval_anchors_v2.py train16v2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_train16v2.json" \
    --cond-mode "$mode" \
    || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts)"; return 1; }
  python3 experiments/eval_anchors.py idrecover \
    "$RES/${tag}_train16v2.json.feats.npz" "$RES/${tag}_idrec_v2.json" \
    || { echo "=== [$tag] idrecover FAILED rc=$? $(ts)"; return 1; }
  # npz kept for audit (matches Walker C4 precedent; small files)
}

run_c4() {  # run_c4 <mode> <seed>
  local mode=$1 s=$2; local tag="c5base_${mode}_s$s"
  local donef="exp/main/${tag}_cheetah_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train "$tag" "$mode" "$s" || return 1
  eval_train16 "$tag" "$mode" || return 1
  if [ "$mode" = text ]; then
    python3 experiments/eval_anchors_v2.py heldoutv2 \
      "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_heldoutv2.json" \
      --cond-mode text --sig-file "$RES/${tag}_train16v2.json" \
      || { echo "=== [$tag] heldoutv2 FAILED rc=$? $(ts)"; return 1; }
  fi
  rm -f "$EXP/$tag"/second_last_snapshot.pt "$EXP/$tag"/snapshot_1000.pt \
        "$EXP/$tag"/snapshot_25000.pt 2>/dev/null || true
  flag "$tag" && echo "=== [$tag] ALL DONE $(ts)"
}

if [ "$PARTITION" = A ]; then
  JOBS=(random:1 random:2 random:3 text:1)
elif [ "$PARTITION" = B ]; then
  JOBS=(text:2 text:3)
else
  echo "unknown PARTITION=$PARTITION" >&2; exit 2
fi

echo "### CHEETAH_C4 PART=$PARTITION start $(ts) (${#JOBS[@]} runs)"
for spec in "${JOBS[@]}"; do
  mode=${spec%%:*}; s=${spec##*:}
  run_c4 "$mode" "$s"
done
echo "### CHEETAH_C4 PART=$PARTITION ALL DONE $(ts) ###"
echo "PARTITION_C4_${PARTITION}_EXIT=0"
