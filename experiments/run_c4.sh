#!/usr/bin/env bash
# C4 representation-isolation baselines (Round-1 review fix):
#   text   — VLM text features via fixed projection (semantics, no temporal
#            trajectory pooling), 3 seeds
#   random — fixed seeded random unit codes (no semantic content), 3 seeds
# Both: same semantic reward targets S_i, same 50k budget, pure_prob=1.0
# (matched to anchor_off, the mixup-off comparator).
# Usage: PARTITION=C bash experiments/run_c4.sh  (single partition, 6 runs)
set -euo pipefail
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled
export PYTHONPATH=.:${PYTHONPATH:-}
export MKL_SERVICE_FORCE_INTEL=1
export HYDRA_FULL_ERROR=1

LIB="$REPO/exp/anchor_lib/walker_lib.pt"
EXP="$REPO/exp/main"
RES="$REPO/exp/main_results"
mkdir -p "$EXP" "$RES" logs

ANCHOR_BASE="agent=genrl_anchor train_from_data=False"
ANCHOR_LIB="agent.anchor.lib_path=$LIB"

train() {  # train <tag> <cond_mode> <seed>
  local tag=$1 mode=$2 s=$3
  echo "=== [$tag] train start $(date +%T)"
  python3 train.py task=walker_walk \
    snapshot_load_dir="$REPO/models/genrl_walker_500k_2.pt" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" "snapshot_dir=$EXP/$tag" \
    $ANCHOR_BASE $ANCHOR_LIB \
    agent.anchor.pure_prob=1.0 agent.anchor.cond_mode=$mode seed=$s \
    > "logs/$tag.log" 2>&1
  echo "=== [$tag] train done $(date +%T)"
}

run_baseline() {  # run_baseline <mode> <seed>
  local mode=$1 s=$2; local tag="m4base_${mode}_s$s"
  train "$tag" "$mode" "$s"
  # train16v2 with the mode's cond matrix (targets stay semantic S_i);
  # random mode has no heldout query path -> train16v2 + idrecover only.
  python3 experiments/eval_anchors_v2.py train16v2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_train16v2.json" \
    --cond-mode "$mode" >> "logs/$tag.log" 2>&1
  python3 experiments/eval_anchors.py idrecover \
    "$RES/${tag}_train16v2.json.feats.npz" "$RES/${tag}_idrec_v2.json" \
    >> "logs/$tag.log" 2>&1
  if [ "$mode" = text ]; then
    python3 experiments/eval_anchors_v2.py heldoutv2 \
      "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_heldoutv2.json" \
      --cond-mode text --sig-file "$RES/${tag}_train16v2.json" \
      >> "logs/$tag.log" 2>&1
  fi
  rm -f "$EXP/$tag"/second_last_snapshot.pt "$EXP/$tag"/snapshot_1000.pt \
        "$EXP/$tag"/snapshot_25000.pt 2>/dev/null || true
  touch "exp/main/${tag}_DONE.flag"
  echo "=== [$tag] all done $(date +%T)"
}

C4_SEEDS="${C4_SEEDS:-1 2 3}"
C4_MODE="${C4_MODE:-text random}"
for s in $C4_SEEDS; do
  for mode in $C4_MODE; do
    run_baseline "$mode" "$s"
  done
done
echo "### C4 ALL DONE $(date +%T) ###"
echo "PARTITION_C_EXIT=0"
