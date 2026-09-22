#!/usr/bin/env bash
# M2 smoke test: tiny anchor training runs (both plain and DIAYN variants)
# to validate the cond-agent training path before the full matrix.
set -euo pipefail
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled
export PYTHONPATH=.:${PYTHONPATH:-}
export MKL_SERVICE_FORCE_INTEL=1
export TF_CPP_MIN_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
LIB="$REPO/exp/anchor_lib/walker_lib.pt"
EXP="$REPO/exp/smoke"

smoke() {  # smoke <tag> [overrides...]
  local tag=$1; shift
  echo "=== [$tag] smoke start $(date +%T)"
  python3 train.py task=walker_walk \
    snapshot_load_dir="$REPO/models/genrl_walker_500k_2.pt" \
    num_train_frames=410 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=200 snapshot_every_frames=25_000 \
    agent=genrl_anchor train_from_data=False \
    agent.anchor.lib_path="$LIB" \
    "hydra.run.dir=$EXP/$tag" \
    "snapshot_dir=$EXP/$tag" \
    "$@" > "logs/$tag.log" 2>&1
  echo "=== [$tag] smoke done $(date +%T)"
}

echo "### smoke: anchor-only ###"
smoke smoke_anchor_off agent.anchor.pure_prob=1.0 seed=99
echo "### smoke: anchor+DIAYN ###"
smoke smoke_diayn agent.anchor.pure_prob=1.0 agent.anchor.diayn=true seed=99
echo "### smoke: mixup-on ###"
smoke smoke_mixon agent.anchor.pure_prob=0.5 seed=99

# end-to-end eval-path smoke (train -> load -> act -> descriptors), 2 eps/arm.
# Catches eval-script bugs BEFORE the 50k-frame runs (2026-08-31 lesson:
# eval_anchors' stoch-flattening bug only surfaced on the first real eval).
echo "### smoke: eval path train16+heldout @2eps ###"
python3 experiments/eval_anchors.py train16 \
  "$EXP/smoke_mixon/last_snapshot.pt" "$EXP/smoke_eval_train16.json" --n_eps 2 \
  >> logs/smoke_eval.log 2>&1
python3 experiments/eval_anchors.py heldout \
  "$EXP/smoke_mixon/last_snapshot.pt" "$EXP/smoke_eval_heldout.json" \
  --n_eps 2 --scan_eps 2 >> logs/smoke_eval.log 2>&1
echo "### SMOKE ALL OK $(date +%T) ###"
