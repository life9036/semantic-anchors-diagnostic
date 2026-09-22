#!/usr/bin/env bash
# run_c4_smoke.sh — M4 preflight smoke: real train path with cond_mode=text
# on Cheetah + minimal v2 evaluation. Exercises: official snapshot load,
# anchor patch, lib build (_ensure_cond_modes incl. CLIP text path),
# imagination training, CHEETAH=1 evaluator env + text cond matrices.
# All outputs deleted after success (smoke only, no flags).
set -euo pipefail
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=.:${PYTHONPATH:-}
export CHEETAH=1
TAG=c5smoke_text_s1
python3 train.py task=cheetah_run \
  snapshot_load_dir="$REPO/models/genrl_cheetah_500k_2.pt" \
  num_train_frames=1001 batch_size=32 batch_length=32 \
  use_wandb=false use_tb=false \
  eval_every_frames=999999 \
  save_every_frames=1000 snapshot_every_frames=25000 \
  "hydra.run.dir=$REPO/exp/main/$TAG" "snapshot_dir=$REPO/exp/main/$TAG" \
  agent=genrl_anchor train_from_data=False \
  "agent.anchor.lib_path=$REPO/exp/anchor_lib/cheetah_lib.pt" \
  agent.anchor.pure_prob=1.0 agent.anchor.cond_mode=text seed=1 \
  > "logs/$TAG.log" 2>&1
python3 experiments/eval_anchors_v2.py train16v2 \
  "$REPO/exp/main/$TAG/last_snapshot.pt" \
  "$REPO/exp/main_results/${TAG}_train16v2.json" \
  --n_eps 2 --cond-mode text >> "logs/$TAG.log" 2>&1
python3 experiments/eval_anchors.py idrecover \
  "$REPO/exp/main_results/${TAG}_train16v2.json.feats.npz" \
  "$REPO/exp/main_results/${TAG}_idrec_v2.json" >> "logs/$TAG.log" 2>&1
rm -rf "$REPO/exp/main/$TAG" "$REPO"/exp/main_results/${TAG}_* 2>/dev/null || true
echo "SMOKE_OK $(date '+%H:%M:%S')"
