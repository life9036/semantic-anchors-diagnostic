#!/usr/bin/env bash
# Trimmed pilot driver (phase 2, after r1): runs r2-r6 WITHOUT mid-training
# real-env evals (eval_every_frames=999999) — posthoc does unified evaluation.
# r1 already completed under the first driver.
set -u
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled
export PYTHONPATH=.:${PYTHONPATH:-}
export MKL_SERVICE_FORCE_INTEL=1
export TF_CPP_MIN_LOG_LEVEL=3
SNAP_DIR="$REPO/exp"

run() {  # run <tag> [hydra overrides...]
  local tag=$1; shift
  echo "=== [$tag] start $(date +%T)"
  python3 train.py task=stickman_walk \
    snapshot_load_dir="$REPO/models/genrl_stickman_500k_2.pt" \
    num_train_frames=30_010 batch_size=32 batch_length=32 \
    agent.imag_reward_fn=anchor_reward \
    eval_modality=task_imag train_from_data=False \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    snapshot_every_frames=15_000 seed=1 \
    "hydra.run.dir=$SNAP_DIR/$tag" \
    "snapshot_dir=$SNAP_DIR/$tag" \
    "$@" > "logs/$tag.log" 2>&1
  echo "=== [$tag] done $(date +%T) rc=$?"
}

echo "### trimmed: r2-r6 (no mid-training env evals) ###"
run p2_r2_anchor_walk_ent +agent.imag_reward_args.beta=0.1 +agent.imag_reward_args.anchor=walk
run p2_r3_entropy_only    agent.imag_reward_fn=entropy_reward +agent.imag_reward_args.beta=0.1
run p2_r4_anchor_run      +agent.imag_reward_args.beta=0.0 +agent.imag_reward_args.anchor=run
run p3_r5_delta_run_slow agent.imag_reward_fn=delta_anchor_reward \
  +agent.imag_reward_args.pos=run +agent.imag_reward_args.neg=slow +agent.imag_reward_args.beta=0.0
run p3_r6_delta_slow_run agent.imag_reward_fn=delta_anchor_reward \
  +agent.imag_reward_args.pos=slow +agent.imag_reward_args.neg=run +agent.imag_reward_args.beta=0.0

echo "### posthoc analysis ###"
python3 - <<'PY'
import json
SNAP_DIR = "/root/autodl-tmp/genrl/exp"
snaps = [f"{SNAP_DIR}/{t}/snapshot_15000.pt" for t in [
    "p2_r1_anchor_walk", "p2_r2_anchor_walk_ent", "p2_r3_entropy_only",
    "p2_r4_anchor_run", "p3_r5_delta_run_slow", "p3_r6_delta_slow_run"]]
json.dump({"snapshots": snaps,
           "anchor_specs": ["walk", "walk+ent", "null", "run",
                            "delta_run-slow", "delta_slow-run"],
           "num_eps": 5, "out": f"{SNAP_DIR}/pilot_posthoc.json"},
          open("exp/posthoc_cfg.json", "w"), indent=2)
print("posthoc_cfg written")
PY
python3 pilots/pilot_posthoc.py > logs/posthoc.log 2>&1
echo "=== posthoc done rc=$? ==="
echo "### TRIMMED DRIVER DONE $(date +%T) ###"
touch exp/PILOTS_ALL_DONE.flag
