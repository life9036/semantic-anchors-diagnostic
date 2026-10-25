#!/usr/bin/env bash
# run_permuted.sh — M5 post-outcome addendum (advisor Round 1 A3, 2026-10-03):
# Walker random-code + PERMUTED reward target cell, completing the 2x2
# {semantic, random} x {identity, permuted} target-pairing grid.
#   cond_mode=random (fixed seeded random unit codes, no semantics)
#   + shuffle=true (reward targets S_perm[i], the predeclared derangement
#     SHUFFLE_PERM = (i+5) mod 16 — same perm as the shuffled-anchor control)
#   3 seeds s1-3, matching the Walker random-code baselines (m4base_random)
#   seed-for-seed for matched-pair analysis (df=2, low precision disclosed).
# Same semantic-target budget: 50k imagined frames, pure_prob=1.0.
# Evaluation is IDENTICAL to the random-code runs: behavior is evaluated
# against the anchor library S_i (train16v2 + idrecover; random has no
# heldout query path, same as Walker C4). Only training targets differ.
# Protocol: review-stage/M5_PROTOCOL_REVISION.md (post-outcome, exploratory;
# no new gates — descriptive endpoints G1-G4).
# Usage (setsid, PARTITION=A|B):
#   setsid bash -c 'cd /root/autodl-tmp/genrl && export CUDA_VISIBLE_DEVICES=0 \
#     PARTITION=A && bash experiments/run_permuted.sh' </dev/null \
#     > logs/m5_permuted_A.log 2>&1 &
set -u
REPO=/root/autodl-tmp/genrl
cd "$REPO"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
export WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=.:${PYTHONPATH:-}
EXP=exp/main
RES=exp/main_results
LIB=exp/anchor_lib/walker_lib.pt
SNAP=models/genrl_walker_500k_2.pt
PARTITION="${PARTITION:-A}"
mkdir -p "$EXP" "$RES" logs
LOGF="logs/m5_permuted_${PARTITION}.log"
exec >> "$LOGF" 2>&1

ts() { date '+%H:%M:%S'; }
flag() { touch "exp/main/${1}_DONE.flag"; }

train() {  # train <tag> <seed>
  local tag=$1 s=$2
  echo "=== [$tag] train start $(ts)"
  python3 train.py task=walker_walk \
    snapshot_load_dir="$REPO/$SNAP" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" "snapshot_dir=$EXP/$tag" \
    agent=genrl_anchor train_from_data=False \
    "agent.anchor.lib_path=$LIB" \
    agent.anchor.pure_prob=1.0 agent.anchor.cond_mode=random \
    agent.anchor.shuffle=true seed=$s \
    > "logs/$tag.log" 2>&1 \
    || { echo "=== [$tag] train FAILED rc=$? $(ts)"; return 1; }
  echo "=== [$tag] train done $(ts)"
}

eval_train16() {  # eval_train16 <tag>
  local tag=$1
  python3 experiments/eval_anchors_v2.py train16v2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_train16v2.json" \
    --cond-mode random \
    || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts)"; return 1; }
  python3 experiments/eval_anchors.py idrecover \
    "$RES/${tag}_train16v2.json.feats.npz" "$RES/${tag}_idrec_v2.json" \
    || { echo "=== [$tag] idrecover FAILED rc=$? $(ts)"; return 1; }
  # npz kept for audit (matches C4 precedent)
}

run_one() {  # run_one <seed>
  local s=$1; local tag="m5base_randperm_s$s"
  local donef="exp/main/${tag}_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train "$tag" "$s" || return 1
  eval_train16 "$tag" || return 1
  rm -f "$EXP/$tag"/second_last_snapshot.pt "$EXP/$tag"/snapshot_1000.pt \
        "$EXP/$tag"/snapshot_25000.pt 2>/dev/null || true
  flag "$tag" && echo "=== [$tag] ALL DONE $(ts)"
}

if [ "$PARTITION" = A ]; then
  JOBS=(1 2)
elif [ "$PARTITION" = B ]; then
  JOBS=(3)
else
  echo "unknown PARTITION=$PARTITION" >&2; exit 2
fi

echo "### M5_PERMUTED PART=$PARTITION start $(ts) (${#JOBS[@]} runs)"
for s in "${JOBS[@]}"; do
  run_one "$s"
done
echo "### M5_PERMUTED PART=$PARTITION ALL DONE $(ts) ###"
echo "M5_PARTITION_${PARTITION}_EXIT=0"
