#!/usr/bin/env bash
# Walker matrix driver (M1/M2/M3): serial runs on one 4090.
# Usage: bash experiments/run_matrix.sh {m1|m2|m3|aggregate-m2|aggregate-m34}
#   m1   GenRL direct-prompt baselines (3 seeds, 50k frames)
#   m2   anchor-only (5) + anchor+DIAYN (5, paired seeds) + shuffled (3)
#        + heldout eval on the off branch (10 eps/arm, M3 comparison)
#   m3   mixup-on (5 seeds, paired with m2's off branch) + heldout (20 eps/arm)
# Each run: train 50k frames, then offline eval, then snapshot cleanup.
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
EXP="$REPO/exp/main"
RES="$REPO/exp/main_results"
mkdir -p "$EXP" "$RES" logs

train() {  # train <tag> [hydra overrides...]
  local tag=$1; shift
  echo "=== [$tag] train start $(date +%T)"
  python3 train.py task=walker_walk \
    snapshot_load_dir="$REPO/models/genrl_walker_500k_2.pt" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" \
    "snapshot_dir=$EXP/$tag" \
    "$@" > "logs/$tag.log" 2>&1
  echo "=== [$tag] train done $(date +%T)"
}

eval_train16() {  # eval_train16 <tag>
  local tag=$1
  python3 experiments/eval_anchors.py train16 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_train16.json" \
    >> "logs/$tag.log" 2>&1
  python3 experiments/eval_anchors.py idrecover \
    "$RES/${tag}_train16.json.feats.npz" "$RES/${tag}_idrec.json" \
    >> "logs/$tag.log" 2>&1
  # feature npz (~0.9GB) is only consumed by idrecover — drop it
  rm -f "$RES/${tag}_train16.json.feats.npz"
}

eval_heldout() {  # eval_heldout <tag> <n_eps>
  local tag=$1 n_eps=$2
  python3 experiments/eval_anchors.py heldout \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_heldout.json" \
    --n_eps "$n_eps" >> "logs/$tag.log" 2>&1
}

cleanup() {  # keep last + final snapshot only
  local tag=$1
  rm -f "$EXP/$tag"/second_last_snapshot.pt \
        "$EXP/$tag"/snapshot_1000.pt \
        "$EXP/$tag"/snapshot_25000.pt 2>/dev/null || true
  echo "=== [$tag] cleaned $(date +%T)"
}

ANCHOR_BASE="agent=genrl_anchor train_from_data=False"
ANCHOR_LIB="agent.anchor.lib_path=$LIB"

# two-GPU partitioning: PARTITION=A|B splits the independent runs of m2/m3.
# A: anchor_off s1-2, diayn s1-3, shuffle s11-13, mixon s1-2 (10 runs)
# B: anchor_off s3-5, diayn s4-5,             mixon s3-5 (8 runs)
PARTITION="${PARTITION:-A}"

run_anchor_off() {  # run_anchor_off <seed>
  local s=$1; local tag="m2_anchor_off_s$s"
  train "$tag" $ANCHOR_BASE $ANCHOR_LIB agent.anchor.pure_prob=1.0 seed=$s
  eval_train16 "$tag"
  eval_heldout "$tag" 10
  cleanup "$tag"
}
run_diayn() {  # run_diayn <seed>
  local s=$1; local tag="m2_diayn_s$s"
  train "$tag" $ANCHOR_BASE $ANCHOR_LIB \
    agent.anchor.pure_prob=1.0 agent.anchor.diayn=true seed=$s
  eval_train16 "$tag"
  cleanup "$tag"
}
run_shuffle() {  # run_shuffle <seed>
  local s=$1; local tag="m2_shuffle_s$s"
  train "$tag" $ANCHOR_BASE $ANCHOR_LIB \
    agent.anchor.pure_prob=1.0 agent.anchor.shuffle=true seed=$s
  eval_train16 "$tag"
  cleanup "$tag"
}
run_mixon() {  # run_mixon <seed>
  local s=$1; local tag="m3_mixon_s$s"
  train "$tag" $ANCHOR_BASE $ANCHOR_LIB agent.anchor.pure_prob=0.5 seed=$s
  eval_train16 "$tag"
  eval_heldout "$tag" 20
  cleanup "$tag"
}

case "${1:-}" in
m1)
  if [ "$PARTITION" = "A" ]; then
    SEEDS="1 2"
  else
    SEEDS="3"
  fi
  for s in $SEEDS; do
    tag="m1_genrl_s$s"
    train "$tag" agent.imag_reward_fn=video_text_reward \
      train_from_data=False \
      eval_modality=task_imag eval_every_frames=5000 seed=$s
    python3 experiments/eval_anchors.py genrl \
      "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_genrl.json" \
      >> "logs/$tag.log" 2>&1
    cleanup "$tag"
  done
  ;;
m2)
  if [ "$PARTITION" = "A" ]; then
    for s in 1 2; do run_anchor_off $s; done
    for s in 1 2 3; do run_diayn $s; done
    for s in 11 12 13; do run_shuffle $s; done
  else
    for s in 3 4 5; do run_anchor_off $s; done
    for s in 4 5; do run_diayn $s; done
  fi
  ;;
m3)
  if [ "$PARTITION" = "A" ]; then
    for s in 1 2; do run_mixon $s; done
  else
    for s in 3 4 5; do run_mixon $s; done
  fi
  ;;
m2rest)  # A-partition tail: only the shuffled-control runs (resume-safe)
  for s in 11 12 13; do run_shuffle $s; done
  ;;
m4ref)  # R030: GenRL direct-prompt references on 3 held-out prompts x1 seed
  if [ "$PARTITION" = "A" ]; then
    REFS=("crouching down:0" "jumping in place:1")
  else
    REFS=("walking with arms swinging:5")
  fi
  for entry in "${REFS[@]}"; do
    p="${entry%%:*}"; q="${entry##*:}"
    tag="m4ref_$(echo "$p" | tr " " "_")"
    train "$tag" agent.imag_reward_fn=video_text_reward \
      train_from_data=False \
      "+agent.imag_reward_args.task_prompt=$p" \
      eval_every_frames=999999 seed=1
    python3 experiments/eval_anchors.py genrl \
      "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_genrl.json" \
      >> "logs/$tag.log" 2>&1
    python3 experiments/eval_anchors.py refdesc \
      "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_refdesc.json" \
      --q "$q" >> "logs/$tag.log" 2>&1
    cleanup "$tag"
    # tight disk policy: keep only the final model
    rm -f "$EXP/$tag"/snapshot_*.pt "$EXP/$tag"/second_last_snapshot.pt
  done
  ;;
aggregate-m2)
  python3 experiments/eval_anchors.py aggregate "$RES" \
    "$RES/aggregate_m2.json" --task m2
  ;;
aggregate-m34)
  python3 experiments/eval_anchors.py aggregate "$RES" \
    "$RES/aggregate_m34.json" --task m34
  ;;
*)
  echo "usage: bash experiments/run_matrix.sh {m1|m2|m3|aggregate-m2|aggregate-m34}"
  exit 2
  ;;
esac

touch "$EXP/${1}_${PARTITION}_DONE.flag"
echo "### PHASE $1 (PART=$PARTITION) ALL DONE $(date +%T) ###"
