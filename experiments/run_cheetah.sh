#!/usr/bin/env bash
# run_cheetah.sh — Cheetah 确认子集（B5）: mixon 5 + anchor_off 5 + diayn 3 + genrl 3 = 16 run
# 训练 + v2 评估一体，skip-if-DONE 幂等，PARTITION=A|B 两分区并行。
# Usage (setsid):
#   setsid bash -c 'cd ~/autodl-tmp/genrl && export CUDA_VISIBLE_DEVICES=0 \
#     PARTITION=A && bash experiments/run_cheetah.sh' </dev/null > logs/cheetah_A.log 2>&1 &
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
LOGF="logs/cheetah_${PARTITION}.log"
exec >> "$LOGF" 2>&1

ts() { date '+%H:%M:%S'; }
flag() { touch "exp/main/${1}_cheetah_DONE.flag"; }

train() {  # train <tag> <seed> [extra args...]
  local tag=$1 s=$2; shift 2
  echo "=== [$tag] train start $(ts)"
  python3 train.py task=cheetah_run \
    snapshot_load_dir="$REPO/$SNAP" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" "snapshot_dir=$EXP/$tag" \
    agent=genrl_anchor train_from_data=False \
    "agent.anchor.lib_path=$LIB" "$@" seed=$s \
    > "logs/$tag.log" 2>&1 \
    || { echo "=== [$tag] train FAILED rc=$? $(ts)"; return 1; }
  echo "=== [$tag] train done $(ts)"
}

train_genrl() {  # train_genrl <tag> <seed>  (vanilla GenRL prompt baseline)
  local tag=$1 s=$2
  echo "=== [$tag] train start $(ts)"
  python3 train.py task=cheetah_run \
    snapshot_load_dir="$REPO/$SNAP" \
    num_train_frames=50_010 batch_size=32 batch_length=32 \
    use_wandb=false use_tb=false \
    eval_every_frames=999999 \
    save_every_frames=1000 snapshot_every_frames=25_000 \
    "hydra.run.dir=$EXP/$tag" "snapshot_dir=$EXP/$tag" \
    agent.imag_reward_fn=video_text_reward train_from_data=False \
    eval_modality=task_imag eval_every_frames=5000 seed=$s \
    > "logs/$tag.log" 2>&1 \
    || { echo "=== [$tag] train FAILED rc=$? $(ts)"; return 1; }
  echo "=== [$tag] train done $(ts)"
}

eval_train16() {  # eval_train16 <tag>
  local tag=$1
  python3 experiments/eval_anchors_v2.py train16v2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_train16v2.json" \
    || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts)"; return 1; }
  python3 experiments/eval_anchors.py idrecover \
    "$RES/${tag}_train16v2.json.feats.npz" "$RES/${tag}_idrec_v2.json" \
    || { echo "=== [$tag] idrecover FAILED rc=$? $(ts)"; return 1; }
  case "$tag" in c5_anchor_off_*) ;; *) rm -f "$RES/${tag}_train16v2.json.feats.npz" ;; esac
}

eval_heldout() {  # eval_heldout <tag>
  local tag=$1
  python3 experiments/eval_anchors_v2.py heldoutv2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_heldoutv2.json" \
    --sig-file "$RES/${tag}_train16v2.json" \
    || { echo "=== [$tag] heldoutv2 FAILED rc=$? $(ts)"; return 1; }
}

eval_genrl() {  # eval_genrl <tag>
  local tag=$1
  python3 experiments/eval_anchors_v2.py genrlv2 \
    "$EXP/$tag/last_snapshot.pt" "$RES/${tag}_genrlv2.json" \
    || { echo "=== [$tag] genrlv2 FAILED rc=$? $(ts)"; return 1; }
}

run_anchor_off() {  # <seed>
  # NOTE (2026-09-03, SCI-3 refit): heldout interpolation analysis is
  # Walker-only evidence in this paper; Cheetah anchor_off runs keep
  # train16v2 + idrecover only (cross-domain C1 confirmation). Disclosed
  # as a scope cut from the original B5 preregistration.
  local s=$1; local tag="c5_anchor_off_s$s"
  local donef="exp/main/${tag}_cheetah_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train "$tag" "$s" agent.anchor.pure_prob=1.0
  eval_train16 "$tag" && flag "$tag" && echo "=== [$tag] ALL DONE $(ts)"
}

run_diayn() {  # <seed>
  local s=$1; local tag="c5_diayn_s$s"
  local donef="exp/main/${tag}_cheetah_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train "$tag" "$s" agent.anchor.pure_prob=1.0 agent.anchor.diayn=true
  eval_train16 "$tag" && flag "$tag" && echo "=== [$tag] ALL DONE $(ts)"
}

run_mixon() {  # <seed>
  local s=$1; local tag="c5_mixon_s$s"
  local donef="exp/main/${tag}_cheetah_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train "$tag" "$s" agent.anchor.pure_prob=0.5
  eval_train16 "$tag" && eval_heldout "$tag" && flag "$tag" \
    && echo "=== [$tag] ALL DONE $(ts)"
}

run_genrl() {  # <seed>
  local s=$1; local tag="c5_genrl_s$s"
  local donef="exp/main/${tag}_cheetah_DONE.flag"
  [ -f "$donef" ] && { echo "=== [$tag] skip $(ts)"; return 0; }
  train_genrl "$tag" "$s"
  eval_genrl "$tag" && flag "$tag" && echo "=== [$tag] ALL DONE $(ts)"
}

if [ "$PARTITION" = A ]; then
  JOBS=(anchor_off:1 anchor_off:2 diayn:1 diayn:2 mixon:1 mixon:2 mixon:3 genrl:1)
elif [ "$PARTITION" = B ]; then
  JOBS=(anchor_off:3 anchor_off:4 anchor_off:5 diayn:3 mixon:4 mixon:5 genrl:2 genrl:3)
else
  echo "unknown PARTITION=$PARTITION" >&2; exit 2
fi

echo "### CHEETAH PART=$PARTITION start $(ts) (${#JOBS[@]} runs)"
for spec in "${JOBS[@]}"; do
  kind=${spec%%:*}; s=${spec##*:}
  case "$kind" in
    anchor_off) run_anchor_off "$s" ;;
    diayn)      run_diayn "$s" ;;
    mixon)      run_mixon "$s" ;;
    genrl)      run_genrl "$s" ;;
  esac
done
echo "### CHEETAH PART=$PARTITION ALL DONE $(ts) ###"
echo "PARTITION_C5_${PARTITION}_EXIT=0"
