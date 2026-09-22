#!/usr/bin/env bash
# reeval_all.sh — Round-1-review re-evaluation driver (eval_anchors_v2.py), PARALLEL edition.
# No retraining: runs from retained last_snapshot.pt checkpoints.
# Same jobs / output JSONs / DONE flags as the sequential edition. Adds:
#   - skip-if-DONE (idempotent: safe to restart after a mid-run kill)
#   - two waves per partition: wave 1 = train16v2/genrlv2/refdescv2 (P=5),
#     wave 2 = heldoutv2 (P=5) — heldoutv2 needs its tag's train16v2 JSON
#     as --sig-file, hence the wave barrier
#   - P=5 parallel workers per partition (rollouts are CPU-bound; 208-core box,
#     eval VRAM <2GB/process, C4 training runs on the same cards — bursts only)
# Partition A: anchor_off s1-5 + shuffle s11-13 + m1 genrl + m4ref refdesc
# Partition B: diayn s1-5 + mixon s1-5
# Usage (setsid, append-mode log to preserve watchdog history):
#   setsid bash -c 'cd ~/autodl-tmp/genrl && export CUDA_VISIBLE_DEVICES=0 \
#     PARTITION=A && bash experiments/reeval_all.sh' \
#     </dev/null >> logs/phase_A_reeval.log 2>&1 &
set -u
cd ~/autodl-tmp/genrl
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate genrl
export WANDB_MODE=disabled MKL_SERVICE_FORCE_INTEL=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=.:${PYTHONPATH:-}

PART=${PARTITION:-A}
RES=exp/main_results
EVAL="python experiments/eval_anchors_v2.py"
EVAL1="python experiments/eval_anchors.py"          # idrecover (unchanged, verified)
SHUFFLE_PERM="5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4"  # perm[i]=(i+5)%16
LOGF="logs/phase_${PART}_reeval.log"
exec >> "$LOGF" 2>&1

ts() { date '+%H:%M:%S'; }
flag() { touch "exp/main/${1}_reeval_DONE.flag"; }

run_train16() {  # run_train16 <tag> [perm]
  local tag=$1 perm=${2:-}
  local donef="exp/main/${tag}_reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] train16v2 skip (flag) $(ts)"; return 0; fi
  local snap="exp/main/${tag}/last_snapshot.pt"
  if [ -n "$perm" ]; then
    $EVAL train16v2 "$snap" "$RES/${tag}_train16v2.json" --perm "$perm" \
      || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts) — NO flag"; return 1; }
  else
    $EVAL train16v2 "$snap" "$RES/${tag}_train16v2.json" \
      || { echo "=== [$tag] train16v2 FAILED rc=$? $(ts) — NO flag"; return 1; }
  fi
  $EVAL1 idrecover "$RES/${tag}_train16v2.json.feats.npz" \
                   "$RES/${tag}_idrec_v2.json" \
    || { echo "=== [$tag] idrecover FAILED rc=$? $(ts) — NO flag"; return 1; }
  # C7: per-episode data lives in the JSONs now; the npz is classifier-only.
  # Keep it for the audit-critical anchor_off runs, drop elsewhere (disk).
  case "$tag" in m2_anchor_off_*) ;; *) rm -f "$RES/${tag}_train16v2.json.feats.npz" ;; esac
  flag "$tag"; echo "=== [$tag] train16v2+idrec done $(ts)"
}

run_heldout() {  # run_heldout <tag>
  local tag=$1
  local donef="exp/main/${tag}_heldout_reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] heldoutv2 skip (flag) $(ts)"; return 0; fi
  local snap="exp/main/${tag}/last_snapshot.pt"
  $EVAL heldoutv2 "$snap" "$RES/${tag}_heldoutv2.json" \
        --sig-file "$RES/${tag}_train16v2.json" \
    || { echo "=== [$tag] heldoutv2 FAILED rc=$? $(ts) — NO flag"; return 1; }
  flag "${tag}_heldout"; echo "=== [$tag] heldoutv2 done $(ts)"
}

run_genrl() {  # run_genrl <tag>
  local tag=$1
  local donef="exp/main/${tag}_reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] genrlv2 skip (flag) $(ts)"; return 0; fi
  $EVAL genrlv2 "exp/main/${tag}/last_snapshot.pt" "$RES/${tag}_genrlv2.json" \
    || { echo "=== [$tag] genrlv2 FAILED rc=$? $(ts) — NO flag"; return 1; }
  flag "$tag"; echo "=== [$tag] genrlv2 done $(ts)"
}

run_refdesc() {  # run_refdesc <tag> <q>
  local tag=$1 q=$2
  local donef="exp/main/${tag}_reeval_DONE.flag"
  if [ -f "$donef" ]; then echo "=== [$tag] refdescv2 q=$q skip (flag) $(ts)"; return 0; fi
  $EVAL refdescv2 "exp/main/${tag}/last_snapshot.pt" \
        "$RES/${tag}_refdescv2.json" --q "$q" \
    || { echo "=== [$tag] refdescv2 q=$q FAILED rc=$? $(ts) — NO flag"; return 1; }
  flag "$tag"; echo "=== [$tag] refdescv2 q=$q done $(ts)"
}

dispatch() {  # dispatch "kind:tag[:arg]" — colon-free tag names only
  IFS=: read -r kind tag arg <<< "$1"
  case "$kind" in
    train16p) run_train16 "$tag" "$SHUFFLE_PERM" ;;
    train16)  run_train16 "$tag" ;;
    held)     run_heldout "$tag" ;;
    genrl)    run_genrl "$tag" ;;
    refdesc)  run_refdesc "$tag" "$arg" ;;
    *) echo "unknown job: $1" >&2; return 2 ;;
  esac
}
export -f dispatch run_train16 run_heldout run_genrl run_refdesc ts flag
export PART RES EVAL EVAL1 SHUFFLE_PERM

# Job lists are ALWAYS the full partition set — done runs skip instantly via flags,
# so the script is safe to relaunch after a mid-run kill.
if [ "$PART" = A ]; then
  WAVE1=(
    "train16:m2_anchor_off_s1" "train16:m2_anchor_off_s2" "train16:m2_anchor_off_s3"
    "train16:m2_anchor_off_s4" "train16:m2_anchor_off_s5"
    "train16p:m2_shuffle_s11" "train16p:m2_shuffle_s12" "train16p:m2_shuffle_s13"
    # NOTE (2026-09-02): m1 genrl jobs REMOVED — exp/main/m1_genrl_s{1..3}/
    # last_snapshot.pt were lost in the old machine's takeover (clone preflight
    # accepted: original m1_genrl_s*_genrl.json v1 JSONs retain per-episode
    # rewards). Re-add + retrain (~9-10 GPU-h) only if a v2 baseline re-eval is
    # ever decided.
    "refdesc:m4ref_crouching_down:0" "refdesc:m4ref_jumping_in_place:1"
    "refdesc:m4ref_walking_with_arms_swinging:5"
  )
  WAVE2=(
    "held:m2_anchor_off_s1" "held:m2_anchor_off_s2" "held:m2_anchor_off_s3"
    "held:m2_anchor_off_s4" "held:m2_anchor_off_s5"
  )
elif [ "$PART" = B ]; then
  WAVE1=(
    "train16:m2_diayn_s1" "train16:m2_diayn_s2" "train16:m2_diayn_s3"
    "train16:m2_diayn_s4" "train16:m2_diayn_s5"
    "train16:m3_mixon_s1" "train16:m3_mixon_s2" "train16:m3_mixon_s3"
    "train16:m3_mixon_s4" "train16:m3_mixon_s5"
  )
  WAVE2=(
    "held:m3_mixon_s1" "held:m3_mixon_s2" "held:m3_mixon_s3"
    "held:m3_mixon_s4" "held:m3_mixon_s5"
  )
else
  echo "unknown PARTITION=$PART" >&2; exit 2
fi

echo "### REEVAL PART=$PART parallel edition start $(ts) (wave1=${#WAVE1[@]} wave2=${#WAVE2[@]})"
printf '%s\n' "${WAVE1[@]}" | xargs -P 5 -I{} bash -c 'dispatch "$1"' _ {}
echo "### REEVAL PART=$PART wave1 done $(ts)"
printf '%s\n' "${WAVE2[@]}" | xargs -P 5 -I{} bash -c 'dispatch "$1"' _ {}
echo "### REEVAL PART=$PART ALL DONE $(ts) ###"
echo "PARTITION_${PART}_EXIT=0"
