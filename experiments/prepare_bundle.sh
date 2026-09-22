#!/usr/bin/env bash
# Runs ON THE OLD BOX: bundle everything the new dual-4090 box needs.
# Usage:
#   bash experiments/prepare_bundle.sh            # creates ~/genrl_bundle.tar
#   NEW_BOX='-p PORT root@HOST' bash experiments/prepare_bundle.sh   # direct push
set -euo pipefail

BUNDLE=~/genrl_bundle.tar
# paths the new box needs (hardcoded bert cache path in tools/genrl_utils.py!)
PATHS=(
  ~/autodl-tmp/genrl
  ~/miniconda3/envs/genrl
  ~/.cache/huggingface/hub/models--bert-large-uncased
)

# prune transient/partial training state from the bundle's genrl copy
echo "=== pruning transient dirs from exp ==="
cd ~/autodl-tmp/genrl
rm -rf exp/main/m1_genrl_s* exp/main/m2_* exp/main/m3_* exp/smoke logs/m1_* logs/m2_* logs/m3_* logs/smoke_* 2>/dev/null || true
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== creating bundle ($BUNDLE) ==="
tar cf "$BUNDLE" -C / \
  --exclude="*.log" \
  $(for p in "${PATHS[@]}"; do echo "${p#/}"; done)
ls -la "$BUNDLE"

if [ -n "${NEW_BOX:-}" ]; then
  echo "=== pushing to new box: $NEW_BOX ==="
  # shellcheck disable=SC2086
  ssh $NEW_BOX "mkdir -p /tmp/bundle_in && cd /tmp/bundle_in" || true
  # shellcheck disable=SC2086
  cat "$BUNDLE" | ssh $NEW_BOX "cd / && tar xf - -C /"
  echo "=== push done ==="
fi
echo "BUNDLE READY"
