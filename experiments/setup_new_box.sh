#!/usr/bin/env bash
# Runs ON THE NEW BOX after the bundle is unpacked: validate the copied
# environment end-to-end (imports + GPU + a tiny anchor training run).
set -euo pipefail
export HYDRA_FULL_ERROR=1
cd ~/autodl-tmp/genrl
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate genrl
echo "=== python/torch ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== imports ==="
PYTHONPATH=.: python -c "import experiments.anchor_agent, experiments.eval_anchors, experiments.preflight; print('IMPORT_OK')"
echo "=== anchor lib present ==="
ls -la exp/anchor_lib/walker_lib.pt models/genrl_walker_500k_2.pt
echo "=== bert cache ==="
ls -d ~/.cache/huggingface/hub/models--bert-large-uncased && \
  ls ~/.cache/huggingface/hub/models--bert-large-uncased/snapshots/ | head -2
echo "=== smoke: tiny anchor run ==="
WANDB_MODE=disabled PYTHONPATH=.: python train.py task=walker_walk \
  snapshot_load_dir="$HOME/autodl-tmp/genrl/models/genrl_walker_500k_2.pt" \
  num_train_frames=210 batch_size=32 batch_length=32 \
  use_wandb=false use_tb=false \
  eval_every_frames=999999 \
  save_every_frames=1000 snapshot_every_frames=25_000 \
  agent=genrl_anchor train_from_data=False \
  agent.anchor.lib_path="$HOME/autodl-tmp/genrl/exp/anchor_lib/walker_lib.pt" \
  agent.anchor.pure_prob=1.0 seed=99 \
  "hydra.run.dir=$HOME/autodl-tmp/genrl/exp/smoke_newbox" \
  "snapshot_dir=$HOME/autodl-tmp/genrl/exp/smoke_newbox" \
  > logs/setup_smoke.log 2>&1
echo "=== smoke rc=$? (expect 0) ==="
tail -5 logs/setup_smoke.log
echo "NEW BOX READY"
