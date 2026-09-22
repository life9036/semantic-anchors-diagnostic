"""Build the Cheetah anchor library (B5 cross-domain confirmation subset).

One-shot, zero training: loads the retained Cheetah snapshot, runs the
connector video_imagine pipeline for the same 24 prompts (16 train + 8
held-out), and caches exp/anchor_lib/cheetah_lib.pt.

Usage:  CUDA_VISIBLE_DEVICES=<gpu> CHEETAH=1 python experiments/build_cheetah_lib.py
"""
import os
import sys

REPO = os.path.expanduser('~/autodl-tmp/genrl')
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('CHEETAH', '1')

import torch  # noqa: E402

from experiments.anchor_lib import build_library  # noqa: E402
from experiments.eval_anchors_v2 import load_agent  # noqa: E402

SNAP = 'models/genrl_cheetah_500k_2.pt'
OUT = 'exp/anchor_lib/cheetah_lib.pt'

# Official HF snapshots store the bare agent object (not a dict payload);
# load_agent now handles both.
agent = load_agent(SNAP)
lib = build_library(agent, lib_path=OUT)
print('CHEETAH LIB BUILT ->', OUT)
print({k: (tuple(v.shape) if isinstance(v, torch.Tensor) else v)
       for k, v in lib.items() if k in ('S', 'm', 'c', 'c_rand', 'c_text',
                                        'held_S', 'held_c', 'edges',
                                        'g_dim', 'train_prompts')})
