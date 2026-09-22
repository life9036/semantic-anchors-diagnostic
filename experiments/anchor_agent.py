"""AnchorGenRLAgent: the fresh-instantiation entry point (agent=genrl_anchor).

IMPORTANT: train.py load_snapshot() replaces the workspace agent with the
object stored in the pretrained WM snapshot (a plain GenRLAgent), so the
REAL mechanism is experiments.anchor_patch.install() — class-level method
patches on GenRLAgent (pickle-clean, no instance-bound methods) applied at
import time by train.py and eval_anchors.py. This class only exists so that
hydra's agent=genrl_anchor has a _target_ that initializes the anchor
attributes on a fresh agent.
"""
from agent.genrl import GenRLAgent

from experiments import anchor_patch


class AnchorGenRLAgent(GenRLAgent):
    """Shared c-conditioned policy trained by neighborhood-mixture
    imagination. Methods come from anchor_patch.install() (class level)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # creates wm + connector (+ imag behavior)
        anchor_patch.patch_agent(self)   # instance attrs + cond imag behavior
