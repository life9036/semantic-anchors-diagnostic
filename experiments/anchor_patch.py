"""Anchor-conditioned training as CLASS-LEVEL patches on GenRLAgent.

Why class-level (2026-08-31 lesson): instance-bound MethodType attributes
survive torch.save but FAIL torch.load — unpickling resolves the bound
method on the partially-constructed agent and nn.Module.__getattr__ raises.
Class-level function assignment pickles nothing on the instance; the class
reference resolves to the installed patched class in any process that
imports anchor_patch.install(). The cond sampling uses a registered buffer
instead of a bound-method sampler for the same reason.

Every patched function delegates to the ORIGINAL GenRLAgent implementation
when cfg has no 'anchor' section, so M1 baselines (plain genrl agent) are
completely unaffected by install().
"""
import numpy as np
import torch

from agent.dreamer import stop_gradient
import agent.dreamer_utils as common

from experiments.anchor_lib import (build_library, sample_conditions,
                                    anchor_cond_reward)
from experiments.anchor_models import CondActorCritic

DEFAULT_DIAYN_OPT = dict(opt='adam', lr=3e-5, eps=1e-5, clip=100, wd=1e-6)

_ORIG = {}
_installed = False


def install():
    """Assign the patched methods onto GenRLAgent (idempotent)."""
    global _installed
    if _installed:
        return
    from agent.genrl import GenRLAgent
    _ORIG['act'] = GenRLAgent.act
    _ORIG['update_imag_behavior'] = GenRLAgent.update_imag_behavior
    _ORIG['instantiate_imag_behavior'] = GenRLAgent.instantiate_imag_behavior
    GenRLAgent.act = act
    GenRLAgent.update_imag_behavior = update_imag_behavior
    GenRLAgent.instantiate_imag_behavior = instantiate_imag_behavior
    GenRLAgent._anchor_cfg = _anchor_cfg
    GenRLAgent._ensure_lib = _ensure_lib
    GenRLAgent._move_lib_to_device = _move_lib_to_device
    _installed = True


def _is_anchor(self):
    return bool(getattr(self.cfg, 'anchor', None))


def _anchor_cfg(self):
    return getattr(self, 'anchor_cfg', None) or getattr(self.cfg, 'anchor', {})


def _ensure_lib(self):
    # safe access: during GenRLAgent.__init__ (class-level patched method is
    # dispatched before patch_agent ran) the attribute does not exist yet
    if getattr(self, 'lib', None) is None:
        lib_path = _anchor_cfg(self).get('lib_path', None) or None
        self.lib = build_library(self, lib_path)
    return self.lib


def _move_lib_to_device(self, lib):
    return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in lib.items()}


def instantiate_imag_behavior(self):
    if not _is_anchor(self):
        return _ORIG['instantiate_imag_behavior'](self)
    # pure-anchor conditions as a registered buffer (pickle-clean: no bound
    # methods, no agent back-references in the behavior)
    lib = _move_lib_to_device(self, _ensure_lib(self))
    cmode = _anchor_cfg(self).get('cond_mode', 'anchor')
    cond_key = {'anchor': 'c', 'random': 'c_rand', 'text': 'c_text'}[cmode]
    conds = lib[cond_key]                         # [K, d] normalized
    self._imag_behavior = CondActorCritic(
        self.cfg, self.act_spec, self.wm.inp_size,
        self.wm.heads['decoder']._conv_in[0].out_features,
        name='imag',
        conds=conds,
        diayn=_anchor_cfg(self).get('diayn', False),
        K=_anchor_cfg(self).get('k_train', 16),
        beta_mi=_anchor_cfg(self).get('beta_mi', 1.0),
        diayn_opt=getattr(self.cfg, 'diayn_opt', None) or DEFAULT_DIAYN_OPT,
    ).to(self.device)
    self._imag_behavior.rewnorm = common.StreamNorm(
        **self.cfg.imag_reward_norm, device=self.device)


def update_imag_behavior(self, state=None, outputs=None, metrics={},
                         seq_data=None):
    """Anchor-conditioned imagination update (replaces GenRLAgent's)."""
    if not _is_anchor(self):
        return _ORIG['update_imag_behavior'](self, state, outputs, metrics,
                                             seq_data)
    if getattr(self.cfg, 'imag_reward_fn', None) is None:
        return outputs['post'], metrics
    if outputs is not None:
        post = outputs['post']
        is_terminal = outputs['is_terminal']
    else:
        seq_data = self.wm.preprocess(seq_data)
        embed = self.wm.encoder(seq_data)
        post, _ = self.wm.rssm.observe(
            embed, seq_data['action'], seq_data['is_first'])
        is_terminal = seq_data['is_terminal']
    start = {k: stop_gradient(v) for k, v in post.items()}

    lib = _move_lib_to_device(self, _ensure_lib(self))
    # WorldModel.imagine flattens the start states to batch*length items;
    # conditions must be sampled per flattened start (not per batch row).
    N = start['deter'].reshape(-1, start['deter'].shape[-1]).shape[0]
    # DIAYN requires a pure anchor label per item (mixup-off by spec)
    pure_prob = 1.0 if _anchor_cfg(self).get('diayn', False) \
        else _anchor_cfg(self).get('pure_prob', 0.5)
    conds, targets, ids = sample_conditions(
        lib, N,
        pure_prob=pure_prob,
        shuffle=_anchor_cfg(self).get('shuffle', False),
        cond_mode=_anchor_cfg(self).get('cond_mode', 'anchor'),
        device=self.device)
    targets = targets.detach()

    def reward_fn(seq):
        return anchor_cond_reward(
            self, seq, targets, window=_anchor_cfg(self).get('window', 2))

    metrics.update(self._imag_behavior.update(
        self.wm, start, is_terminal, reward_fn, conds, targets, ids))
    return start, metrics


def act(self, obs, meta, step, eval_mode, state, cond=None):
    """Cond-aware act: cond [d] numpy/tensor, constant per episode."""
    if not _is_anchor(self):
        return _ORIG['act'](self, obs, meta, step, eval_mode, state)
    if cond is None:
        return _ORIG['act'](self, obs, meta, step, eval_mode, state)
    if self.cfg.only_random_actions:
        return (np.random.uniform(-1, 1, self.act_dim)
                .astype(self.act_spec.dtype), (None, None))
    obs = {k: torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0)
           for k, v in obs.items()}
    if state is None:
        latent = self.wm.rssm.initial(len(obs['reward']))
        action = torch.zeros(
            (len(obs['reward']),) + self.act_spec.shape, device=self.device)
    else:
        latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], should_sample)
    feat = self.wm.rssm.get_feat(latent)
    c = torch.as_tensor(cond, device=self.device).reshape(1, -1).float()
    feat = torch.cat([feat, c], -1)
    actor = self._acting_behavior.actor(feat)
    if eval_mode:
        try:
            action = actor.mean
        except AttributeError:
            action = actor._mean
    else:
        action = actor.sample()
    new_state = (latent, action)
    return action.cpu().numpy()[0], new_state


def patch_agent(agent):
    """Attach anchor instance ATTRIBUTES (no bound methods) and rebuild the
    imag behavior. Class-level methods come from install(). Idempotent."""
    install()
    agent.anchor_cfg = dict(getattr(agent.cfg, 'anchor', {}))
    agent.K = agent.anchor_cfg.get('k_train', 16)
    agent.anchor_dim = agent.wm.heads['decoder']._conv_in[0].out_features
    agent.lib = None
    # config validation (pre-registered variants)
    diayn = agent.anchor_cfg.get('diayn', False)
    shuffle = agent.anchor_cfg.get('shuffle', False)
    pure = agent.anchor_cfg.get('pure_prob', 0.5)
    cmode = agent.anchor_cfg.get('cond_mode', 'anchor')
    if cmode not in ('anchor', 'random', 'text'):
        raise ValueError(f'unknown cond_mode {cmode}')
    if diayn and pure != 1.0:
        raise ValueError('DIAYN variant requires pure_prob=1.0 (mixup-off)')
    if shuffle and pure != 1.0:
        raise ValueError('shuffle control requires pure_prob=1.0 (mixup-off)')
    if shuffle and cmode not in ('anchor', 'random'):
        raise ValueError('shuffle control defined for cond_mode in '
                         '{anchor, random} (M5 permuted-target cell: random)')
    if getattr(agent.cfg, 'imag_reward_fn', None) is not None:
        agent.instantiate_imag_behavior()
    return agent
