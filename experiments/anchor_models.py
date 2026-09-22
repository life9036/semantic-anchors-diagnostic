"""Model classes for the anchor-conditioned policy (no agent import — keeps
the module graph acyclic: anchor_patch -> anchor_models, anchor_agent ->
anchor_patch + anchor_models)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.dreamer import ActorCritic, stop_gradient
import agent.dreamer_utils as common


class CondActorMLP(nn.Module):
    """Wraps the actor MLP so cond-free callers (train.py warmup, plain eval)
    get a randomly sampled pure-anchor condition appended; inputs that already
    carry a condition (dim > feat_size) pass through unchanged.

    The anchor conditions live in a registered buffer [K, d] — NO bound
    methods or agent back-references, so snapshots pickle cleanly."""

    def __init__(self, mlp, feat_size, conds=None):
        super().__init__()
        self.mlp = mlp
        self.feat_size = feat_size
        if conds is not None:
            self.register_buffer('conds', conds)

    def forward(self, x):
        if x.shape[-1] == self.feat_size:
            idx = torch.randint(0, self.conds.shape[0],
                                (x.shape[0],), device=x.device)
            x = torch.cat([x, self.conds[idx]], dim=-1)
        return self.mlp(x)


class DIAYNDiscriminator(nn.Module):
    """q_psi(i|s_t): K-class MLP on the frozen latent state (deter+stoch,
    same input as the actor but WITHOUT the condition). Pre-registered spec:
    3 hidden layers x 512, SiLU, LayerNorm, output K logits."""

    def __init__(self, feat_size, K=16, layers=3, units=512):
        super().__init__()
        net = []
        last = feat_size
        for _ in range(layers):
            net += [nn.Linear(last, units, bias=False), nn.LayerNorm(units),
                    nn.SiLU()]
            last = units
        net += [nn.Linear(last, K)]
        self.net = nn.Sequential(*net)

    def forward(self, feat):
        return self.net(feat)


class CondActorCritic(ActorCritic):
    """ActorCritic with a c-conditioned actor and critic (input feat+c)."""

    def __init__(self, config, act_spec, feat_size, cond_size, name='',
                 conds=None, diayn=False, K=16, beta_mi=1.0,
                 diayn_opt=None):
        super().__init__(config, act_spec, feat_size + cond_size, name)
        self.feat_size = feat_size
        self.cond_size = cond_size
        self.diayn_enabled = diayn
        self.K = K
        self.beta_mi = beta_mi
        # wrap actor for cond-free callers (train.py warmup uses no task_cond)
        self.actor = CondActorMLP(self.actor, feat_size, conds)
        if diayn:
            self.diayn = DIAYNDiscriminator(feat_size, K=K).to(self.device)
            self.diayn_opt = common.Optimizer(
                'diayn', self.diayn.parameters(),
                **diayn_opt if diayn_opt else dict(
                    opt='adam', lr=3e-5, eps=1e-5, clip=100, wd=1e-6),
                use_amp=self._use_amp)
            self._diayn_updates = 0

    def _cat(self, seq, feat_key='feat'):
        """seq dict with the condition appended to the named feature."""
        out = dict(seq)
        out[feat_key] = torch.cat([seq[feat_key], seq['task']], dim=-1)
        return out

    def update(self, world_model, start, is_terminal, reward_fn, conds,
               targets, ids):
        """cond-aware version of ActorCritic.update. conds [B,d], targets
        [B,T,d] (normalized, caller-detached), ids [B] (pure anchor id / -1)."""
        metrics = {}
        hor = self.cfg.imag_horizon
        with common.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = world_model.imagine(
                    self.actor, start, is_terminal, hor, task_cond=conds)
                reward = reward_fn(seq)
                if self.diayn_enabled:
                    r_mi = self._mi_reward(world_model, seq, ids)
                    reward = reward + self.beta_mi * r_mi.unsqueeze(-1)
                    metrics['diayn_mi_reward'] = r_mi.mean()
                seq['reward'], mets1 = self.rewnorm(reward)
                mets1 = {f'reward_{k}': v for k, v in mets1.items()}
                target, mets2, baseline = self.target(seq)
                actor_loss, mets3 = self.actor_loss(seq, target, baseline)
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
        if self.diayn_enabled:
            self._diayn_updates += 1
            if self._diayn_updates % 4 == 0:
                metrics.update(self._update_diayn(world_model, seq, ids))
        with common.RequiresGrad(self.critic):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                seq = {k: stop_gradient(v) for k, v in seq.items()}
                critic_loss, mets4 = self.critic_loss(seq, target)
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()
        return {f'{self.name}_{k}'.strip('_') : v for k, v in metrics.items()}

    def _mi_reward(self, world_model, seq, ids):
        """r_MI = log q(i|s) - log(1/K), gradient flows into the policy
        (standard DIAYN); states fed to q are NOT detached here."""
        feat = torch.cat([seq['deter'], world_model.rssm.get_stoch(seq)], -1)
        logq = torch.log_softmax(self.diayn(feat), -1)       # [T+1, B, K]
        ids_ts = ids[None].repeat(feat.shape[0], 1)          # [T+1, B]
        r = logq.gather(-1, ids_ts[..., None]).squeeze(-1) - np.log(1.0 / self.K)
        return r                                             # [T+1, B]

    def _update_diayn(self, world_model, seq, ids):
        """CE on the same batch every 4 actor steps, fully detached states."""
        with common.RequiresGrad(self.diayn):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                # detached inputs; clear grads left by the r_MI flow-through
                # (actor backward) so the CE update is pure
                self.diayn.zero_grad(set_to_none=True)
                feat = torch.cat(
                    [seq['deter'], world_model.rssm.get_stoch(seq)], -1
                ).detach()
                logits = self.diayn(feat)                    # [T+1, B, K]
                ce = F.cross_entropy(
                    logits.reshape(-1, self.K),
                    ids.repeat(feat.shape[0], 1).reshape(-1))
            m = self.diayn_opt(ce, self.diayn.parameters())
            return {f'diayn_{k}': v for k, v in m.items()}

    def actor_loss(self, seq, target, baseline):
        return super().actor_loss(self._cat(seq), target, baseline)

    def critic_loss(self, seq, target):
        return super().critic_loss(self._cat(seq), target)

    def target(self, seq):
        return super().target(self._cat(seq))
