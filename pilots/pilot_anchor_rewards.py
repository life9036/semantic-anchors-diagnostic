"""Pilot scratch module: semantic-anchor reward variants for GenRL.

Registered into the `agent.genrl` module namespace by a 5-line hook in train.py
(see pilots/README.md) so `agent.imag_reward_fn=anchor_reward` resolves via the
existing globals() dispatch. This file is pilot scratch — delete after the
pilot phase closes; the real implementation goes into the main method.

Reward variants:
  - anchor_reward:       deterministic assignment to ONE text anchor:
                         r = cos(conv_proj(s), target_anchor_seq) [+ beta * H(pi)]
                         (no discriminator / MI machinery)
  - entropy_reward:      MI-only proxy: r = beta * H(pi(.|s)) — coverage without
                         any semantic target (DIAYN-family entropy term)
  - delta_anchor_reward: verb-delta anchor: r = cos(s, anchor_pos) - cos(s, anchor_neg)
                         ("what changed", not "what it looks like")
"""
import torch
import torch.nn.functional as F

PILOT_ANCHORS = {
    'walk':      'robot walk fast clean',
    'run':       'robot run fast clean',
    'stand':     'standing up',
    'backwards': 'robot walking backwards',
    'crouch':    'robot crouching low',
    'jump':      'robot jumping up and down',
    'slow':      'robot walking slowly',
    'flip':      'doing flips',
}


def _get_clip(agent):
    wm = agent.wm
    if hasattr(wm, 'viclip_model') and wm.viclip_model is not None:
        return wm.viclip_model
    from tools.genrl_utils import viclip_global_instance
    if not viclip_global_instance._instantiated:
        viclip_global_instance.instantiate()
    return viclip_global_instance.viclip


def _anchor_targets(agent, B, T, anchors):
    """Per-anchor latent target sequences via the frozen connector (cached)."""
    if not hasattr(agent, 'pilot_anchor_targets'):
        wm = agent.wm
        clip = _get_clip(agent)
        targets = {}
        with torch.no_grad():
            for key, prompt in anchors.items():
                fe = clip.get_txt_feat(prompt).to(agent.device)
                ve = fe.reshape(1, 1, -1).repeat(B, T, 1)
                stats = wm.connector.video_imagine(
                    ve, dreamer_init=None, sample=False,
                    reset_every_n_frames=False, denoise=True)
                targets[key] = {
                    k: v.permute([1, 0] + list(range(2, len(v.shape))))
                    for k, v in stats.items()}
        agent.pilot_anchor_targets = targets
    return agent.pilot_anchor_targets


def _conv_stoch(agent, seq):
    stoch = agent.wm.rssm.get_stoch(seq)
    return agent.wm.heads['decoder']._conv_in[0](stoch)


def anchor_cos_scores(agent, seq, anchors):
    """dict anchor_key -> [T, B] cosine score between trajectory and anchor target."""
    T, B = (seq['deter'] if 'deter' in seq else seq['stoch']).shape[:2]
    targets = _anchor_targets(agent, B, T, anchors)
    agent_conv = _conv_stoch(agent, seq)
    out = {}
    for key, tseq in targets.items():
        tstoch = agent.wm.connector.get_stoch(tseq)
        tconv = agent.wm.heads['decoder']._conv_in[0](tstoch)
        out[key] = F.cosine_similarity(tconv, agent_conv, dim=-1)
    return out


def _policy_entropy(agent, seq):
    """H(pi(.|s)) of the imagination actor — the brief's MI-proxy term."""
    actor = agent._imag_behavior.actor
    feat = torch.cat([seq['deter'], agent.wm.rssm.get_stoch(seq)], -1)
    if hasattr(actor, 'get_dist'):
        dist = actor.get_dist(feat)
    else:
        dist = actor(feat)
    if hasattr(dist, 'entropy'):
        return dist.entropy()
    raise RuntimeError('actor dist API not supported by this codebase revision')


def anchor_reward(agent, seq, anchor='walk', beta=0.0, **kwargs):
    """Deterministic anchor assignment; NO discriminator/MI machinery."""
    scores = anchor_cos_scores(agent, seq, {anchor: PILOT_ANCHORS[anchor]})
    r = scores[anchor]
    if beta > 0.0:
        with torch.no_grad():
            r = r + beta * _policy_entropy(agent, seq)
    return r.unsqueeze(-1)


def entropy_reward(agent, seq, beta=0.1, **kwargs):
    """MI-only proxy: pure policy-entropy maximization (no semantic target)."""
    with torch.no_grad():
        r = beta * _policy_entropy(agent, seq)
    return r.unsqueeze(-1)


def delta_anchor_reward(agent, seq, pos='run', neg='slow', beta=0.0, **kwargs):
    """Verb-delta anchor: cos(s, pos) - cos(s, neg) — change, not appearance."""
    scores = anchor_cos_scores(
        agent, seq, {pos: PILOT_ANCHORS[pos], neg: PILOT_ANCHORS[neg]})
    r = scores[pos] - scores[neg]
    if beta > 0.0:
        with torch.no_grad():
            r = r + beta * _policy_entropy(agent, seq)
    return r.unsqueeze(-1)
