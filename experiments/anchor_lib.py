"""Anchor library: frozen VLM+connector prompt -> coupled anchor pairs (S_i, m_i).

Space (pre-registered, pilot P1/P2 validated): g-space = the decoder's frozen
`_conv_in[0]` projection applied to RSSM stoch samples — the same shared space
GenRL's cosine reward (and the pilot anchor rewards) are computed in.

  S_i in R^{T x d}: per-step L2-normalized connector-imagined trajectory
                   (T = imag_horizon = 16, phase-locked)
  m_i in R^d:      temporal mean of S_i (unnormalized)
  c_i = normalize(m_i)

Consistent mixing contract: a condition mixing c = normalize(Σ w_j m_j) is
paired with a reward target trajectory S(phi) = normalize(Σ w_j S_j(phi)) using
the SAME weights — mixing happens identically on both sides of the coupling.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.prompts import TRAIN_PROMPTS, HELDOUT_PROMPTS, SHUFFLE_PERM

T = 16  # imag horizon, phase-locked with the imagination rollout


def g_proj(wm, stoch):
    """Frozen projection stoch -> reward space (decoder conv_in head)."""
    return wm.heads['decoder']._conv_in[0](stoch)


def get_clip(agent):
    """ViCLIP handle: prefer the WM-attached model, fall back to the global
    singleton (same pattern as the pilots' pilot_anchor_rewards)."""
    wm = agent.wm
    if hasattr(wm, 'viclip_model') and wm.viclip_model is not None:
        return wm.viclip_model
    from tools.genrl_utils import viclip_global_instance
    if not viclip_global_instance._instantiated:
        viclip_global_instance.instantiate()
    return viclip_global_instance.viclip


def _prompt_trajectory(agent, prompt):
    """Connector-imagined trajectory for one prompt, in g-space.

    Returns S in R^{T x d} (per-step L2-normalized) and m = mean_t S (raw)."""
    wm = agent.wm
    clip = get_clip(agent)
    dev = agent.device
    with torch.no_grad():
        fe = clip.get_txt_feat(prompt).to(dev)       # [1, 512]
        ve = fe.reshape(1, 1, -1).repeat(1, T, 1)    # [1, T, 512]
        stats = wm.connector.video_imagine(
            ve, dreamer_init=None, sample=False,
            reset_every_n_frames=False, denoise=True)
        stoch = wm.connector.get_stoch(stats)        # [1, T, 32, 32]
        S = g_proj(wm, stoch)[0]                     # [T, d]
        S = F.normalize(S, dim=-1)
    m = S.mean(0)                                    # unnormalized mean
    return S, m


def build_library(agent, lib_path=None):
    """Build the full anchor library once (zero training). Cached on disk so
    every downstream run loads bit-identical anchors.

    Returns dict:
      train_prompts, heldout_prompts,
      S  [K, T, d]  per-step normalized trajectories
      m  [K, d]     unnormalized temporal means
      c  [K, d]     normalized means (the pure conditions)
      c_rand [K, d] fixed seeded random unit codes (C4 baseline, no semantics)
      c_text [K, d] fixed-projection VLM text features (C4: semantics, no
                    trajectory pooling); held_c_text for held-out prompts
      edges         frozen neighborhood graph: symmetric top-2 edges
      held_S/held_m/held_c for the held-out prompts
      shuffle_perm, g_dim
    """
    if lib_path is not None and os.path.exists(lib_path):
        lib = torch.load(lib_path, map_location='cpu')
        assert lib['train_prompts'] == TRAIN_PROMPTS
        assert lib['heldout_prompts'] == HELDOUT_PROMPTS
        if 'c_rand' not in lib:      # C4 addition to a pre-existing cache
            lib = _ensure_cond_modes(agent, lib)
            if lib_path is not None:
                torch.save(lib, lib_path)
        return lib

    wm = agent.wm
    K = len(TRAIN_PROMPTS)
    S, m = [], []
    for p in TRAIN_PROMPTS:
        Si, mi = _prompt_trajectory(agent, p)
        S.append(Si)
        m.append(mi)
    S = torch.stack(S)   # [K, T, d]
    m = torch.stack(m)   # [K, d]
    c = F.normalize(m, dim=-1)

    # Frozen neighborhood graph on TRAIN anchors: per-anchor 2 nearest
    # neighbors (cos on c, excluding self via masked diagonal), symmetrized.
    sim = c @ c.T
    sim = sim.masked_fill(
        torch.eye(K, dtype=torch.bool, device=c.device), -1e9)
    edges = set()
    for i in range(K):
        _, top = sim[i].topk(2)
        for j in top.tolist():
            edges.add(tuple(sorted((i, j))))
    edges = sorted(edges)

    hS, hm = [], []
    for p in HELDOUT_PROMPTS:
        Si, mi = _prompt_trajectory(agent, p)
        hS.append(Si)
        hm.append(mi)
    hS = torch.stack(hS)
    hm = torch.stack(hm)
    hc = F.normalize(hm, dim=-1)

    lib = dict(
        train_prompts=list(TRAIN_PROMPTS),
        heldout_prompts=list(HELDOUT_PROMPTS),
        S=S, m=m, c=c,
        edges=[list(e) for e in edges],
        held_S=hS, held_m=hm, held_c=hc,
        shuffle_perm=SHUFFLE_PERM,
        g_dim=S.shape[-1],
    )
    lib = _ensure_cond_modes(agent, lib)
    if lib_path is not None:
        d = os.path.dirname(lib_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(lib, lib_path)
    return lib


def _ensure_cond_modes(agent, lib):
    """C4 baseline condition matrices (deterministic, no training):
      c_rand: fixed seeded random unit codes (no semantic content)
      c_text: fixed random projection of VLM text features (semantics, no
              temporal trajectory pooling)
    """
    d = lib['g_dim']
    dev = lib['c'].device
    gen = torch.Generator(device=dev).manual_seed(0)
    P = torch.randn(d, 512, generator=gen, device=dev) / (512 ** 0.5)

    def text_cond(prompt):
        with torch.no_grad():
            fe = get_clip(agent).get_txt_feat(prompt).to(dev)   # [1, 512]
        return F.normalize(fe @ P.T, dim=-1)                    # [1, d]

    lib['c_rand'] = F.normalize(
        torch.randn(len(lib['train_prompts']), d, generator=gen,
                    device=dev), dim=-1)
    lib['c_text'] = torch.cat(
        [text_cond(p) for p in lib['train_prompts']], 0)
    lib['held_c_text'] = torch.cat(
        [text_cond(p) for p in lib['heldout_prompts']], 0)
    return lib


def sample_conditions(lib, B, pure_prob, shuffle=False, device='cuda',
                      cond_mode='anchor'):
    """Sample one condition per batch item.

    pure_prob: probability of a PURE anchor item (0.5 = mixup-on, 1.0 = off).
    cond_mode (C4 baselines, reward targets are ALWAYS the semantic S):
      'anchor': cond = c_i (coupled anchor)          [main method]
      'random': cond = c_rand_i (fixed random codes, no semantics)
      'text':   cond = c_text_i (VLM text feature via fixed projection,
                semantics WITHOUT temporal trajectory pooling)
    Per item:
      pure:  i ~ U(K);            cond = cond_matrix[i],  target = S_i
      mix:   edge (a,b) ~ U(E);   alpha ~ U[0,1]
             cond = normalize(alpha*cond_a + (1-alpha)*cond_b)
             target_t = normalize(alpha*S_a[t] + (1-alpha)*S_b[t])
    shuffle (control): target = S_perm[i] (pure items only).
    cond_mode='anchor' + shuffle = shuffled-anchor control (semantic
    conditions, permuted targets); cond_mode='random' + shuffle = M5
    permuted-target cell (random codes, permuted targets).

    Returns conds [B,d], targets [B,T,d], ids [B] (pure anchor id, -1 for mix).
    """
    key = {'anchor': 'c', 'random': 'c_rand', 'text': 'c_text'}[cond_mode]
    c = lib[key].to(device)
    m = lib['m'].to(device)
    S = lib['S'].to(device)
    K, Tsteps, d = S.shape
    edges = torch.as_tensor(lib['edges'], device=device)  # [E, 2]
    assert not (shuffle and cond_mode not in ('anchor', 'random')), \
        'shuffle control defined for cond_mode in {anchor, random}'

    is_pure = torch.rand(B, device=device) < pure_prob
    conds = torch.zeros(B, d, device=device)
    targets = torch.zeros(B, Tsteps, d, device=device)
    ids = torch.full((B,), -1, dtype=torch.long, device=device)

    n_pure = int(is_pure.sum())
    if n_pure > 0:
        idx = torch.randint(0, K, (n_pure,), device=device)
        pure_rows = is_pure.nonzero(as_tuple=False).squeeze(-1)
        conds[pure_rows] = c[idx]
        if shuffle:
            perm = torch.as_tensor(lib['shuffle_perm'], device=device)
            targets[pure_rows] = S[perm[idx]]
        else:
            targets[pure_rows] = S[idx]
        ids[pure_rows] = idx

    mix_rows = (~is_pure).nonzero(as_tuple=False).squeeze(-1)
    n_mix = int((~is_pure).sum())
    if n_mix > 0:
        e = torch.randint(0, edges.shape[0], (n_mix,), device=device)
        a, b = edges[e, 0], edges[e, 1]            # [n_mix]
        alpha = torch.rand(n_mix, device=device)   # [n_mix]
        # mix in the CONDITION space of the mode (targets stay semantic)
        conds[mix_rows] = F.normalize(
            alpha[:, None] * c[a] + (1 - alpha[:, None]) * c[b], dim=-1)
        targets[mix_rows] = F.normalize(
            alpha[:, None, None] * S[a] + (1 - alpha[:, None, None]) * S[b],
            dim=-1)
    return conds, targets, ids


def anchor_cond_reward(agent, seq, targets, window=2):
    """r_t = max_{phi in [t-2,t+2] cap [1,T]} cos(g(s_t), target[b, phi]).

    targets [B,T,d] are already per-step L2-normalized (stop-grad enforced by
    the caller). seq is the imagination rollout ([T+1, B, ...] states).
    Computed in fp32 regardless of the caller's autocast state: the reward is
    the training signal, keep its precision deterministic."""
    wm = agent.wm
    stoch = wm.rssm.get_stoch(seq)              # [T+1, B, 32, 32]
    with torch.cuda.amp.autocast(enabled=False):
        gs = g_proj(wm, stoch.float())          # [T+1, B, d]
        gs = F.normalize(gs, dim=-1)
        Tsteps, B, d = gs.shape
        sims = torch.einsum('tbd,bpd->tbp', gs, targets.float())  # [T+1, B, T]
        t_idx = torch.arange(Tsteps, device=gs.device)
        p_idx = torch.arange(1, T + 1, device=gs.device)   # anchor steps 1..16
        mask = (t_idx[:, None] - p_idx[None, :]).abs() <= window
        # -inf (not -1e9): fp16-safe masking
        sims = sims.masked_fill(~mask[:, None, :], float('-inf'))
        r = sims.max(dim=-1).values.unsqueeze(-1)   # [T+1, B, 1]
    return r
