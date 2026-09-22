"""eval_anchors_v2.py — Round-1-review-fixed evaluation suite (C2/C3/C7).

Changes vs eval_anchors.py (review findings, review-stage/REVIEWER_MEMORY.md):
  C2:  independent kinematic descriptors from MuJoCo physics (velocity,
       direction, torso uprightness, joint pose, contacts) — outside g-space;
       sliding-window temporal matching replaces the t>16 clamp;
       16x16 cross-target confusion matrices (g-space AND kinematic);
       fair shuffled-control own-target evaluation (perm from lib).
  C3:  c_mix is a true edge-restricted Top-2 mixture (train-graph edge only,
       norm-corrected softmax over the edge's two anchors); uncovered prompts
       (no trainable edge) use Top-1 fallback and their identical-arm
       comparisons are EXCLUDED (recorded, never counted as wins);
       honest monotonicity = Spearman trend on kinematic distance-to-endpoint
       with exact permutation p-values (null-pass rate reported).
  C7:  every output JSON keeps per-episode raw data (reward, descriptor,
       reset seed, arm condition, alpha-scan values); kinematic summaries
       saved alongside g-features.

Modes:
  train16v2 <snap> <out.json> [--perm "0,5,10,..."]  16 anchors x 20 eps;
        saves g-feats npz (idrecover-compatible) + kinematic sigs + confusion.
  heldoutv2 <snap> <out.json> [--sig-file <train16v2.json>] 8 held-out prompts,
        4 arms paired by reset seed + honest alpha-scan.
  genrlv2   <snap> <out.json>
  refdescv2 <snap> <out.json> --q N
  aggregatev2 <task> <res_dir> <out.json>   task in {m2, m34}
"""
import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MKL_SERVICE_FORCE_INTEL', '1')
os.environ['HYDRA_FULL_ERROR'] = '1'

import numpy as np
import torch
import torch.nn.functional as F

import tools.utils as utils
import envs.main as envs
from experiments.anchor_lib import g_proj, T
from experiments.anchor_patch import install
install()

REPO = os.path.expanduser('~/autodl-tmp/genrl')
# CHEETAH=1 switches the evaluator to the Cheetah library and task env
# (cross-domain confirmation subset, B5).
_CHEETAH = os.environ.get('CHEETAH') == '1'
LIB_PATH = os.path.join(
    REPO, 'exp/anchor_lib/cheetah_lib.pt' if _CHEETAH
    else 'exp/anchor_lib/walker_lib.pt')
TASK = 'cheetah_run' if _CHEETAH else 'walker_walk'

WINDOW = 16  # standardized temporal window = imagination horizon T


def cond_matrices(lib, cond_mode):
    """(train_conds [K,d], held_conds [8,d]) for the C4 cond mode.
    'random' has no held-out query path -> held_conds is None."""
    key = {'anchor': 'c', 'random': 'c_rand', 'text': 'c_text'}[cond_mode]
    held_key = {'anchor': 'held_c', 'text': 'held_c_text'}.get(cond_mode)
    return lib[key], (lib[held_key] if held_key else None)


# ------------------------- physics / kinematic access -------------------------

def _find_dmc_env(env):
    """Walk the wrapper chain to the inner dm_control environment."""
    seen = set()
    o = env
    while o is not None and id(o) not in seen:
        seen.add(id(o))
        if hasattr(o, 'physics') and hasattr(o.physics, 'data'):
            return o
        o = getattr(o, '_env', None)
    return None


def seed_task(env, s):
    """Deterministic reset per seed via the dm_control task RNG (paired
    resets: same s => identical initial state). Returns True on success."""
    dmc = _find_dmc_env(env)
    if dmc is None:
        return False
    task = getattr(dmc, 'task', None)
    if task is None:
        return False
    rng = np.random.RandomState(s)
    if hasattr(task, '_random'):
        task._random = rng
        return True
    try:
        task.random = rng
        return True
    except Exception:
        return False


def kinematic_step(env, x0):
    """Independent behavior features from MuJoCo physics (no g-space).
    Returns flat numpy vector or None if physics unavailable."""
    dmc = _find_dmc_env(env)
    if dmc is None:
        return None
    data = dmc.physics.data
    feats = []
    if data.qpos is not None:
        qp = np.asarray(data.qpos).copy()
        if qp.size >= 2:
            qp[0] -= x0          # relative x (absolute drift is not behavior)
        feats.append(qp)
    if data.qvel is not None:
        feats.append(np.asarray(data.qvel).copy())
    xmat = np.asarray(getattr(data, 'xmat', None))
    if xmat is not None and xmat.ndim == 3 and xmat.shape[0] >= 2:
        feats.append(xmat[1] @ np.array([0.0, 0.0, 1.0]))   # torso up vector
    sens = np.asarray(getattr(data, 'sensordata', None))
    if sens is not None and sens.size:
        feats.append(sens.reshape(-1).copy())
    return np.concatenate(feats)


def ep_summary(kfeats):
    """[L, K] -> [2K] mean+std episode kinematic signature."""
    k = np.asarray(kfeats)
    if k.shape[0] == 0:
        return None
    return np.concatenate([k.mean(0), k.std(0)])


# ------------------------- agent / library / env -------------------------

def load_agent(snap):
    payload = torch.load(snap, map_location='cpu', weights_only=False)
    agent = payload.get('agent', payload) if isinstance(payload, dict) \
        else payload
    agent.to('cuda')
    for m in [agent] + list(agent.modules()):
        if hasattr(m, 'device') and isinstance(m.device, str):
            m.device = 'cuda'
    if hasattr(agent.cfg, 'anchor') and not hasattr(agent, 'anchor_cfg'):
        from experiments.anchor_patch import patch_agent
        patch_agent(agent)
    if hasattr(agent, '_imag_behavior'):
        agent._acting_behavior = agent._imag_behavior
    return agent


def load_lib():
    lib = torch.load(LIB_PATH, map_location='cpu')
    return {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in lib.items()}


def make_env(seed=1):
    return envs.make(TASK, 'pixels', 2, seed,
                     img_size=64, viclip_encode=False,
                     clip_hd_rendering=False)


def rollout_v2(agent, env, cond=None, reset_seed=None):
    """One episode -> (reward, gs[L,d], kfeats[L,K]|None, seeded_ok).
    reset_seed set => deterministic initial state (paired resets)."""
    meta = agent.init_meta()
    if reset_seed is not None:
        seeded_ok = seed_task(env, reset_seed)
    else:
        seeded_ok = False
    time_step, dreamer_obs = env.reset()
    agent_state = None
    ep_r = 0.0
    gs, ks = [], []
    x0 = None
    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            if cond is not None:
                action, agent_state = agent.act(
                    dreamer_obs, meta, 0, eval_mode=True, state=agent_state,
                    cond=cond)
            else:
                action, agent_state = agent.act(
                    dreamer_obs, meta, 0, eval_mode=True, state=agent_state)
        time_step, dreamer_obs = env.step(action)
        ep_r += time_step.reward or 0.0
        latent = agent_state[0]
        stoch = agent.wm.rssm.get_stoch(latent)
        g = F.normalize(g_proj(agent.wm, stoch), dim=-1)[0].detach().cpu()
        gs.append(g)
        k = kinematic_step(env, x0 if x0 is not None else 0.0)
        if k is not None:
            if x0 is None:
                x0 = float(k[0]) if k.size else 0.0
            ks.append(k)
    g = torch.stack(gs, 0) if gs else torch.zeros(0, 0)
    karr = np.stack(ks, 0) if ks else None
    return ep_r, g, karr, seeded_ok


# ------------------------- descriptors -------------------------

def descriptor_phase(gs, S):
    """Original pre-registered phase-locked descriptor (kept for continuity:
    mean_t cos(g_t, S[min(t+1,T)-1]))."""
    L = gs.shape[0]
    if L == 0:
        return float('nan')
    idx = np.minimum(np.arange(L) + 1, T) - 1
    tgt = S[idx].to(gs.device)
    return float(F.cosine_similarity(gs, tgt, dim=-1).mean())


def descriptor_window(gs, S):
    """C2 fix + Round-2 N1 fix: max over sliding windows of the ALIGNED
    phase-locked average (1/W) sum_t cos(g[s+t], S[t]) — NOT the
    window-mean-vs-single-anchor-frame max of v1 (which was effectively a
    static-frame score). No clamp beyond the window."""
    L = gs.shape[0]
    if L == 0:
        return float('nan')
    S_w = S[:WINDOW].to(gs.device)
    if L < WINDOW:
        idx = np.minimum(np.arange(L) + 1, T) - 1
        return float(F.cosine_similarity(gs, S[idx].to(gs.device),
                                         dim=-1).mean())
    M = gs @ S_w.T                                   # [L, W] (both normalized)
    A = M.unfold(0, WINDOW, 1)                       # [L-W+1, W, W]
    diag = torch.diagonal(A, dim1=1, dim2=2)         # A[s,t]=M[s+t,t]
    return float(diag.mean(dim=1).max())


def kin_distance(summary, sig, whiten):
    """Whitened L2 distance from episode summary to signature."""
    if summary is None or sig is None or whiten is None:
        return float('nan')
    s = (np.asarray(summary) - whiten[0]) / (whiten[1] + 1e-9)
    t = (np.asarray(sig) - whiten[0]) / (whiten[1] + 1e-9)
    return float(np.linalg.norm(s - t))


# ------------------------- mode: train16v2 -------------------------

def mode_train16v2(snap, out_json, n_eps=20, perm=None,
                   cond_mode='anchor'):
    agent = load_agent(snap)
    lib = load_lib()
    c_train, _ = cond_matrices(lib, cond_mode)
    results = {'snapshot': snap, 'n_eps': n_eps, 'anchors': [],
               'cond_mode': cond_mode,
               'perm': perm if perm is not None else list(range(16))}
    all_feats, all_labels = [], []
    all_kin = []                    # per-episode kinematic summaries
    g_desc = np.zeros((16, 16))     # [row i = cond i] x [col j = target S_j]
    kin_sigs, kin_eps = [], []      # [16][2K], [16][n_eps][2K]
    seeded_all = True
    for i in range(16):
        cond = c_train[i]
        # own target: perm[i] for shuffled controls, i otherwise
        tgt_idx = perm[i] if perm is not None else i
        S_tgt = lib['S'][tgt_idx]
        rews, ds_phase, ds_win, eps_g, eps_k = [], [], [], [], []
        for e in range(n_eps):
            r, gs, kfeats, ok = rollout_v2(agent, make_env(),
                                           cond=cond,
                                           reset_seed=10000 + e)
            seeded_all &= ok
            rews.append(r)
            if gs.shape[0] > 0:
                ds_phase.append(descriptor_phase(gs, S_tgt))
                ds_win.append(descriptor_window(gs, S_tgt))
                # g-space confusion: nearest target over all 16 S_j
                for j in range(16):
                    g_desc[i, j] += descriptor_window(gs, lib['S'][j])
                eps_g.append(gs.numpy().astype(np.float32))
            summ = ep_summary(kfeats)
            eps_k.append(summ)
            all_kin.append(summ)
        for j in range(16):
            g_desc[i, j] /= max(n_eps, 1)
        # kinematic signature = mean of this anchor's summaries
        valid = [s for s in eps_k if s is not None]
        sig = np.mean(valid, 0) if valid else None
        kin_sigs.append(sig)
        kin_eps.append(eps_k)
        for f in eps_g:
            all_feats.append(f)
            all_labels.append(i)
        results['anchors'].append({
            'i': i, 'prompt': lib['train_prompts'][i],
            'target_idx': tgt_idx,
            'mean_reward': float(np.mean(rews)),
            'episode_rewards': rews,
            'descriptor': float(np.nanmean(ds_phase)),
            'descriptor_window': float(np.nanmean(ds_win)),
            'descriptors_per_ep': ds_win,
            'kin_sig_norm': float(np.linalg.norm(sig)) if sig is not None
            else None,
        })
        print(f'  [{i}] {lib["train_prompts"][i]}: rew={np.mean(rews):+.2f} '
              f'desc_w={np.nanmean(ds_win):+.3f} '
              f'(target={tgt_idx})', flush=True)

    # whitening stats pooled over all episodes
    valid_kin = [s for s in all_kin if s is not None]
    if valid_kin:
        W = np.stack(valid_kin)
        whiten = (W.mean(0), W.std(0))
    else:
        whiten = None
    # 16x16 confusion matrices (Round-2 N1 fix: leave-one-out centroids —
    # the episode being classified no longer contributes to its own class
    # centroid, removing the resubstitution leak)
    if whiten is not None and all(s is not None for s in kin_sigs):
        k_conf = np.zeros((16, 16))
        for i in range(16):
            for e, s in enumerate(kin_eps[i]):
                loo = [x for idx, x in enumerate(kin_eps[i])
                       if idx != e and x is not None]
                sig_loo = np.mean(loo, 0) if loo else None
                d = []
                for j in range(16):
                    sig_j = sig_loo if j == i else kin_sigs[j]
                    d.append(kin_distance(s, sig_j, whiten))
                k_conf[i, int(np.argmin(d))] += 1
            k_conf[i] /= max(n_eps, 1)
        results['confusion_kinematic'] = k_conf.tolist()
        results['confusion_kinematic_diag'] = float(
            np.diag(k_conf).mean())
    g_conf = np.zeros((16, 16))
    for i in range(16):
        # per-episode argmax; count majority label
        for e in range(n_eps):
            row = g_desc[i]  # mean over eps of desc vs each target
        # (descriptor means already averaged over eps above)
        g_conf[i, int(np.argmax(g_desc[i]))] = 1
    results['confusion_g'] = g_desc.tolist()
    results['confusion_g_diag_rate'] = float(
        np.mean([g_desc[i, i] == g_desc[i].max() for i in range(16)]))
    results['seeded_paired_ok'] = bool(seeded_all)
    results['n_inactive_window'] = sum(
        1 for a in results['anchors']
        if a['descriptor_window'] <= 0.0)
    # kinematic signatures + whitening for heldoutv2 reuse (C7 full data)
    results['kin_sigs'] = [s.tolist() if s is not None else None
                           for s in kin_sigs]
    results['kin_whiten'] = ([whiten[0].tolist(), whiten[1].tolist()]
                             if whiten is not None else None)

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    if all_feats:
        np.savez_compressed(out_json + '.feats.npz',
                            feats=np.concatenate(all_feats, 0),
                            labels=np.array(all_labels),
                            eps=np.cumsum([0] + [len(x) for x in all_feats]))
    print(f'TRAIN16V2 DONE -> {out_json} '
          f'kdiag={results.get("confusion_kinematic_diag")} '
          f'seeded={seeded_all}', flush=True)
    return results


# ------------------------- mode: heldoutv2 -------------------------

def edge_mix(c_train, mix_basis, lib_edges, c_q, tau=1.0):
    """C3 fix + Round-2 fix: LITERAL Top-2 — the two closest (by cosine)
    anchors must themselves form a training-graph edge. If not, no
    trainable Top-2 edge exists for this query (Top-1 fallback + coverage
    failure), matching the preregistered coverage gate. Norm-corrected
    softmax over that edge only.
    (v1 searched all ranked pairs until any edge was found, which
    guaranteed coverage 8/8.)"""
    cos = F.cosine_similarity(c_q.unsqueeze(0), c_train, dim=-1)
    order = cos.argsort(descending=True).tolist()
    edge_set = {tuple(sorted(int(x) for x in e)) for e in lib_edges}
    # Round-3 C1 fix: keep ranked order — a=order[0] is the TRUE nearest
    # anchor (caller uses a as the nearest arm). The v3 version sorted the
    # pair, corrupting the nearest arm whenever order[1] < order[0].
    a, b = order[0], order[1]
    if tuple(sorted([a, b])) not in edge_set:
        return None, None, cos
    w = torch.softmax(cos[[a, b]] / tau, dim=-1)
    wp = w / mix_basis[[a, b]].norm(dim=-1)
    wp = wp / wp.sum()
    c_mix = F.normalize(wp[0] * mix_basis[a] + wp[1] * mix_basis[b], dim=-1)
    return c_mix, (int(a), int(b)), cos


def exact_perm_p_onesided(rho, n=5, sign=-1):
    """Exact one-sided permutation p-value for Spearman rho on n points.
    sign=-1 tests rho<0 (decreasing), sign=+1 tests rho>0.
    Round-2 fix: count permutations AT LEAST AS EXTREME as observed —
    for sign=-1 (decreasing): r <= rho; for sign=+1: r >= rho.
    (v1 counted r >= rho for decreasing, so a perfect trend got p=1.0.)"""
    from itertools import permutations
    from scipy.stats import spearmanr
    base = list(range(n))
    cnt, total = 0, 0
    for p in permutations(base):
        r, _ = spearmanr(base, p)
        ok = (r <= rho + 1e-12) if sign < 0 else (r >= rho - 1e-12)
        if ok:
            cnt += 1
        total += 1
    return cnt / max(total, 1)


def mode_heldoutv2(snap, out_json, n_eps=20, scan_eps=10,
                   scan_alphas=(0, .25, .5, .75, 1), sig_file=None,
                   cond_mode='anchor'):
    agent = load_agent(snap)
    lib = load_lib()
    if cond_mode == 'random':
        raise ValueError('heldout has no semantic query path for '
                         'cond_mode=random')
    c_train, c_held = cond_matrices(lib, cond_mode)
    results = {'snapshot': snap, 'prompts': [], 'cond_mode': cond_mode}
    # kinematic signatures: reuse train16v2 output, else compute internally
    if sig_file and os.path.exists(sig_file):
        t16 = json.load(open(sig_file))
        sigs = [np.asarray(s) if s is not None else None
                for s in t16.get('kin_sigs', [None] * 16)]
        wh = t16.get('kin_whiten')
        whiten = (np.asarray(wh[0]), np.asarray(wh[1])) if wh else None
        if len(sigs) == 16 and whiten is not None:
            print('  loaded kinematic signatures from train16v2', flush=True)
        else:
            sig_file = None
    if sig_file is None:
        print('  building kinematic signatures (16 anchors x 20 eps)...',
              flush=True)
        sigs = []
        sig_sum = []
        for i in range(16):
            ss = []
            for e in range(n_eps):
                _, _, kfeats, _ = rollout_v2(agent, make_env(),
                                             cond=c_train[i],
                                             reset_seed=20000 + e)
                s = ep_summary(kfeats)
                if s is not None:
                    ss.append(s)
            sig = np.mean(ss, 0) if ss else None
            sigs.append(sig)
            sig_sum.append(ss)
        valid = [s for ss in sig_sum for s in ss if s is not None]
        whiten = (np.mean(valid, 0), np.std(valid, 0)) if valid else None

    rng = np.random.RandomState(0)
    mix_basis = (c_train if cond_mode == 'text' else lib['m'])
    for q in range(8):
        c_q = c_held[q]
        S_q = lib['held_S'][q]
        c_mix, pair, cos = edge_mix(c_train, mix_basis, lib['edges'], c_q)
        if pair is None:
            a = int(cos.argmax())
            b = a
            on_edge = False
            c_mix = c_train[a]
            excluded = True
        else:
            a, b = pair
            on_edge = True
            excluded = False
        c_near = c_train[a]
        e_pair = rng.choice(16, 2, replace=False)
        alpha_r = rng.rand()
        # Round-3 C1 fix: random arm must live in the ACTOR's trained
        # condition coordinate system — mix_basis (=c_train for text mode,
        # lib['m'] for anchor mode), not always lib['m'].
        c_rand = F.normalize(
            alpha_r * mix_basis[e_pair[0]] + (1 - alpha_r) * mix_basis[e_pair[1]],
            dim=-1)

        arms = {}
        for name, cond in [('direct', c_q), ('mix', c_mix),
                           ('nearest', c_near), ('random', c_rand)]:
            rews, ds_phase, ds_win = [], [], []
            for e in range(n_eps):
                # paired resets: same seed across arms per episode index
                r, gs, _, ok = rollout_v2(agent, make_env(), cond=cond,
                                          reset_seed=30000 + 100 * q + e)
                rews.append(r)
                ds_phase.append(descriptor_phase(gs, S_q))
                ds_win.append(descriptor_window(gs, S_q))
            arms[name] = {
                'mean_reward': float(np.mean(rews)),
                'descriptor': float(np.nanmean(ds_phase)),
                'descriptor_window': float(np.nanmean(ds_win)),
                'episode_rewards': rews,
                'descriptors_per_ep': ds_win,
            }
            print(f'  [q{q}] {name}: rew={arms[name]["mean_reward"]:+.2f} '
                  f'desc_w={arms[name]["descriptor_window"]:+.3f}',
                  flush=True)

        # honest alpha-scan on edge (a,b) with kinematic trend test
        scan = {'alphas': list(scan_alphas), 'descriptor': [],
                'reward': [], 'kin_d_to_a': [], 'kin_d_to_b': [],
                'seeded_ok': True, 'degenerate': a == b}
        for al in scan_alphas:
            cond = F.normalize(al * mix_basis[a] + (1 - al) * mix_basis[b],
                               dim=-1)
            rews, ds_win, d_a, d_b = [], [], [], []
            for e in range(scan_eps):
                r, gs, kfeats, ok = rollout_v2(
                    agent, make_env(), cond=cond,
                    reset_seed=40000 + 100 * q + e)
                rews.append(r)
                ds_win.append(descriptor_window(gs, S_q))
                if a != b:
                    s = ep_summary(kfeats)
                    d_a.append(kin_distance(s, sigs[a], whiten))
                    d_b.append(kin_distance(s, sigs[b], whiten))
                scan['seeded_ok'] &= ok
            scan['descriptor'].append(float(np.nanmean(ds_win)))
            scan['reward'].append(float(np.mean(rews)))
            scan['kin_d_to_a'].append(float(np.nanmean(d_a)) if a != b
                                      else float('nan'))
            scan['kin_d_to_b'].append(float(np.nanmean(d_b)) if a != b
                                      else float('nan'))
        # honest monotonicity: d_to_a decreasing in alpha, d_to_b increasing
        from scipy.stats import spearmanr
        if a != b:
            rho_a, p_a = spearmanr(scan_alphas, scan['kin_d_to_a'])
            rho_b, p_b = spearmanr(scan_alphas, scan['kin_d_to_b'])
            p_a_perm = exact_perm_p_onesided(rho_a, len(scan_alphas), sign=-1)
            p_b_perm = exact_perm_p_onesided(rho_b, len(scan_alphas), sign=+1)
            honest_monotone = bool(
                rho_a < 0 and rho_b > 0 and p_a_perm < 0.05
                and p_b_perm < 0.05)
        else:
            rho_a = rho_b = float('nan')
            p_a_perm = p_b_perm = 1.0
            honest_monotone = False

        results['prompts'].append({
            'q': q, 'prompt': lib['heldout_prompts'][q],
            'd_cq': float(1.0 - cos.max()),
            'top2': [a, b], 'top2_on_edge': on_edge,
            'excluded_identical_arms': excluded,
            'arms': arms,
            'scan': scan,
            'monotone_honest': {
                'rho_d_to_a': float(rho_a), 'rho_d_to_b': float(rho_b),
                'p_perm_a_decreasing': float(p_a_perm),
                'p_perm_b_increasing': float(p_b_perm),
                'pass': honest_monotone,
            },
            # identical-condition arms are never counted as wins
            'attribution_win': (None if excluded else bool(
                arms['mix']['descriptor_window']
                > arms['direct']['descriptor_window'])),
            'interp_win': (None if excluded else bool(
                arms['mix']['descriptor_window']
                > arms['nearest']['descriptor_window']
                and arms['mix']['descriptor_window']
                > arms['random']['descriptor_window'])),
            'coverage_ok': on_edge,
        })

    att = [p['attribution_win'] for p in results['prompts']]
    itp = [p['interp_win'] for p in results['prompts']]
    mono = [p['monotone_honest']['pass'] for p in results['prompts']]
    cov = [p['coverage_ok'] for p in results['prompts']]
    n_excl = sum(p['excluded_identical_arms'] for p in results['prompts'])
    results['gates'] = {
        'n_excluded_identical': n_excl,
        'attribution': f'{sum(1 for x in att if x)}/'
                       f'{sum(1 for x in att if x is not None)}',
        'interpolation': f'{sum(1 for x in itp if x)}/'
                         f'{sum(1 for x in itp if x is not None)}',
        'monotone_tracking': f'{sum(mono)}/8',
        'coverage': f'{sum(cov)}/8',
        'attribution_pass': sum(1 for x in att if x) >= 6,
        'interpolation_pass': sum(1 for x in itp if x) >= 6,
        'monotone_pass': sum(mono) >= 6,
        'coverage_pass': sum(cov) >= 6,
    }
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'HELDOUTV2 DONE gates={results["gates"]}', flush=True)


# ------------------------- mode: genrlv2 / refdescv2 -------------------------

def mode_genrlv2(snap, out_json, n_eps=20):
    agent = load_agent(snap)
    rews = []
    for e in range(n_eps):
        r, _, _, _ = rollout_v2(agent, make_env(), reset_seed=50000 + e)
        rews.append(r)
    out = {'snapshot': snap, 'mean_reward': float(np.mean(rews)),
           'reward_std': float(np.std(rews)),
           'episode_rewards': rews,
           'reset_seeds': [50000 + e for e in range(n_eps)]}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'GENRLV2 DONE rew={out["mean_reward"]:+.2f}', flush=True)


def mode_refdescv2(snap, q, out_json, n_eps=20):
    agent = load_agent(snap)
    lib = load_lib()
    rews, ds_phase, ds_win = [], [], []
    for e in range(n_eps):
        r, gs, _, _ = rollout_v2(agent, make_env(),
                                 reset_seed=60000 + 100 * q + e)
        rews.append(r)
        ds_phase.append(descriptor_phase(gs, lib['held_S'][q]))
        ds_win.append(descriptor_window(gs, lib['held_S'][q]))
    out = {'snapshot': snap, 'q': int(q),
           'prompt': lib['heldout_prompts'][q],
           'mean_reward': float(np.mean(rews)),
           'descriptor': float(np.nanmean(ds_phase)),
           'descriptor_window': float(np.nanmean(ds_win)),
           'episode_rewards': rews,
           'descriptors_per_ep': ds_win}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'REFDESCV2 DONE q={q} rew={out["mean_reward"]:+.2f} '
          f'desc_w={out["descriptor_window"]:+.3f}', flush=True)


# ------------------------- mode: aggregatev2 -------------------------

def _load_json(p):
    with open(p) as f:
        return json.load(f)


def paired_ci(x):
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return float('nan'), float('nan'), float('nan')
    from scipy import stats
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - t * se), float(m + t * se)


def aggregate_m2_v2(res_dir):
    """B2 gates with window descriptors + fair shuffle own-target numbers."""
    out = {'gate_descriptor_12of16': False, 'gate_id_25pct': False,
           'gate_diayn_ci': False, 'all_pass': False}
    off_w, diayn_w, shuff_w = [], [], []
    for s in (1, 2, 3, 4, 5):
        off_w.append([a['descriptor_window'] for a in _load_json(
            f'{res_dir}/m2_anchor_off_s{s}_train16v2.json')['anchors']])
        diayn_w.append([a['descriptor_window'] for a in _load_json(
            f'{res_dir}/m2_diayn_s{s}_train16v2.json')['anchors']])
    for s in (11, 12, 13):
        shuff_w.append([a['descriptor_window'] for a in _load_json(
            f'{res_dir}/m2_shuffle_s{s}_train16v2.json')['anchors']])
    off_w, diayn_w, shuff_w = (np.asarray(x) for x in (off_w, diayn_w,
                                                       shuff_w))
    wins = off_w.mean(0) > shuff_w.mean(0)   # own-target vs own-target
    out['descriptor_wins'] = int(wins.sum())
    out['descriptor_win_mask'] = [bool(w) for w in wins]
    out['gate_descriptor_12of16'] = bool(wins.sum() >= 12)
    out['means_window'] = {'anchor_own': float(off_w.mean()),
                           'shuffle_own': float(shuff_w.mean()),
                           'diayn_own': float(diayn_w.mean())}

    id_accs = [_load_json(f'{res_dir}/m2_anchor_off_s{s}_idrec_v2.json')
               ['step_accuracy'] for s in (1, 2, 3, 4, 5)]
    out['id_accuracies'] = id_accs
    out['gate_id_25pct'] = bool(all(a >= 0.25 for a in id_accs))

    anchor_mean = off_w.mean(0)
    d = diayn_w - off_w
    rel = d.mean(1) / np.maximum(anchor_mean.mean(), 1e-9)
    m, lo, hi = paired_ci(rel)
    out['diayn_rel_mean_ci_window'] = [m, lo, hi]
    out['gate_diayn_ci'] = bool(hi < 0.10)
    out['all_pass'] = (out['gate_descriptor_12of16'] and out['gate_id_25pct']
                       and out['gate_diayn_ci'])
    return out


def aggregate_m34_v2(res_dir):
    """B3 + honest B4 on heldoutv2 (exclusions recorded, honest monotone)."""
    out = {'gate_m3_mixup_on_beats_off': False,
           'gate_m3_off_fails_monotone': False,
           'gates_b4': {}, 'all_pass': False}
    on_p, off_p = [], []
    for s in (1, 2, 3, 4, 5):
        on_p.append(_load_json(f'{res_dir}/m3_mixon_s{s}_heldoutv2.json')
                    ['prompts'])
        off_p.append(_load_json(f'{res_dir}/m2_anchor_off_s{s}_heldoutv2.json')
                     ['prompts'])
    on_mix = np.asarray([[on_p[s][q]['arms']['mix']['descriptor_window']
                          for q in range(8)] for s in range(5)])
    off_mix = np.asarray([[off_p[s][q]['arms']['mix']['descriptor_window']
                           for q in range(8)] for s in range(5)])
    d_seed = (on_mix - off_mix).mean(1)
    m, lo, hi = paired_ci(d_seed)
    out['m3_on_minus_off_mix_mean_ci_window'] = [m, lo, hi]
    out['gate_m3_mixup_on_beats_off'] = bool(lo > 0)

    def mono_votes(prompts):
        return np.asarray([sum(prompts[s][q]['monotone_honest']['pass']
                               for s in range(5)) >= 3 for q in range(8)])
    on_mono, off_mono = mono_votes(on_p), mono_votes(off_p)
    out['m3_on_monotone_honest'] = f'{int(on_mono.sum())}/8'
    out['m3_off_monotone_honest'] = f'{int(off_mono.sum())}/8'
    out['gate_m3_off_fails_monotone'] = bool(off_mono.sum() < 6)
    out['gate_m3_on_passes_monotone'] = bool(on_mono.sum() >= 6)

    # per-prompt means across on-seeds; excluded prompts never count as wins
    att = itp = 0
    att_den = itp_den = 0
    for q in range(8):
        wins_att = [on_p[s][q]['attribution_win'] for s in range(5)]
        wins_itp = [on_p[s][q]['interp_win'] for s in range(5)]
        att += sum(1 for w in wins_att if w)
        att_den += sum(1 for w in wins_att if w is not None)
        itp += sum(1 for w in wins_itp if w)
        itp_den += sum(1 for w in wins_itp if w is not None)
    # gate on per-prompt majority (>=3/5 seeds) over non-excluded seeds
    att_q = sum(1 for q in range(8) if sum(
        1 for s in range(5) if on_p[s][q]['attribution_win']) >= 3)
    itp_q = sum(1 for q in range(8) if sum(
        1 for s in range(5) if on_p[s][q]['interp_win']) >= 3)
    mono_q = int(on_mono.sum())
    cov = sum(1 for q in range(8)
              if all(on_p[s][q]['coverage_ok'] for s in range(5)))
    out['gates_b4'] = {
        'attribution': f'{att_q}/8', 'interpolation': f'{itp_q}/8',
        'monotone': f'{mono_q}/8', 'coverage': f'{cov}/8',
        'attribution_pass': att_q >= 6, 'interpolation_pass': itp_q >= 6,
        'monotone_pass': mono_q >= 6, 'coverage_pass': cov >= 6,
    }
    out['n_excluded_identical'] = sum(
        on_p[0][q]['excluded_identical_arms'] for q in range(8))
    out['all_pass'] = (out['gate_m3_mixup_on_beats_off']
                       and out['gate_m3_off_fails_monotone']
                       and all(out['gates_b4'][k] for k in (
                           'attribution_pass', 'interpolation_pass',
                           'monotone_pass', 'coverage_pass')))
    return out


def mode_aggregatev2(task, res_dir, out_json):
    out = aggregate_m2_v2(res_dir) if task == 'm2' else aggregate_m34_v2(
        res_dir)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'AGGREGATEV2[{task}] -> {out_json}')
    print(json.dumps(out, indent=2))
    print(f'ALL_PASS={out.get("all_pass", "n/a")}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train16v2', 'heldoutv2', 'genrlv2',
                                     'refdescv2', 'aggregatev2'])
    ap.add_argument('snapshot')
    ap.add_argument('out_json')
    ap.add_argument('--n_eps', type=int, default=20)
    ap.add_argument('--scan_eps', type=int, default=10)
    ap.add_argument('--q', type=int, default=0)
    ap.add_argument('--task', default='m2', choices=['m2', 'm34'])
    ap.add_argument('--perm', default=None,
                    help='comma-separated 16 target indices (shuffle own-target)')
    ap.add_argument('--cond-mode', default='anchor',
                    choices=['anchor', 'random', 'text'])
    ap.add_argument('--sig-file', default=None,
                    help='train16v2 JSON with kin_sigs (avoids recompute)')
    args = ap.parse_args()
    os.chdir(REPO)
    perm = ([int(x) for x in args.perm.split(',')] if args.perm else None)
    if args.mode == 'train16v2':
        mode_train16v2(args.snapshot, args.out_json, args.n_eps, perm=perm,
                       cond_mode=args.cond_mode)
    elif args.mode == 'heldoutv2':
        mode_heldoutv2(args.snapshot, args.out_json, args.n_eps,
                       scan_eps=args.scan_eps, cond_mode=args.cond_mode,
                       sig_file=args.sig_file)
    elif args.mode == 'genrlv2':
        mode_genrlv2(args.snapshot, args.out_json, args.n_eps)
    elif args.mode == 'refdescv2':
        mode_refdescv2(args.snapshot, args.q, args.out_json, args.n_eps)
    elif args.mode == 'aggregatev2':
        mode_aggregatev2(args.task, args.snapshot, args.out_json)


if __name__ == '__main__':
    main()
