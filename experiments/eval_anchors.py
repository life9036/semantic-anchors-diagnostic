"""Offline evaluation suite for the anchor experiments (M2/M3/M4).

Modes:
  train16  <snapshot> <out.json>       16 anchor descriptors x 20 eps each;
                                       saves per-step g(stoch) features to
                                       <out>.feats.npz for ID recovery.
  idrecover <feats.npz> <out.json>     train/test 16-class classifier
                                       (pre-registered: 3x512 MLP, 70/30
                                       episode split, per-step accuracy).
  heldout  <snapshot> <out.json>       8 held-out prompts, four-way attribution
                                       (direct c_q / c_mix / nearest anchor /
                                       random interpolation) x 20 eps +
                                       alpha-scan on the top-2 train edge
                                       (5 x 10 eps); computes the three gates.
  genrl    <snapshot> <out.json>       plain env-reward eval (GenRL baselines).

Pre-registered operationalizations (2026-08-31):
  descriptor_q(ep) = mean_t cos(g(s_t), S_q(min(t+1, T)))  (phase-locked)
  attribution gate: >=6/8 prompts with desc(c_mix) > desc(direct) STRICTLY
  interpolation gate: >=6/8 with desc(c_mix) > nearest AND > random
  monotone tracking: alpha-profile over the top-2 edge is non-decreasing up
    to its argmax and non-increasing after (<=1 total violation allowed)
  coverage gate: >=6/8 held-out prompts whose top-2 train anchors form an
    edge of the frozen train graph (else Top-1 fallback + coverage fail)
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
import torch.nn as nn
import torch.nn.functional as F

import tools.utils as utils
import envs.main as envs
from experiments.anchor_lib import g_proj, T
# class-level anchor methods must be installed before any snapshot load
# (anchor-trained agents are plain GenRLAgent + class patches)
from experiments.anchor_patch import install
install()

REPO = os.path.expanduser('~/autodl-tmp/genrl')
LIB_PATH = os.path.join(REPO, 'exp/anchor_lib/walker_lib.pt')


def load_agent(snap):
    payload = torch.load(snap, map_location='cpu', weights_only=False)
    agent = payload.get('agent', payload)
    agent.to('cuda')
    for m in [agent] + list(agent.modules()):
        if hasattr(m, 'device') and isinstance(m.device, str):
            m.device = 'cuda'
    # anchor-trained agents are plain GenRLAgent + injected methods; if a
    # snapshot predates the patch (or pickling dropped a method), re-apply.
    if hasattr(agent.cfg, 'anchor') and not hasattr(agent, 'anchor_cfg'):
        from experiments.anchor_patch import patch_agent
        patch_agent(agent)
    # the imagination behavior is the trained policy for ALL modes (anchor
    # runs and GenRL baselines alike — same swap as the pilot final eval)
    if hasattr(agent, '_imag_behavior'):
        agent._acting_behavior = agent._imag_behavior
    return agent


def load_lib():
    lib = torch.load(LIB_PATH, map_location='cpu')
    return {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in lib.items()}


def make_env(task='walker_walk', seed=1):
    return envs.make(task, 'pixels', 2, seed, img_size=64,
                     viclip_encode=False, clip_hd_rendering=False)


def rollout(agent, env, cond=None):
    """One episode -> (reward, [g(stoch_t)] normalized per step)."""
    meta = agent.init_meta()
    time_step, dreamer_obs = env.reset()
    agent_state = None
    ep_r = 0.0
    gs = []
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
        # get_stoch flattens [1,32,32] -> [1,1024]; g_proj needs the flat form
        stoch = agent.wm.rssm.get_stoch(latent)
        g = F.normalize(g_proj(agent.wm, stoch), dim=-1)[0].detach().cpu()
        gs.append(g)
    return ep_r, torch.stack(gs, 0) if gs else torch.zeros(0, 0)


def descriptor(gs, S):
    """gs [L, d], S [T, d] (normalized): mean_t cos(g_t, S[min(t+1,T)-1])."""
    L = gs.shape[0]
    if L == 0:
        return float('nan')
    idx = np.minimum(np.arange(L) + 1, T) - 1
    tgt = S[idx].to(gs.device)
    return float(F.cosine_similarity(gs, tgt, dim=-1).mean())


def run_eps(agent, env, cond, n_eps, S_q):
    rews, descs, feats = [], [], []
    for _ in range(n_eps):
        r, gs = rollout(agent, env, cond=cond)
        rews.append(r)
        if gs.shape[0] > 0:
            descs.append(descriptor(gs, S_q))
            feats.append(gs)
    return float(np.mean(rews)), float(np.mean(descs)), feats


# ------------------------- mode: train16 -------------------------

def mode_train16(snap, out_json, n_eps=20):
    agent = load_agent(snap)
    lib = load_lib()
    env = make_env()
    results = {'snapshot': snap, 'n_eps': n_eps, 'anchors': []}
    all_feats, all_labels = [], []
    for i in range(16):
        cond = lib['c'][i]
        rew, desc, feats = run_eps(agent, env, cond, n_eps, lib['S'][i])
        results['anchors'].append({
            'i': i, 'prompt': lib['train_prompts'][i],
            'mean_reward': rew, 'descriptor': desc})
        for f in feats:
            all_feats.append(f.numpy().astype(np.float32))
            all_labels.append(i)
        print(f'  [{i}] {lib["train_prompts"][i]}: rew={rew:+.2f} '
              f'desc={desc:+.3f}', flush=True)
    inactive = sum(1 for a in results['anchors'] if a['descriptor'] <= 0.0)
    results['n_inactive'] = inactive
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_json + '.feats.npz',
                        feats=np.concatenate(all_feats, 0) if all_feats
                        else np.zeros((0, 0)),
                        labels=np.array(all_labels),
                        eps=np.cumsum([0] + [len(f) for f in all_feats]))
    print(f'TRAIN16 DONE -> {out_json} inactive={inactive}', flush=True)


# ------------------------- mode: idrecover -------------------------

class IDClassifier(nn.Module):
    def __init__(self, d, K=16, layers=3, units=512):
        super().__init__()
        net = []
        last = d
        for _ in range(layers):
            net += [nn.Linear(last, units, bias=False), nn.LayerNorm(units),
                    nn.SiLU()]
            last = units
        net += [nn.Linear(last, K)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def mode_idrecover(feats_npz, out_json, seed=0, epochs=50, bs=256):
    z = np.load(feats_npz)
    feats, labels = z['feats'], z['labels']
    ep_ends = z['eps']
    # one label per episode (labels[i] pairs with ep_ends[i]..ep_ends[i+1])
    eps = [(feats[ep_ends[i]:ep_ends[i + 1]], int(labels[i]))
           for i in range(len(ep_ends) - 1)]
    n = len(eps)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_tr = int(round(0.7 * n))
    tr_eps = [eps[j] for j in order[:n_tr]]
    te_eps = [eps[j] for j in order[n_tr:]]
    Xtr = torch.as_tensor(np.concatenate([e[0] for e in tr_eps], 0)).cuda()
    ytr = torch.cat(
        [torch.full((len(e[0]),), e[1], dtype=torch.long) for e in tr_eps]
    ).cuda()
    model = IDClassifier(Xtr.shape[1]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0])
        for s in range(0, Xtr.shape[0], bs):
            idx = perm[s:s + bs]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    # per-step accuracy + per-episode majority-vote accuracy on test
    step_correct, step_total = 0, 0
    ep_correct = 0
    with torch.no_grad():
        for fe, lab in te_eps:
            logits = model(torch.as_tensor(fe).cuda())
            pred = logits.argmax(-1)
            step_correct += (pred == lab).sum().item()
            step_total += len(pred)
            vote = torch.bincount(pred, minlength=16).argmax().item()
            ep_correct += int(vote == lab)
    step_acc = step_correct / max(step_total, 1)
    ep_acc = ep_correct / max(len(te_eps), 1)
    out = {'step_accuracy': step_acc, 'episode_accuracy': ep_acc,
           'n_train_eps': n_tr, 'n_test_eps': len(te_eps),
           'gate_id_recover_25pct': bool(step_acc >= 0.25)}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'IDRECOVER DONE step_acc={step_acc:.3f} ep_acc={ep_acc:.3f} '
          f'gate={"PASS" if out["gate_id_recover_25pct"] else "FAIL"}',
          flush=True)


# ------------------------- mode: heldout -------------------------

def softmax_mix(lib, c_q, tau=1.0):
    """c_mix = normalize(Sum_j w'_j m_j), w'_j = (w_j/||m_j||)/Sum (norm-cor)."""
    cos = F.cosine_similarity(c_q.unsqueeze(0), lib['c'], dim=-1)
    w = torch.softmax(cos / tau, dim=-1)
    wp = w / lib['m'].norm(dim=-1)
    wp = wp / wp.sum()
    return F.normalize((wp.unsqueeze(-1) * lib['m']).sum(0), dim=-1), w


def top2_on_edge(lib, q):
    cos = F.cosine_similarity(lib['held_c'][q].unsqueeze(0), lib['c'], dim=-1)
    top2 = cos.topk(2).indices
    edge_set = {tuple(e) for e in lib['edges']}
    pair = tuple(sorted(top2.tolist()))
    return top2, pair in edge_set


def mode_heldout(snap, out_json, n_eps=20, scan_eps=10, scan_alphas=(0, .25, .5, .75, 1)):
    agent = load_agent(snap)
    lib = load_lib()
    env = make_env()
    results = {'snapshot': snap, 'prompts': []}
    rng = np.random.RandomState(0)
    for q in range(8):
        c_q = lib['held_c'][q]
        S_q = lib['held_S'][q]
        d_cq = float(1.0 - F.cosine_similarity(
            c_q.unsqueeze(0), lib['c'], dim=-1).max())
        top2, on_edge = top2_on_edge(lib, q)
        a, b = int(top2[0]), int(top2[1])

        # arms
        c_mix, w_all = softmax_mix(lib, c_q)
        c_near = lib['c'][a]
        if not on_edge:            # Top-1 fallback: mix restricted to j*
            c_mix = c_near
        e_pair = rng.choice(16, 2, replace=False)   # two distinct anchors
        alpha_r = rng.rand()
        c_rand = F.normalize(
            alpha_r * lib['m'][e_pair[0]] + (1 - alpha_r) * lib['m'][e_pair[1]],
            dim=-1)

        arms = {}
        for name, cond in [('direct', c_q), ('mix', c_mix),
                           ('nearest', c_near), ('random', c_rand)]:
            rew, desc, _ = run_eps(agent, env, cond, n_eps, S_q)
            arms[name] = {'mean_reward': rew, 'descriptor': desc}
            print(f'  [q{q}] {name}: rew={rew:+.2f} desc={desc:+.3f}',
                  flush=True)

        # alpha-scan on the top-2 train edge
        scan_desc, scan_rew = [], []
        for al in scan_alphas:
            cond = F.normalize(
                al * lib['m'][a] + (1 - al) * lib['m'][b], dim=-1)
            rew, desc, _ = run_eps(agent, env, cond, scan_eps, S_q)
            scan_desc.append(desc)
            scan_rew.append(rew)
        amax = int(np.argmax(scan_desc))
        viol = 0
        for t in range(amax):
            viol += int(scan_desc[t] > scan_desc[t + 1] + 1e-6)
        for t in range(amax, len(scan_desc) - 1):
            viol += int(scan_desc[t] < scan_desc[t + 1] - 1e-6)
        monotone = viol <= 1

        results['prompts'].append({
            'q': q, 'prompt': lib['heldout_prompts'][q],
            'd_cq': d_cq, 'top2': [a, b], 'top2_on_edge': on_edge,
            'arms': arms,
            'scan': {'alphas': list(scan_alphas), 'descriptor': scan_desc,
                     'reward': scan_rew},
            'attribution_win': bool(arms['mix']['descriptor']
                                    > arms['direct']['descriptor']),
            'interp_win': bool(arms['mix']['descriptor']
                               > arms['nearest']['descriptor']
                               and arms['mix']['descriptor']
                               > arms['random']['descriptor']),
            'monotone': monotone,
            'coverage_ok': on_edge,
        })

    att = sum(p['attribution_win'] for p in results['prompts'])
    itp = sum(p['interp_win'] for p in results['prompts'])
    mono = sum(p['monotone'] for p in results['prompts'])
    cov = sum(p['coverage_ok'] for p in results['prompts'])
    results['gates'] = {
        'attribution': f'{att}/8', 'interpolation': f'{itp}/8',
        'monotone_tracking': f'{mono}/8', 'coverage': f'{cov}/8',
        'attribution_pass': att >= 6, 'interpolation_pass': itp >= 6,
        'monotone_pass': mono >= 6, 'coverage_pass': cov >= 6,
        'all_pass': att >= 6 and itp >= 6 and mono >= 6 and cov >= 6,
    }
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'HELDOUT DONE gates={results["gates"]}', flush=True)


# ------------------------- mode: genrl -------------------------

def mode_genrl(snap, out_json, n_eps=20):
    agent = load_agent(snap)
    env = make_env()
    rews = []
    for _ in range(n_eps):
        r, _ = rollout(agent, env, cond=None)
        rews.append(r)
    out = {'snapshot': snap, 'mean_reward': float(np.mean(rews)),
           'reward_std': float(np.std(rews)),
           'episode_rewards': rews}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'GENRL DONE rew={out["mean_reward"]:+.2f}+-{out["reward_std"]:.2f}',
          flush=True)


# ------------------------- mode: refdesc -------------------------

def mode_refdesc(snap, q, out_json, n_eps=20):
    """R030 reference metric: descriptor achievement of a GenRL prompt-
    trained policy vs the prompt's OWN held-out anchor trajectory (same
    descriptor definition as the four-way attribution arms)."""
    agent = load_agent(snap)
    lib = load_lib()
    env = make_env()
    rews, descs = [], []
    for _ in range(n_eps):
        r, gs = rollout(agent, env, cond=None)
        rews.append(r)
        if gs.shape[0] > 0:
            descs.append(descriptor(gs, lib['held_S'][q]))
    out = {'snapshot': snap, 'q': int(q),
           'prompt': lib['heldout_prompts'][q],
           'mean_reward': float(np.mean(rews)),
           'descriptor': float(np.mean(descs)),
           'episode_rewards': rews}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'REFDESC DONE q={q} rew={out["mean_reward"]:+.2f} '
          f'desc={out["descriptor"]:+.3f}', flush=True)


# ------------------------- mode: aggregate -------------------------

def _load_json(p):
    with open(p) as f:
        return json.load(f)


def paired_ci(rel_diffs):
    """Paired 95% CI (t, n-1 df) on relative differences."""
    rel_diffs = np.asarray(rel_diffs)
    n = len(rel_diffs)
    if n < 2:
        return float('nan'), float('nan'), float('nan')
    m = rel_diffs.mean()
    se = rel_diffs.std(ddof=1) / np.sqrt(n)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - tcrit * se), float(m + tcrit * se)


def aggregate_m2(res_dir):
    """B2 gates: descriptor wins vs shuffled, ID recovery, DIAYN paired CI."""
    out = {'gate_descriptor_12of16': False, 'gate_id_25pct': False,
           'gate_diayn_ci': False, 'all_pass': False}
    off_desc = []      # [seed][anchor]
    diayn_desc = []    # [seed][anchor]
    shuff_desc = []    # [seed][anchor]
    for s in (1, 2, 3, 4, 5):
        off_desc.append([a['descriptor'] for a in _load_json(
            f'{res_dir}/m2_anchor_off_s{s}_train16.json')['anchors']])
        diayn_desc.append([a['descriptor'] for a in _load_json(
            f'{res_dir}/m2_diayn_s{s}_train16.json')['anchors']])
    for s in (11, 12, 13):
        shuff_desc.append([a['descriptor'] for a in _load_json(
            f'{res_dir}/m2_shuffle_s{s}_train16.json')['anchors']])
    off_desc = np.asarray(off_desc)          # [5, 16]
    diayn_desc = np.asarray(diayn_desc)      # [5, 16]
    shuff_desc = np.asarray(shuff_desc)      # [3, 16]

    # (a) >=12/16 anchors: mean anchor-only > mean shuffled
    wins = off_desc.mean(0) > shuff_desc.mean(0)
    out['descriptor_wins'] = int(wins.sum())
    out['descriptor_win_mask'] = [bool(w) for w in wins]
    out['gate_descriptor_12of16'] = bool(wins.sum() >= 12)

    # (b) ID recovery >= 25% (per seed, all seeds)
    id_accs = []
    for s in (1, 2, 3, 4, 5):
        id_accs.append(_load_json(f'{res_dir}/m2_anchor_off_s{s}_idrec.json')
                       ['step_accuracy'])
    out['id_accuracies'] = id_accs
    out['gate_id_25pct'] = bool(all(a >= 0.25 for a in id_accs))

    # (c) paired 95% CI excludes DIAYN >10% relative descriptor advantage
    anchor_mean = off_desc.mean(0)                 # per-anchor [16]
    d = diayn_desc - off_desc                      # [5, 16]
    rel = d.mean(1) / np.maximum(anchor_mean.mean(), 1e-9)
    m, lo, hi = paired_ci(rel)
    out['diayn_rel_mean_ci'] = [m, lo, hi]
    out['gate_diayn_ci'] = bool(hi < 0.10)
    # also report per-anchor relative diffs
    out['per_anchor_rel_diff'] = {
        str(i): float(d[:, i].mean() / max(anchor_mean[i], 1e-9))
        for i in range(16)}
    out['all_pass'] = (out['gate_descriptor_12of16'] and out['gate_id_25pct']
                       and out['gate_diayn_ci'])
    return out


def aggregate_m34(res_dir):
    """B3+B4: mixup on-vs-off interpolation comparison + four-way gates
    (gates on per-prompt means across the 5 mixup-on policies)."""
    out = {'gate_m3_mixup_on_beats_off': False,
           'gate_m3_off_fails_monotone': False,
           'gates_b4': {}, 'all_pass': False}
    on_prompts = []    # [seed][q]
    off_prompts = []   # [seed][q]
    for s in (1, 2, 3, 4, 5):
        on_prompts.append(_load_json(
            f'{res_dir}/m3_mixon_s{s}_heldout.json')['prompts'])
        off_prompts.append(_load_json(
            f'{res_dir}/m2_anchor_off_s{s}_heldout.json')['prompts'])

    def arm_desc(prompts, q, arm):
        return np.asarray([p[q]['arms'][arm]['descriptor']
                           for p in prompts])

    # B3: on vs off on the c_mix arm (per-prompt means, paired seeds)
    on_mix_per_seed = np.asarray(
        [[on_prompts[s][q]['arms']['mix']['descriptor'] for q in range(8)]
         for s in range(5)])          # [5, 8]
    off_mix_per_seed = np.asarray(
        [[off_prompts[s][q]['arms']['mix']['descriptor'] for q in range(8)]
         for s in range(5)])
    d_seed = (on_mix_per_seed - off_mix_per_seed).mean(1)   # [5]
    m, lo, hi = paired_ci(d_seed)
    out['m3_on_minus_off_mix_mean_ci'] = [m, lo, hi]
    out['gate_m3_mixup_on_beats_off'] = bool(lo > 0)

    # monotone tracking: per prompt majority over seeds; on passes, off fails
    on_mono = np.asarray([sum(
        on_prompts[s][q]['monotone'] for s in range(5)) >= 3
        for q in range(8)])
    off_mono = np.asarray([sum(
        off_prompts[s][q]['monotone'] for s in range(5)) >= 3
        for q in range(8)])
    out['m3_on_monotone'] = f'{int(on_mono.sum())}/8'
    out['m3_off_monotone'] = f'{int(off_mono.sum())}/8'
    out['gate_m3_off_fails_monotone'] = bool(off_mono.sum() < 6)
    out['gate_m3_on_passes_monotone'] = bool(on_mono.sum() >= 6)

    # B4 four-way gates on per-prompt MEAN descriptors over the 5 on-seeds
    att = sum(1 for q in range(8)
              if arm_desc(on_prompts, q, 'mix').mean()
              > arm_desc(on_prompts, q, 'direct').mean())
    itp = sum(1 for q in range(8)
              if arm_desc(on_prompts, q, 'mix').mean()
              > arm_desc(on_prompts, q, 'nearest').mean()
              and arm_desc(on_prompts, q, 'mix').mean()
              > arm_desc(on_prompts, q, 'random').mean())
    mono = int(on_mono.sum())
    cov = sum(1 for q in range(8)
              if all(on_prompts[s][q]['coverage_ok'] for s in range(5)))
    out['gates_b4'] = {
        'attribution': f'{att}/8', 'interpolation': f'{itp}/8',
        'monotone': f'{mono}/8', 'coverage': f'{cov}/8',
        'attribution_pass': att >= 6, 'interpolation_pass': itp >= 6,
        'monotone_pass': mono >= 6, 'coverage_pass': cov >= 6,
    }
    out['all_pass'] = (out['gate_m3_mixup_on_beats_off']
                       and out['gate_m3_off_fails_monotone']
                       and all(out['gates_b4'][k] for k in (
                           'attribution_pass', 'interpolation_pass',
                           'monotone_pass', 'coverage_pass')))
    return out


def mode_aggregate(res_dir, out_json, task):
    if task == 'm2':
        out = aggregate_m2(res_dir)
    elif task in ('m3', 'm34'):
        out = aggregate_m34(res_dir)
    else:
        raise ValueError(task)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'AGGREGATE[{task}] -> {out_json}')
    print(json.dumps(out, indent=2))
    print(f'ALL_PASS={out.get("all_pass", "n/a")}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train16', 'idrecover', 'heldout', 'genrl',
                                     'refdesc', 'aggregate'])
    ap.add_argument('snapshot')
    ap.add_argument('out_json')
    ap.add_argument('--n_eps', type=int, default=20)
    ap.add_argument('--scan_eps', type=int, default=10)
    ap.add_argument('--q', type=int, default=0)
    ap.add_argument('--task', default='m2', choices=['m2', 'm3', 'm34'])
    args = ap.parse_args()
    os.chdir(REPO)
    if args.mode == 'train16':
        mode_train16(args.snapshot, args.out_json, args.n_eps)
    elif args.mode == 'idrecover':
        mode_idrecover(args.snapshot, args.out_json)
    elif args.mode == 'heldout':
        mode_heldout(args.snapshot, args.out_json, args.n_eps,
                     scan_eps=args.scan_eps)
    elif args.mode == 'genrl':
        mode_genrl(args.snapshot, args.out_json, args.n_eps)
    elif args.mode == 'refdesc':
        mode_refdesc(args.snapshot, args.q, args.out_json, args.n_eps)
    elif args.mode == 'aggregate':
        mode_aggregate(args.snapshot, args.out_json, args.task)


if __name__ == '__main__':
    main()
