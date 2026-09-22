"""M0 preflight: integrity & compatibility hard gates (R000-R002).

R000 checkpoint integrity: SHA256 + param norms + action/trajectory hashes +
fresh re-evaluation of all existing checkpoints (6 stickman pilot runs + 3
pretrained WMs), duplicate-checkpoint control and random-policy control.
Adjudicates the reported posthoc-evaluator corruption (5 identical outputs).

R001 g-projection contract: decoder._conv_in exists on every WM, agent-stoch
and connector-stoch project to the same dim, L2 normalization is valid.

R002 pooling collision: build the Walker anchor library (16 train + 8 held-out
prompts), report ||m_j||, the m-Gram max off-diag cos, collisions (cos>0.95
with different descriptors), and cache the library for all downstream runs.

Gates PASS only if: no duplicate checkpoints, fresh evals discriminate, g
contract holds, no collisions. Any failure => do NOT start training.
"""
import os
import sys
import json
import hashlib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MKL_SERVICE_FORCE_INTEL', '1')
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import numpy as np
import torch
import torch.nn.functional as F

import tools.utils as utils
import envs.main as envs
from experiments.anchor_lib import build_library, g_proj, T
from experiments.prompts import TRAIN_PROMPTS, HELDOUT_PROMPTS

REPO = os.path.expanduser('~/autodl-tmp/genrl')
PILOT_SNAPS = [
    'exp/p2_r1_anchor_walk/last_snapshot.pt',
    'exp/p2_r2_anchor_walk_ent/last_snapshot.pt',
    'exp/p2_r3_entropy_only/last_snapshot.pt',
    'exp/p2_r4_anchor_run/last_snapshot.pt',
    'exp/p3_r5_delta_run_slow/last_snapshot.pt',
    'exp/p3_r6_delta_slow_run/last_snapshot.pt',
]
WM_SNAPS = [
    'models/genrl_stickman_500k_2.pt',
    'models/genrl_walker_500k_2.pt',
    'models/genrl_cheetah_500k_2.pt',
]
LIB_PATH = 'exp/anchor_lib/walker_lib.pt'
OUT_PATH = 'exp/preflight/preflight_report.json'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_agent(payload):
    if isinstance(payload, dict) and 'agent' in payload:
        return payload['agent']
    return payload


def param_norms(agent):
    out = {}
    targets = {
        'wm.encoder': agent.wm.encoder,
        'wm.rssm': agent.wm.rssm,
        'wm.decoder_conv_in': agent.wm.heads['decoder']._conv_in[0],
        'wm.connector': agent.wm.connector,
    }
    if hasattr(agent, '_imag_behavior'):
        targets['imag.actor'] = agent._imag_behavior.actor
        targets['imag.critic'] = agent._imag_behavior.critic
    for name, mod in targets.items():
        try:
            params = list(mod.parameters())
            out[name] = float(sum(p.data.abs().mean().item() for p in params)
                              / max(len(params), 1))
        except Exception as e:
            out[name] = f'ERR: {e}'
    return out


def rollout_hashes(agent, task, num_eps, seed=1):
    """Fresh env evaluation: episode rewards + action/trajectory hashes."""
    agent.to('cuda')
    for m in [agent] + list(agent.modules()):
        if hasattr(m, 'device') and isinstance(m.device, str):
            m.device = 'cuda'
    swap = (hasattr(agent, '_imag_behavior') and
            agent._acting_behavior is not agent._imag_behavior)
    if swap:
        agent._backup_acting_behavior = agent._acting_behavior
        agent._acting_behavior = agent._imag_behavior
    env = envs.make(task, 'pixels', 2, seed, img_size=64,
                    viclip_encode=False, clip_hd_rendering=False)
    meta = agent.init_meta()
    ep_rews, act_hash, traj_hash = [], hashlib.sha256(), hashlib.sha256()
    for _ in range(num_eps):
        time_step, dreamer_obs = env.reset()
        agent_state = None
        ep_r = 0.0
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action, agent_state = agent.act(
                    dreamer_obs, meta, 0, eval_mode=True, state=agent_state)
            time_step, dreamer_obs = env.step(action)
            ep_r += time_step.reward or 0.0
            act_hash.update(action.tobytes())
            traj_hash.update(
                dreamer_obs['observation'][::4].astype(np.uint8).tobytes())
        ep_rews.append(float(ep_r))
    del env
    if swap:
        agent._acting_behavior = agent._backup_acting_behavior
    agent.to('cpu')
    torch.cuda.empty_cache()
    return ep_rews, act_hash.hexdigest()[:16], traj_hash.hexdigest()[:16]


def random_control(task, num_eps, seed=1):
    env = envs.make(task, 'pixels', 2, seed, img_size=64,
                    viclip_encode=False, clip_hd_rendering=False)
    spec = env.act_space['action']
    low = np.asarray(getattr(spec, 'minimum', getattr(spec, 'low', -1.0)))
    high = np.asarray(getattr(spec, 'maximum', getattr(spec, 'high', 1.0)))
    rng = np.random.RandomState(0)   # seeded local RNG for reproducibility
    ep_rews, act_hash, traj_hash = [], hashlib.sha256(), hashlib.sha256()
    for _ in range(num_eps):
        time_step, dreamer_obs = env.reset()
        ep_r = 0.0
        while not time_step.last():
            action = rng.uniform(low, high).astype(np.float32)
            time_step, dreamer_obs = env.step(action)
            ep_r += time_step.reward or 0.0
            act_hash.update(action.tobytes())
            traj_hash.update(
                dreamer_obs['observation'][::4].astype(np.uint8).tobytes())
        ep_rews.append(float(ep_r))
    del env
    return ep_rews, act_hash.hexdigest()[:16], traj_hash.hexdigest()[:16]


def r001_g_contract(wm_payload, name):
    """g contract: conv_in exists, and the two REAL state paths the reward
    uses — rssm.initial() for agent states, connector.video_imagine() for
    anchor states — project to the same dim with valid L2 normalization."""
    wm = extract_agent(wm_payload).wm
    wm.to('cuda')
    for m in [wm] + list(wm.modules()):
        if hasattr(m, 'device') and isinstance(m.device, str):
            m.device = 'cuda'
    res = {}
    dec = wm.heads['decoder']
    if not hasattr(dec, '_conv_in'):
        res['fatal'] = 'decoder has no _conv_in'
        return res
    lin = dec._conv_in[0]
    res['g_in_dim'] = lin.in_features
    res['g_out_dim'] = lin.out_features
    res['agent_stoch_size'] = wm.rssm.get_stoch_size()
    res['connector_stoch_size'] = wm.connector.get_stoch_size()
    with torch.no_grad():
        # agent side: the imagination start-state path (rssm.initial)
        a_state = wm.rssm.initial(1)
        a_stoch = wm.rssm.get_stoch(a_state)          # [1, 1024]
        g_agent = g_proj(wm, a_stoch)                 # [1, d]
        # connector side: the anchor-library path (video_imagine with a
        # text-embed-shaped input, exactly as build_library does)
        ve = torch.randn(1, T, wm.connector.viclip_emb_dim,
                         device=wm.device)
        ve = F.normalize(ve, dim=-1)
        stats = wm.connector.video_imagine(
            ve, dreamer_init=None, sample=False,
            reset_every_n_frames=False, denoise=True)
        c_stoch = wm.connector.get_stoch(stats)       # [1, T, 1024]
        g_conn = g_proj(wm, c_stoch)                  # [1, T, d]
    res['g_agent_out_shape'] = list(g_agent.shape)
    res['g_conn_out_shape'] = list(g_conn.shape)
    res['g_norm_min'] = float(
        min(g_agent.norm(dim=-1).min(), g_conn.norm(dim=-1).min()))
    res['g_l2_normalized_ok'] = bool(
        (F.normalize(g_agent, dim=-1).norm(dim=-1) - 1).abs().max() < 1e-5
        and (F.normalize(g_conn, dim=-1).norm(dim=-1) - 1).abs().max() < 1e-5)
    res['contract_ok'] = bool(
        g_agent.shape[-1] == lin.out_features
        and g_conn.shape[-1] == lin.out_features
        and res['agent_stoch_size'] == res['connector_stoch_size']
        and res['g_norm_min'] > 1e-6 and res['g_l2_normalized_ok'])
    return res


def main():
    os.chdir(REPO)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    # resume support: PREFLIGHT_START=r001/r002 reuses previous sections
    start = os.environ.get('PREFLIGHT_START', 'r000')
    report = {'gates': {}, 'r000': {}, 'r001': {}, 'r002': {}}
    if start != 'r000' and os.path.exists(OUT_PATH):
        prev = json.load(open(OUT_PATH))
        for k in ('r000', 'r001', 'r002'):
            report[k] = prev.get(k, {})
    report['started_at_section'] = start

    # ---------- R000 ----------
    if start <= 'r000':
        print('=== R000 checkpoint integrity ===', flush=True)
        r000 = report['r000']
        r000['files'] = []
        seen_sha, seen_ahash = {}, {}
        for rel in PILOT_SNAPS + WM_SNAPS:
            path = os.path.join(REPO, rel)
            sha = sha256_file(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            agent = extract_agent(payload)
            entry = {'file': rel, 'sha256': sha[:16],
                     'size_mb': round(os.path.getsize(path) / 1e6, 1),
                     'param_norms': param_norms(agent),
                     'has_imag_behavior': hasattr(agent, '_imag_behavior'),
                     'has_connector': hasattr(agent.wm, 'connector')}
            if rel in PILOT_SNAPS:
                rews, ah, th = rollout_hashes(agent, 'stickman_walk', 5)
                entry.update(episode_rewards=rews, action_hash=ah, traj_hash=th)
                seen_sha.setdefault(sha, []).append(rel)
                seen_ahash.setdefault(ah, []).append(rel)
                print(f'  {rel}: rew={np.mean(rews):+.2f}+-{np.std(rews):.2f} '
                      f'ahash={ah} thash={th}', flush=True)
            else:
                print(f'  {rel}: sha={sha[:16]} size={entry["size_mb"]}MB',
                      flush=True)
            r000['files'].append(entry)
            del payload, agent
            torch.cuda.empty_cache()

        dup_sha = {k: v for k, v in seen_sha.items() if len(v) > 1}
        dup_ahash = {k: v for k, v in seen_ahash.items() if len(v) > 1}
        # corruption pattern of the old evaluator: ANY repeated reward-vector
        # signature (rounded) across checkpoints fails the gate
        fresh_rews = [f['episode_rewards'] for f in r000['files'][:6]]
        sigs = [tuple(round(r, 2) for r in rews) for rews in fresh_rews]
        sig_counts = {s: [r000['files'][:6][i]['file'] for i in range(6)
                          if sigs[i] == s] for s in set(sigs)}
        repeated_sigs = {k: v for k, v in sig_counts.items() if len(v) > 1}
        print(f'  duplicate SHA files: {dup_sha or "none"}', flush=True)
        print(f'  duplicate action-hash files: {dup_ahash or "none"}',
              flush=True)
        print(f'  repeated reward signatures: {repeated_sigs or "none"}',
              flush=True)

        rc_rews, rc_ah, rc_th = random_control('stickman_walk', 5)
        r000['random_control'] = {'episode_rewards': rc_rews,
                                  'action_hash': rc_ah, 'traj_hash': rc_th}
        rc_distinct = all(rc_ah != f['action_hash']
                          for f in r000['files'][:6])
        print(f'  random control: rew={np.mean(rc_rews):+.2f} ahash={rc_ah}; '
              f'distinct from all pilots: {rc_distinct}', flush=True)

        gate_r000 = (not dup_sha and not dup_ahash and not repeated_sigs
                     and rc_distinct)
        report['gates']['R000_checkpoint_integrity'] = bool(gate_r000)
        print(f'R000 gate: {"PASS" if gate_r000 else "FAIL"}', flush=True)

    # ---------- R001 ----------
    if start <= 'r001':
        print('=== R001 g projection contract ===', flush=True)
        for rel in WM_SNAPS:
            payload = torch.load(os.path.join(REPO, rel), map_location='cpu',
                                 weights_only=False)
            res = r001_g_contract(payload, rel)
            report['r001'][rel] = res
            print(f'  {rel}: {res}', flush=True)
            del payload
        gate_r001 = all(
            r.get('contract_ok', False) for r in report['r001'].values())
        report['gates']['R001_g_contract'] = bool(gate_r001)
        print(f'R001 gate: {"PASS" if gate_r001 else "FAIL"}', flush=True)

    # ---------- R002 ----------
    if start <= 'r002':
        print('=== R002 pooling collision (walker anchor library) ===',
              flush=True)
        with hydra.initialize(config_path='..'):  # experiments/ -> repo root
            cfg = hydra.compose('train', overrides=[
                'task=walker_walk', 'use_wandb=false', 'use_tb=false',
                'train_from_data=false',
                'snapshot_load_dir=%s' % os.path.join(REPO, WM_SNAPS[1]),
            ])
        cfg.workdir = REPO
        from train import Workspace
        ws = Workspace(cfg, workdir=Path(REPO))
        ws.root_dir = Path(REPO)
        agent = ws.agent
        lib = build_library(agent, os.path.join(REPO, LIB_PATH))
        m, c = lib['m'], lib['c']
        mnorms = m.norm(dim=-1)
        gram = c @ c.T
        off = gram.clone()
        off.fill_diagonal_(-1)
        maxcos = float(off.max())
        pair = tuple(int(x)
                     for x in np.unravel_index(int(off.argmax()), off.shape))
        collisions = [(int(i), int(j), round(float(gram[i, j]), 3))
                      for i in range(16) for j in range(i + 1, 16)
                      if float(gram[i, j]) > 0.95]
        s_norms = lib['S'].norm(dim=-1)
        # edge-mixture norms: the exact mixtures the alpha-scan / training uses
        mix_norms = []
        for (a, b) in lib['edges']:
            for al in (0.25, 0.5, 0.75):
                mix_norms.append(float(
                    F.normalize(al * m[a] + (1 - al) * m[b], dim=-1).norm()))
        mix_norm_min = min(mix_norms)
        # held-out diagnostics: d(c_q) = 1 - max_j cos(c_q, c_j); top-2 cover
        hc = lib['held_c']
        hsim = hc @ c.T
        dq = 1.0 - hsim.max(dim=-1).values
        top2 = hsim.topk(2, dim=-1).indices
        edge_set = {tuple(e) for e in lib['edges']}
        covered = [tuple(sorted(p.tolist())) in edge_set for p in top2]
        r002 = report['r002'] = {
            'm_norm_min/mean/max': [float(mnorms.min()), float(mnorms.mean()),
                                    float(mnorms.max())],
            'm_gram_max_offdiag_cos': maxcos,
            'm_gram_max_pair': list(pair),
            'collisions_cos_gt_0.95': collisions,
            'S_step_norm_min': float(s_norms.min()),
            'S_step_norm_max': float(s_norms.max()),
            'edge_mix_norm_min': mix_norm_min,
            'n_edges': len(lib['edges']),
            'edges': lib['edges'],
            'heldout_d_cq': [float(x) for x in dq],
            'heldout_top2_pairs': [p.tolist() for p in top2],
            'heldout_top2_covered_by_train_edges': [bool(x) for x in covered],
            'heldout_cover_gate_preview': f'{sum(covered)}/8',
            'lib_path': LIB_PATH,
            'prompts': {'train': lib['train_prompts'],
                        'heldout': lib['heldout_prompts']},
        }
        print(f'  ||m|| min/mean/max: {r002["m_norm_min/mean/max"]}',
              flush=True)
        print(f'  gram max off-diag cos: {maxcos:.3f} at {pair}', flush=True)
        print(f'  collisions: {collisions or "none"}', flush=True)
        print(f'  edge-mix norm min: {mix_norm_min:.3f}', flush=True)
        print(f'  edges: {lib["edges"]}', flush=True)
        print(f'  heldout d(c_q): {[round(float(x),2) for x in dq]}',
              flush=True)
        print(f'  heldout top-2 edge coverage: {sum(covered)}/8', flush=True)
        EPS = 1e-3
        gate_r002 = (len(collisions) == 0
                     and float(mnorms.min()) > EPS
                     and float(s_norms.min()) > 1 - 1e-3
                     and mix_norm_min > EPS)
        report['gates']['R002_no_collision'] = bool(gate_r002)
        print(f'R002 gate: {"PASS" if gate_r002 else "FAIL"}', flush=True)

    # ---------- verdict ----------
    gates = report['gates']
    report['ALL_GATES_PASS'] = bool(
        gates.get('R000_checkpoint_integrity', False)
        and gates.get('R001_g_contract', False)
        and gates.get('R002_no_collision', False))
    with open(OUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'PREFLIGHT DONE -> {OUT_PATH} '
          f'ALL_PASS={report["ALL_GATES_PASS"]}', flush=True)
    return 0 if report['ALL_GATES_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
