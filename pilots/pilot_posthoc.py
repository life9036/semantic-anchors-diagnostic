"""Pilot post-hoc analysis for P2 (triage A1: is MI necessary?) and
P3 (triage A4: verb-delta anchors).

Config: exp/posthoc_cfg.json written by the driver:
  {"snapshots": [abs paths], "anchor_specs": [...], "num_eps": 5, "out": "..."}

For each trained run snapshot:
  - evaluate the imagination policy in the real env (episode rewards)
  - extract per-episode latent state sequences from the agent's rssm state
  - semantic profile: mean cosine score of each policy's rollouts vs the four
    base anchor targets (walk/run/stand/backwards)
  - cross-policy diversity + within-policy coverage over latent means
"""
import os
import copy
import json
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MKL_SERVICE_FORCE_INTEL', '1')
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from train import Workspace
import tools.utils as utils
from pilots.pilot_anchor_rewards import anchor_cos_scores, PILOT_ANCHORS

BASE_ANCHORS = ['walk', 'run', 'stand', 'backwards']


def _state_stoch(state):
    # agent.act returns (action, state); state may be (rssm_dict, action) or dict
    if isinstance(state, (tuple, list)):
        for item in state:
            if isinstance(item, dict) and 'stoch' in item:
                return item['stoch']
        for item in state:
            if isinstance(item, torch.Tensor) and item.numel() > 16:
                return item
    if isinstance(state, dict):
        if 'stoch' in state:
            return state['stoch']
        return next(iter(state.values()))
    return state


def main():
    os.chdir(os.path.expanduser('~/autodl-tmp/genrl'))
    cfg_path = os.environ.get('POSTHOC_CFG', 'exp/posthoc_cfg.json')
    pcfg = json.load(open(cfg_path))
    snaps, specs = pcfg['snapshots'], pcfg['anchor_specs']
    assert len(snaps) == len(specs)

    with hydra.initialize(config_path='..'):  # pilots/ -> repo root (hydra 1.1: must be relative)
        base = hydra.compose('train', overrides=[
            'task=stickman_walk', 'use_wandb=false', 'use_tb=false',
            'train_from_data=false'])

    rows = []
    for snap, spec in zip(snaps, specs):
        c = copy.deepcopy(base)
        c.snapshot_load_dir = snap
        c.workdir = os.getcwd()
        ws = Workspace(c)  # loads the snapshot in __init__
        agent = ws.agent
        agent._backup_acting_behavior = agent._acting_behavior
        agent._acting_behavior = agent._imag_behavior
        meta = agent.init_meta()

        ep_rewards, ep_stoch = [], []
        for ep in range(pcfg['num_eps']):
            time_step, dreamer_obs = ws.eval_env.reset()
            agent_state = None
            ep_r = 0.0
            stoch_list = []
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action, agent_state = agent.act(
                        dreamer_obs, meta, 0, eval_mode=True, state=agent_state)
                time_step, dreamer_obs = ws.eval_env.step(action)
                ep_r += time_step.reward or 0.0
                stoch_list.append(_state_stoch(agent_state).detach().cpu())
            agent._acting_behavior = agent._backup_acting_behavior
            ep_rewards.append(ep_r)
            ep_stoch.append(torch.stack(stoch_list, 0)[:, None])  # [T,1,K,D]

        profile = {}
        with torch.no_grad():
            for s in ep_stoch:  # [T,1,K,D]; chunk to batch_length to bound memory
                T = s.shape[0]
                starts = range(0, max(T - 32 + 1, 1), 32) if T >= 32 else [0]
                for start in starts:
                    chunk = s[start:start + 32] if T >= 32 else s
                    seq = {'stoch': chunk.to(ws.device)}
                    scores = anchor_cos_scores(agent, seq, {
                        k: PILOT_ANCHORS[k] for k in BASE_ANCHORS})
                    for k, v in scores.items():
                        profile.setdefault(k, []).append(v.mean().item())
        profile = {k: float(np.mean(v)) for k, v in profile.items()}

        latent_means = np.stack(
            [s.mean(dim=(0, 2)).numpy().reshape(-1) for s in ep_stoch])
        rows.append({
            'snapshot': snap,
            'anchor_spec': spec,
            'mean_episode_reward': float(np.mean(ep_rewards)),
            'reward_std': float(np.std(ep_rewards)),
            'semantic_profile': profile,
            'latent_mean': latent_means,
        })
        print(f'posthoc {spec}: rew={np.mean(ep_rewards):+.2f} '
              f'profile={ {k: round(v, 2) for k, v in profile.items()} }',
              flush=True)

    n = len(rows)
    div = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            div[i, j] = float(1.0 - F.cosine_similarity(
                torch.as_tensor(rows[i]['latent_mean']),
                torch.as_tensor(rows[j]['latent_mean']), dim=-1).mean())
    for i in range(n):
        lm = rows[i]['latent_mean']
        d = [np.linalg.norm(lm[a] - lm[b])
             for a in range(len(lm)) for b in range(a + 1, len(lm))]
        rows[i]['coverage'] = float(np.mean(d)) if d else 0.0

    out = {
        'policies': [{k: r[k] for k in (
            'snapshot', 'anchor_spec', 'mean_episode_reward', 'reward_std',
            'semantic_profile', 'coverage')} for r in rows],
        'pairwise_diversity': div.tolist(),
    }
    with open(pcfg['out'], 'w') as f:
        json.dump(out, f, indent=2)
    print(f'POSTHOC DONE -> {pcfg["out"]}', flush=True)


if __name__ == '__main__':
    main()
