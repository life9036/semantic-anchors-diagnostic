"""Pilot final evaluation v2 — connector-free.

Loads each run snapshot DIRECTLY via torch.load (no Workspace indirection),
runs env eval (train.py eval_imag_behavior replication, 8 episodes) and
computes behavioral metrics from the recorded pixel rollouts:
  - episode reward (task success)
  - cross-policy pixel diversity: mean pairwise (1 - cosine) of rollout
    observation means — behavioral dissimilarity
  - within-policy coverage: mean pairwise L2 of per-episode obs means

Config: exp/final_eval_cfg.json {"snapshots": [...], "anchor_specs": [...],
  "num_eps": 8, "out": "exp/pilot_final_eval.json"}
"""
import os
import json
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MKL_SERVICE_FORCE_INTEL', '1')
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import numpy as np
import torch

import tools.utils as utils
import envs.main as envs

FRAME_EVERY = 4  # subsample frames for pixel metrics


def main():
    os.chdir(os.path.expanduser('~/autodl-tmp/genrl'))
    pcfg = json.load(open(os.environ.get('FINAL_EVAL_CFG', 'exp/final_eval_cfg.json')))
    snaps, specs = pcfg['snapshots'], pcfg['anchor_specs']
    num_eps = pcfg['num_eps']

    with hydra.initialize(config_path='..'):  # pilots/ -> repo root
        base = hydra.compose('train', overrides=[
            'task=stickman_walk', 'use_wandb=false', 'use_tb=false',
            'train_from_data=false'])

    rows = []
    for si, (snap, spec) in enumerate(zip(snaps, specs)):
        payload = torch.load(snap, map_location='cpu', weights_only=False)
        agent = payload.get('agent')
        if agent is None and not isinstance(payload, dict):
            agent = payload
        assert agent is not None, f'no agent in {snap}'
        agent = agent.to('cuda')
        for m in [agent] + list(agent.modules()):
            if hasattr(m, 'device') and isinstance(m.device, str):
                m.device = 'cuda'
        agent._backup_acting_behavior = agent._acting_behavior
        agent._acting_behavior = agent._imag_behavior

        env = envs.make('stickman_walk', 'pixels', 2, 1,
                        img_size=base.img_size,
                        viclip_encode=base.viclip_encode,
                        clip_hd_rendering=base.clip_hd_rendering)
        meta = agent.init_meta()
        ep_rewards, ep_frames = [], []
        for ep in range(num_eps):
            time_step, dreamer_obs = env.reset()
            agent_state = None
            ep_r = 0.0
            frames = []
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action, agent_state = agent.act(
                        dreamer_obs, meta, 0, eval_mode=True, state=agent_state)
                time_step, dreamer_obs = env.step(action)
                ep_r += time_step.reward or 0.0
                if len(frames) % FRAME_EVERY == 0:
                    frames.append(dreamer_obs['observation'].astype(np.float32) / 255.0)
            ep_rewards.append(ep_r)
            ep_frames.append(np.stack(frames, 0))
        del env
        agent._acting_behavior = agent._backup_acting_behavior

        # per-episode mean frame (grayscale mean to keep it cheap)
        obs_means = np.stack(
            [f.mean(axis=(0, 3)) for f in ep_frames])  # [eps, H, W]
        flat = obs_means.reshape(len(obs_means), -1)
        rows.append({
            'snapshot': snap, 'anchor_spec': spec,
            'mean_episode_reward': float(np.mean(ep_rewards)),
            'reward_std': float(np.std(ep_rewards)),
            'episode_rewards': [float(r) for r in ep_rewards],
            'obs_mean_flat': flat,
        })
        print(f'eval {spec}: rew={np.mean(ep_rewards):+.2f}±{np.std(ep_rewards):.2f}',
              flush=True)

    n = len(rows)
    div = np.zeros((n, n))
    for i in range(n):
        mi = rows[i]['obs_mean_flat'].mean(0)
        for j in range(n):
            mj = rows[j]['obs_mean_flat'].mean(0)
            div[i, j] = float(1.0 - (mi @ mj) / (
                np.linalg.norm(mi) * np.linalg.norm(mj) + 1e-9))
    coverage = []
    for i in range(n):
        f = rows[i]['obs_mean_flat']
        d = [np.linalg.norm(f[a] - f[b])
             for a in range(len(f)) for b in range(a + 1, len(f))]
        coverage.append(float(np.mean(d)) if d else 0.0)

    out = {
        'policies': [{'snapshot': r['snapshot'], 'anchor_spec': r['anchor_spec'],
                      'mean_episode_reward': r['mean_episode_reward'],
                      'reward_std': r['reward_std'],
                      'episode_rewards': r['episode_rewards'],
                      'coverage': coverage[i]} for i, r in enumerate(rows)],
        'pairwise_diversity': div.tolist(),
    }
    with open(pcfg['out'], 'w') as f:
        json.dump(out, f, indent=2)
    print(f'FINAL EVAL DONE -> {pcfg["out"]}', flush=True)


if __name__ == '__main__':
    main()
