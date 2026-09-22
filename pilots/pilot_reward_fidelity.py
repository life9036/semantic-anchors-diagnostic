"""Pilot P1 (triage A23): VLM cosine-reward fidelity diagnostic.

Question: does the GenRL stepwise cosine reward track ground-truth task
success, and how does that correlation degrade under controlled temporal
target shifts and time-warp "hacking"?

Method (no RL training):
  1. Restore the pretrained stickman snapshot (WM + connector + aligner).
  2. Sample trajectory chunks from the offline stickman replay dataset and
     push them through the WM (agent.wm.observe_data) to get latent seqs.
  3. For each prompt in {walk, run, stand, backwards}: build the anchor target
     latent sequence via the frozen connector, then compute per-chunk summed
     cosine reward under target shifts k in {-8, -4, 0, +4, +8}.
  4. Ground truth: summed environment reward recorded in the replay chunk.
  5. Report Spearman rho(reward, success) per (prompt, shift) cell, plus the
     time-warp hack gap: mean reward gain from time-warping trajectories
     (early target hit) without any change in environment success.
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
import torch.nn.functional as F
from scipy.stats import spearmanr
from dm_env import specs

from train import Workspace
from tools.replay import ReplayBuffer, make_replay_loader

PROMPTS = {
    'walk':      'robot walk fast clean',
    'run':       'robot run fast clean',
    'stand':     'standing up',
    'backwards': 'robot walking backwards',
}
SHIFTS = [-8, -4, 0, 4, 8]
N_BATCHES = 8   # batch_size=32 chunks per batch -> 256 chunks


def build_target(ws, prompt, T, B):
    wm = ws.agent.wm
    if hasattr(wm, 'viclip_model') and wm.viclip_model is not None:
        clip = wm.viclip_model
    else:
        from tools.genrl_utils import viclip_global_instance
        if not viclip_global_instance._instantiated:
            viclip_global_instance.instantiate()
        clip = viclip_global_instance.viclip
    with torch.no_grad():
        fe = clip.get_txt_feat(prompt).to(ws.device)
        ve = fe.reshape(1, 1, -1).repeat(B, T, 1)
        stats = wm.connector.video_imagine(
            ve, dreamer_init=None, sample=False,
            reset_every_n_frames=False, denoise=True)
    return {k: v.permute([1, 0] + list(range(2, len(v.shape))))
            for k, v in stats.items()}


def stepwise_cos(ws, agent_seq, target_seq):
    tstoch = ws.agent.wm.connector.get_stoch(target_seq)
    tconv = ws.agent.wm.heads['decoder']._conv_in[0](tstoch)
    aconv = ws.agent.wm.heads['decoder']._conv_in[0](
        ws.agent.wm.rssm.get_stoch(agent_seq))
    return F.cosine_similarity(tconv, aconv, dim=-1)


def main():
    os.chdir(os.path.expanduser('~/autodl-tmp/genrl'))
    with hydra.initialize(config_path='..'):  # pilots/ -> repo root (hydra 1.1: must be relative)
        cfg = hydra.compose('train', overrides=[
            'task=stickman_walk', 'use_wandb=false', 'use_tb=false',
            'train_from_data=true', 'replay_load_dir=data/stickman_example',
            'snapshot_load_dir=models/genrl_stickman_500k_2.pt'])
    cfg.workdir = os.getcwd()
    ws = Workspace(cfg)
    agent = ws.agent

    data_specs = (ws.train_env.obs_space, ws.train_env.act_space,
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))
    buffer = ReplayBuffer(data_specs, [], 'data/stickman_example',
                          length=cfg.batch_length, **cfg.replay,
                          device=cfg.device, ignore_extra_keys=True,
                          load_recursive=True)
    loader = make_replay_loader(buffer, cfg.batch_size)

    all_success, all_reward = [], {p: {k: [] for k in SHIFTS} for p in PROMPTS}
    for bi in range(N_BATCHES):
        data = next(iter(loader))
        with torch.no_grad():
            outputs, _ = agent.wm.observe_data(data)
        seq = outputs['post'] if 'post' in outputs else outputs
        T, B = seq['deter'].shape[:2]
        success = data['reward'].sum(0).squeeze(-1).cpu().numpy()
        all_success.append(success)
        for pname, prompt in PROMPTS.items():
            target = build_target(ws, prompt, T, B)
            for k in SHIFTS:
                # shift the TARGET sequence before alignment: reward_t = cos(s_t, S_{t+k})
                t_k = target if k == 0 else {
                    kk: torch.roll(v, shifts=k, dims=0) for kk, v in target.items()}
                shifted = stepwise_cos(ws, seq, t_k)  # [T, B]
                all_reward[pname][k].append(shifted.sum(0).cpu().numpy())
        print(f'P1: batch {bi + 1}/{N_BATCHES} done', flush=True)

    all_success = np.concatenate(all_success)
    summary = {}
    for pname in PROMPTS:
        for k in SHIFTS:
            r = np.concatenate(all_reward[pname][k])
            res = spearmanr(r, all_success)
            rho = float(getattr(res, 'statistic', res.correlation))
            summary[f'{pname}|shift{k}'] = {
                'spearman': rho,
                'mean_reward': float(r.mean()),
            }
            print(f'P1 {pname} shift={k:>3}: rho={rho:+.3f} '
                  f'mean_reward={r.mean():.3f}', flush=True)
    summary['mean_success'] = float(all_success.mean())

    # time-warp hack gap: warp trajectory time (early hit + hold); success
    # cannot change because the replay chunk is fixed — reward gain is pure hack
    hack = {}
    for pname, prompt in PROMPTS.items():
        gains = []
        for bi in range(4):
            data = next(iter(loader))
            with torch.no_grad():
                outputs, _ = agent.wm.observe_data(data)
            seq = outputs['post'] if 'post' in outputs else outputs
            T, B = seq['deter'].shape[:2]
            target = build_target(ws, prompt, T, B)
            base = stepwise_cos(ws, seq, target)
            warp = 4
            idx = torch.minimum(torch.arange(T).repeat(B, 1) // warp,
                                torch.full((B, T), T - 1))
            warped = torch.gather(base, 0, idx.to(base.device))
            gains.append(float((warped.sum(0) - base.sum(0)).mean()))
        hack[pname] = {'mean_warp_gain': float(np.mean(gains))}
        print(f'P1 hack-gap {pname}: mean_warp_gain={np.mean(gains):+.3f}',
              flush=True)

    results = {'spearman_cells': summary, 'hack_gap': hack,
               'snapshot': 'models/genrl_stickman_500k_2.pt',
               'dataset': 'data/stickman_example'}
    os.makedirs('exp', exist_ok=True)
    out = 'exp/pilot_p1_reward_fidelity.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'P1 DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
