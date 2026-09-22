# Design & Pre-Registration — Neighborhood-Coupled Semantic Anchors

Implementation notes for `experiments/` (Workflow 1.5 bridge, 2026-08-31).
This file is the operational pre-registration: it freezes every definition
that the downstream gates depend on. Edit only through a new pre-registered
revision — after M0 passes, changes invalidate results.

## Space

- `g(x) = wm.heads['decoder']._conv_in[0](stoch)` — the frozen shared
  projection used by GenRL's cosine reward; pilot P1/P2 validated this space.
  Input: stoch (32 discrete x 32 classes -> 1024), output d = 1536.
- Anchor trajectory: `S_i in R^{T x d}`, T = imag_horizon = 16, built by
  `connector.video_imagine(text_embed(p_i), dreamer_init=None, sample=False,
  reset_every_n_frames=False, denoise=True)`, then per-step L2-normalized.
- `m_i = mean_t(S_i)` (unnormalized temporal mean); `c_i = normalize(m_i)`.

## Training (data-free imagination, H=16 phase-locked)

- 50k frames/run, batch 32 x 32, snapshot of the pretrained Walker WM
  (`models/genrl_walker_500k_2.pt`), actor/critic conditioned on
  cat(feat, c) — both pi and v take the condition.
- Per batch item: with prob `pure_prob` a pure anchor i~U(K); else an edge
  (a,b)~U(E) of the frozen symmetric top-2 train graph with alpha~U[0,1].
  `pure_prob = 0.5` (mixup-on) / `1.0` (mixup-off, discrete distribution).
- Condition: `c = normalize(alpha*m_a + (1-alpha)*m_b)` (pure: c_i).
  Reward target: `S_tgt(t) = normalize(alpha*S_a[t] + (1-alpha)*S_b[t])`
  (pure: S_i) — the consistent mixing contract.
- Reward: `r_t = max_{phi in [t-2,t+2] cap [1,16]} cos(g(s_t), S_tgt[phi])`,
  targets detached. Joint StreamNorm on |r| (momentum 1.0, scale 1.0).
- Shuffled control: condition c_i, reward target S_perm[i], perm = (i+5)%16
  (derangement), pure anchors only, 3 seeds (11-13).
- DIAYN variant (mixup-off only): `q_psi(i|s_t)` = 3x512 SiLU+LayerNorm MLP
  on the frozen latent state (deter+stoch, no condition), 16 logits; CE every
  4 actor steps on the same batch's imagined states (fully detached);
  `r_MI = log q - log(1/16)` with gradient flowing into the policy (standard
  DIAYN); combined reward `r = r_anchor + 1.0 * r_MI` normalized jointly.
  Discriminator optimizer: Adam lr 3e-5, eps 1e-5, clip 100, wd 1e-6.
- `agent.act` with a constant per-episode cond; cond-free callers get freshly
  sampled pure-anchor conds (warmup / plain eval only).

## Evaluation (offline, env seed 1)

- `descriptor_q(ep) = mean_t cos(g(s_t), S_q(min(t+1,16)))`, phase-locked.
- M2: train16 eval at 20 eps/anchor (320 eps/policy). Gates: (a) >=12/16
  anchors where anchor-only descriptor > shuffled descriptor (shuffled =
  mean over 3 seeds); (b) ID recovery: 16-class classifier (3x512 MLP, Adam
  1e-3, 50 epochs, 70/30 episode split) per-step accuracy >= 25% (chance
  6.25%); (c) paired 95% CI over 5 paired seeds excludes DIAYN >10% relative
  descriptor advantage; inactive anchor = descriptor <= 0.
- M3: held-out suite on BOTH branches; off branch at 10 eps/arm (budget),
  on branch at 20 eps/arm. Gate: mixup-on beats mixup-off on the c_mix arm
  (per-prompt means, paired 5 seeds, 95% CI lower bound > 0), and the off
  branch fails the monotone-tracking gate (per-prompt majority over seeds
  <6/8) while on passes.
- M4 gates (on the 5 mixup-on policies, per-prompt MEAN descriptors across
  seeds, >=6/8 each):
  - attribution: desc(c_mix) > desc(direct c_q) strictly (tie = fail);
  - interpolation: desc(c_mix) > desc(nearest) AND > desc(random interp);
  - monotone tracking: alpha-scan {0,.25,.5,.75,1} on the top-2 train edge is
    non-decreasing up to its argmax and non-increasing after (<=1 violation);
  - coverage: top-2 train anchors of q form an edge of the frozen train graph
    (else Top-1 fallback for c_mix and coverage fail for that prompt).
- Zero-shot retrieval: `w_j = softmax(cos(c_q, c_j)/1)`,
  `w'_j = (w_j/||m_j||) / sum_k(w_k/||m_k||)`, `c_mix = normalize(sum w'_j m_j)`
  over all 16 train anchors. Diagnostic: `d(c_q) = 1 - max_j cos(c_q, c_j)`.

## Registry (frozen)

- Train prompts: experiments/prompts.py TRAIN_PROMPTS (16, GenRL walker
  predicate set + direction/speed variants). Held-out: HELDOUT_PROMPTS (8).
- Neighborhood graph: per-anchor 2 nearest by cos(c_i, c_j) (excluding self),
  symmetrized; built once in preflight, cached in
  `exp/anchor_lib/walker_lib.pt`.

## Budget / disk

- Per run: 50k frames; snapshots at 25k/50k + last; after eval keep only
  last_snapshot.pt + snapshot_50000.pt (~1.1GB/run). Pilot scratch exp dirs
  are deleted after preflight (results JSONs archived locally first).
