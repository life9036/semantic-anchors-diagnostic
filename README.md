# Semantic Anchors or Random Codes? — Experiment Code

Code for the paper **"Semantic Anchors or Random Codes? A Two-Domain Diagnostic of Text-Conditioned Behavior Pretraining"** (submitted for review).

## What this is

A fork of the GenRL codebase (Mazzaglia et al., 2024; citation below), extended with the anchor-conditioning experiments and the prespecified diagnostic protocol reported in the paper:

- `experiments/anchor_lib.py`, `experiments/prompts.py` — semantic-anchor construction (frozen VLM features, checkpoint connector, codebook of 16 prompt-derived conditions)
- `experiments/anchor_patch.py` — method injection into the GenRL training loop (imagination training with phase-tolerant anchor-cosine rewards)
- `experiments/run_matrix.sh`, `run_cheetah.sh`, `run_cheetah_c4.sh` — Walker main matrix, Cheetah B5, and C4 representation-isolation runs
- `experiments/run_permuted.sh`, `aggregate_permuted.py` — M5 permuted-target control (post-outcome, exploratory)
- `experiments/eval_anchors_v2.py` — the paper's **v3 evaluator** (script name "v2" = evaluation generation "v3"): a post-hoc corrected reanalysis of the prespecified evaluator — leave-one-out centroids, aligned evaluation windows, literal Top-2, permutation-tail monotonicity test; hard-coded reset seeds make re-evaluation deterministic; corrected endpoints are treated as exploratory in the paper
- `experiments/DESIGN.md`, `experiments/genrl_anchor.yaml` — experiment design notes and Hydra config

## Reproducing the paper experiments

1. Environment: the `genrl` conda env (Python 3.10, PyTorch 2.2) used for all runs; see the paper appendix for the full protocol.
2. Checkpoints/snapshots: the 83 GB run snapshot archive is not stored in this repository (see the data release below); the Walker baseline snapshot is loaded via `snapshot_load_dir` in the run scripts.
3. Training runs are launched via the `experiments/run_*.sh` scripts (each documents its protocol in the header comment).
4. Evaluation is always `experiments/eval_anchors_v2.py` (the v3 evaluator, see above); corrected endpoints are flagged as exploratory in the paper.
5. Per-cohort aggregates (paper Tables I–III) come from the `aggregate_*.py` scripts over the per-run JSONs.

## Versions

| Tag | Date | Contents |
|---|---|---|
| `anchors-v2` | 2026-10-03 | Complete paper state: adds the M5 permuted-target control (`run_permuted.sh`, `aggregate_permuted.py`; advisor-R1 follow-up) |
| `anchors-v1` | 2026-09-22 | Frozen main-matrix state (Walker matrix, Cheetah B5/C4) |

Reproducing the paper: check out `anchors-v2` (branch `exp/anchors`). `master` retains the upstream demo state.

## Data

Per-run result JSONs, aggregation outputs, and the SHA-256 manifest (`RELEASE_MANIFEST.json`) are released separately — URL/DOI to be added at publication. Cohort naming matches the paper's evaluation generations: `main_results_v3/` holds the Walker v3 corrected-reanalysis per-run JSONs (the paper's claim source; v2 logs are archived but are not the claim source), `v2_aggregates/` the v2/v3/v4 aggregate lineage, `cheetah_c5/` and `cheetah_c4/` the Cheetah cohorts.

## Upstream

Based on GenRL (MIT license; `LICENSE` preserved):

> If you found our code useful, please refer to our work using the following citation:
>
> ```
> @inproceedings{Mazzaglia2024GenRL,
>     title={Gen{RL}: Multimodal-foundation world models for generalization in embodied agents},
>     author={Pietro Mazzaglia and Tim Verbelen and Bart Dhoedt and Aaron Courville and Sai Rajeswar},
>     booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
>     year={2024},
>     url={https://openreview.net/forum?id=za9Jx8yqUA}
> }
> ```
