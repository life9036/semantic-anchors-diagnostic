#!/usr/bin/env python
"""aggregate_permuted.py — M5 Walker random-code + permuted-target aggregation.

Frozen reporting spec: review-stage/M5_PROTOCOL_REVISION.md §2 (G1-G4).
Reads refine-logs/results_archive/m5_permuted/ (m5base_randperm_s1-3) and
pairs against the existing Walker random-code baselines (main_results_v3/
m4base_random_s1-3) and the anchor condition (m2_anchor_off_s1-5, matched
s1-3 for the paired contrasts). Writes m5_permuted/aggregate_m5.json.
"""
import json, math
from scipy import stats

M5 = "refine-logs/results_archive/m5_permuted"
V3 = "refine-logs/results_archive/main_results_v3"
OUT = f"{M5}/aggregate_m5.json"
T_975_DF2 = stats.t.ppf(0.975, df=2)  # 4.302653


def load(p):
    return json.load(open(p, encoding="utf-8"))


def per_run(tag, base):
    tr = load(f"{base}/{tag}_train16v2.json")
    ids = load(f"{base}/{tag}_idrec_v2.json")
    return {
        "kin_diag": tr["confusion_kinematic_diag"],
        "g_diag_rate": tr["confusion_g_diag_rate"],
        "id_step": ids["step_accuracy"],
        "id_ep": ids["episode_accuracy"],
        "reward": float(sum(a["mean_reward"] for a in tr["anchors"]) / len(tr["anchors"])),
        "seeded_paired_ok": tr.get("seeded_paired_ok"),
    }


def paired(series_a, series_b, names):
    diffs = [a - b for a, b in zip(series_a, series_b)]
    n = len(diffs)
    mean = float(sum(diffs) / n)
    sd = float(math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1)))
    ci = [mean - T_975_DF2 * sd / math.sqrt(n), mean + T_975_DF2 * sd / math.sqrt(n)]
    return {
        "pair": names, "n": n, "df": 2,
        "per_seed_diffs": [round(x, 4) for x in diffs],
        "mean_diff": round(mean, 4), "ci95": [round(x, 4) for x in ci],
    }


def main():
    randperm = [f"m5base_randperm_s{i}" for i in (1, 2, 3)]
    random = [f"m4base_random_s{i}" for i in (1, 2, 3)]
    anchor_matched = [f"m2_anchor_off_s{i}" for i in (1, 2, 3)]
    anchor_all = [f"m2_anchor_off_s{i}" for i in (1, 2, 3, 4, 5)]
    shuffle = [f"m2_shuffle_s{i}" for i in (11, 12, 13)]

    runs = {}
    for t in randperm:
        runs[t] = per_run(t, M5)
    for t in random + anchor_all + shuffle:
        runs[t] = per_run(t, V3)

    # cross-check the archived random-code baselines against the paper
    print("random baseline kin diags:", [round(runs[t]["kin_diag"], 4) for t in random])
    print("anchor 5-seed kin diags:", [round(runs[t]["kin_diag"], 4) for t in anchor_all])
    print("shuffle 3-seed kin diags:", [round(runs[t]["kin_diag"], 4) for t in shuffle])

    eps = ["kin_diag", "g_diag_rate", "id_step", "id_ep", "reward"]
    contrasts = {}
    for e in eps:
        contrasts[e] = {
            "randperm_vs_random": paired([runs[t][e] for t in randperm],
                                         [runs[t][e] for t in random],
                                         "m5 randperm - m4 random"),
            "randperm_vs_anchor": paired([runs[t][e] for t in randperm],
                                         [runs[t][e] for t in anchor_matched],
                                         "m5 randperm - anchor (matched s1-3)"),
        }

    agg = {
        "frozen_spec": "M5_PROTOCOL_REVISION.md §2",
        "per_run": {t: {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in runs[t].items()}
                    for t in randperm + random + anchor_all + shuffle},
        "contrasts": contrasts,
        "descriptive_summary": {
            "randperm_kin_mean": round(sum(runs[t]["kin_diag"] for t in randperm) / 3, 4),
            "random_kin_mean": round(sum(runs[t]["kin_diag"] for t in random) / 3, 4),
            "anchor_kin_mean_5": round(sum(runs[t]["kin_diag"] for t in anchor_all) / 5, 4),
            "shuffle_kin_mean": round(sum(runs[t]["kin_diag"] for t in shuffle) / 3, 4),
            "reference": round(1 / 16, 4),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print("wrote", OUT)
    print(json.dumps(agg["descriptive_summary"], indent=2))


if __name__ == "__main__":
    main()
