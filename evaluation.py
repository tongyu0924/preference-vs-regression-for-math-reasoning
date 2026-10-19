import os
from collections import Counter

import numpy as np


def step_level_metrics(scores_by_key, records, pairs):
    from scipy.stats import spearmanr
    key = lambda r: (r["qid"], r["depth"], r["step"])
    s = np.array([scores_by_key[key(r)] for r in records])
    q = np.array([r["q"] for r in records])
    pair_acc = (float(np.mean([scores_by_key[key(p)] > scores_by_key[key(n)] for p, n in pairs]))
                if pairs else float("nan"))
    rho = float(spearmanr(s, q).correlation) if len(set(q)) > 1 else float("nan")
    return dict(pairwise_acc=pair_acc, spearman_vs_Q=rho, mse_vs_Q=float(np.mean((s - q) ** 2)))


def majority_pick(sols):
    preds = [s["pred"] for s in sols if s["pred"] is not None]
    if not preds:
        return sols[0]
    top = Counter(preds).most_common(1)[0][0]
    return next(s for s in sols if s["pred"] == top)


def best_of_n(test_data, score_keys, ns, n_draws, rng):
    """Accuracy of each selection rule, averaged over random size-N subsets."""
    methods = ["random (pass@1)", "majority vote", "oracle (pass@N)"] + score_keys
    curves = {m: [] for m in methods}
    per_q = {m: [] for m in methods}
    for N in ns:
        acc = {m: [] for m in methods}
        for qd in test_data:
            sols = qd["solutions"]
            draws = 1 if N == len(sols) else n_draws
            for _ in range(draws):
                sub = [sols[i] for i in rng.choice(len(sols), size=N, replace=False)]
                acc["random (pass@1)"].append(np.mean([s["correct"] for s in sub]))
                acc["majority vote"].append(majority_pick(sub)["correct"])
                acc["oracle (pass@N)"].append(any(s["correct"] for s in sub))
                for k in score_keys:
                    acc[k].append(max(sub, key=lambda s: s[k])["correct"])
            if N == ns[-1]:
                for m in methods:
                    per_q[m].append(float(np.mean(acc[m][-draws:])))
        for m in methods:
            curves[m].append(float(np.mean(acc[m])))
    return curves, per_q


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = [np.mean(a[i] - b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return float(np.mean(a - b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def make_plots(ns, curves, step_metrics, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"PPM (last step)": ("#1b7f5f", "-", "o"), "PQM (last step)": ("#b5523b", "-", "s"),
             "PPM (min step)": ("#1b7f5f", ":", "o"), "PQM (min step)": ("#b5523b", ":", "s"),
             "majority vote": ("#5b6bbf", "--", "^"), "random (pass@1)": ("#888888", "--", "x"),
             "oracle (pass@N)": ("#bbbbbb", "-.", "")}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, ys in curves.items():
        c, ls, mk = style.get(m, ("black", "-", ""))
        ax.plot(ns, [100 * y for y in ys], ls, color=c, marker=mk, label=m, lw=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (sampled solutions)")
    ax.set_ylabel("GSM8K test accuracy (%)")
    ax.set_title("Best-of-N selection")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_of_n.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (k, title) in zip(axes, [("pairwise_acc", "Held-out pairwise acc. (higher)"),
                                     ("spearman_vs_Q", "Spearman vs. Q (higher)"),
                                     ("mse_vs_Q", "MSE vs. Q (lower)")]):
        vals = [step_metrics[n][k] for n in ["PPM", "PQM"]]
        ax.bar(["PPM", "PQM"], vals, color=["#1b7f5f", "#b5523b"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_metrics.png"), dpi=200)
    plt.close(fig)


def summary_text(res):
    sm, ns, cv = res["step_metrics"], res["best_of_n"]["N"], res["best_of_n"]["curves"]
    d, lo, hi = res["bootstrap_ppm_minus_pqm"]
    ds = res["data"]
    lines = ["", "=" * 70, "SUMMARY", "=" * 70,
             f"Steps: {ds['n_records']} | train pairs: {ds['n_pairs_train']} | "
             f"held-out pairs: {ds['n_pairs_heldout']}",
             f"Nodes with both Q>0 and Q<0 candidates: {ds['frac_mixed_nodes']:.1%}",
             f"Non-terminal Q-values equal to +-1: {ds['frac_q_extreme']:.1%}",
             "", f"{'Metric':34s}{'PPM':>10s}{'PQM':>10s}"]
    for k, lab in [("pairwise_acc", "Held-out pairwise accuracy"),
                   ("spearman_vs_Q", "Spearman vs. Q"), ("mse_vs_Q", "MSE vs. Q")]:
        lines.append(f"{lab:34s}{sm['PPM'][k]:10.3f}{sm['PQM'][k]:10.3f}")
    for agg in ["last step", "min step"]:
        lab = f"Best-of-{ns[-1]} acc. ({agg})"
        lines.append(f"{lab:34s}{100 * cv[f'PPM ({agg})'][-1]:9.1f}%{100 * cv[f'PQM ({agg})'][-1]:9.1f}%")
    lines += ["", "Baselines at N = %d: random %.1f%%, majority vote %.1f%%, oracle %.1f%%" % (
        ns[-1], 100 * cv["random (pass@1)"][-1], 100 * cv["majority vote"][-1],
        100 * cv["oracle (pass@N)"][-1]),
        f"PPM - PQM (Best-of-{ns[-1]}, last step): {100 * d:+.1f} pts, "
        f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]"]
    return "\n".join(lines)
