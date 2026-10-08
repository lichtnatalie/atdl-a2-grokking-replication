"""
Ablations 1 (decoder weight decay) and 2 (decoder learning rate). Compares P(RQI > 0.95 within 10^4 steps)
and steps-to-structure across weight-decay settings, against the theory's
P(lambda_3 > 0) from Fig. 4(a).

Usage:   python experiments/ablation_compare.py --which wd      (Ablation 1)
         python experiments/ablation_compare.py --which lrdec   (Ablation 2)
Inputs:  results/fig4b/runs_<tag>.csv; the baseline (wd 0, eta_dec 1e-4) is shared
Outputs: results/ablation_<which>/summary.csv, ablation_<which>.{pdf,png}
"""

import argparse

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from fig4a_critical_fraction import wilson  # noqa: E402

# (tag, label) in plotting order; the first entry gets the confidence band
ABLATIONS = {
    "wd": [("baseline", "weight decay 0 (released)"), ("wd1", "weight decay 1"),
           ("wd5", "weight decay 5")],
    "lrdec": [("baseline", r"learning rate $10^{-4}$ (released)"),
              ("lrdec1e-5", r"learning rate $10^{-5}$ (slower)"),
              ("lrdec1e-3", r"learning rate $10^{-3}$ (faster)")],
}
STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
GREY = "#8a8a85"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def crossing(r, p, level=0.5):
    """First r where p crosses `level`, by linear interpolation."""
    for i in range(1, len(p)):
        if p[i - 1] < level <= p[i]:
            return r[i - 1] + (level - p[i - 1]) * (r[i] - r[i - 1]) / (p[i] - p[i - 1])
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(ABLATIONS), default="wd")
    which = ap.parse_args().which
    out = os.path.join(ROOT, "results", f"ablation_{which}")
    os.makedirs(out, exist_ok=True)
    th = load(os.path.join(ROOT, "results", "fig4a", "fig4a_summary.csv"))

    rows, curves = [], {}
    for tag, label in ABLATIONS[which]:
        path = os.path.join(ROOT, "results", "fig4b", f"runs_{tag}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            if sum(1 for _ in f) < 2:  # header only: sweep not started yet
                continue
        d = load(path)
        sizes = np.unique(d["n_train"])
        r_list, p_list, lo_list, hi_list = [], [], [], []
        for n in sizes:
            m = d["n_train"] == n
            k, N = int(d["reached"][m].sum()), int(m.sum())
            lo, hi = wilson(k, N)
            succ = d["iter_rqi"][m & (d["reached"] == 1)]
            det = m & (d["dof"] == 2)
            rows.append(dict(setting=tag, n_train=int(n), fraction=round(n / 55, 4), runs=N,
                             reached=k, p_reached=round(k / N, 3), ci_lo=round(max(lo, 0.0), 3),
                             ci_hi=round(hi, 3),
                             reached_given_lambda3_pos=f"{int(d['reached'][det].sum())}/{int(det.sum())}",
                             median_steps_successes=float(np.median(succ)) if len(succ) else np.nan))
            r_list.append(n / 55); p_list.append(k / N); lo_list.append(lo); hi_list.append(hi)
        curves[tag] = tuple(map(np.array, (r_list, p_list, lo_list, hi_list)))
        det_all = d["dof"] == 2
        print(f"{tag}: runs={len(d['n_train'])}, 50% crossing r = "
              f"{crossing(*curves[tag][:2]):.3f}; dof>2 reached "
              f"{int(d['reached'][~det_all].sum())}/{int((~det_all).sum())}; "
              f"dof=2 reached {int(d['reached'][det_all].sum())}/{int(det_all.sum())}")

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.plot(th["fraction"], th["p_unique"], color=GREY, ls="--", lw=1.2,
            label=r"theory $P(\lambda_3>0)$")
    labels = dict(ABLATIONS[which])
    for i, (tag, _) in enumerate(ABLATIONS[which]):
        if tag not in curves:
            continue
        r, p, lo, hi = curves[tag]
        color, marker = STYLE[i]
        if i == 0:  # one band only, so the curves stay readable
            ax.fill_between(r, np.maximum(lo, 0), hi, color=color, alpha=0.15,
                            linewidth=0, label="95% CI (released)")
        ax.plot(r, p, marker=marker, color=color, ms=3.2, lw=1.2, label=labels[tag])
    ax.axvline(0.4, color=GREY, ls=":", lw=0.7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training data fraction $r$")
    ax.set_ylabel(r"$P$(RQI > 0.95 within $10^4$ steps)")
    # legend below the axes, so it never covers the theory curve
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"ablation_{which}.{ext}"), dpi=220)


if __name__ == "__main__":
    main()
