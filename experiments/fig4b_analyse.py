"""
Analyse the Fig. 4(b) sweep (results/fig4b/runs_<tag>.csv).

Censoring: a run that never exceeds RQI 0.95 within the step budget has an
unknown time-to-structure that is larger than the budget. Medians are taken
with censored runs counted as +infinity, which is exact as long as fewer than
half the runs at that size are censored; otherwise the median is reported as
undefined (not plotted).

Panels:
  (a) steps to RQI > 0.95 vs training fraction r; censored runs at the cap.
  (b) fraction of runs reaching RQI > 0.95, next to the theory's P(lambda_3 > 0)
      from Fig. 4(a).
  (c) per run, mirroring the paper's Fig. 5(b): steps to RQI > 0.95 against
      lambda_3 for every run with lambda_3 > 0, failures marked. Slope only.
      (The per-size median version is still printed.) The theory
      predicts n_h = 1/(lambda_3 * eta) for gradient flow; training uses AdamW,
      so only the SCALING (log-log slope) is compared, not the constant.
"""

import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def censored_median(x, reached):
    x = np.where(reached == 1, x, np.inf)
    if np.mean(reached) <= 0.5:
        return np.nan
    return float(np.median(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    d = load(os.path.join(ROOT, "results", "fig4b", f"runs_{a.tag}.csv"))
    th = load(os.path.join(ROOT, "results", "fig4a", "fig4a_summary.csv"))
    cap = d["iter_rqi"].max()

    sizes = np.unique(d["n_train"])
    lines = ["n_train,fraction,runs,reached,frac_reached,median_steps,"
             "runs_dof2,reached_dof2,median_steps_dof2_given_reached,median_inv_lambda3_dof2"]
    agg = []
    for n in sizes:
        m = d["n_train"] == n
        r = d["fraction"][m][0]
        reached = d["reached"][m]
        med = censored_median(d["iter_rqi"][m], reached)
        u = m & (d["dof"] == 2)
        # conditional on success: median steps over runs that DID reach RQI > 0.95
        # (selection-biased towards fast runs; reported only with >= 3 successes)
        su = u & (d["reached"] == 1)
        med2 = float(np.median(d["iter_rqi"][su])) if su.sum() >= 3 else np.nan
        inv = float(np.median(1 / d["lambda3"][u])) if u.any() else np.nan
        agg.append((n, r, m.sum(), reached.sum(), reached.mean(), med,
                    u.sum(), d["reached"][u].sum(), med2, inv))
        lines.append(",".join(f"{v:.6g}" for v in agg[-1]))
    agg = np.array(agg)
    out = os.path.join(ROOT, "results", "fig4b")
    with open(os.path.join(out, f"summary_{a.tag}.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # contingency: does reaching high RQI require a determined representation?
    det = d["dof"] == 2
    rea = d["reached"] == 1
    table = np.array([[np.sum(det & rea), np.sum(det & ~rea)],
                      [np.sum(~det & rea), np.sum(~det & ~rea)]])

    # log-log slope of median steps vs median 1/lambda_3 (sizes with a defined median)
    ok = np.isfinite(agg[:, 8]) & np.isfinite(agg[:, 9])
    slope = icpt = np.nan
    if ok.sum() >= 3:
        slope, icpt = np.polyfit(np.log(agg[ok, 9]), np.log(agg[ok, 8]), 1)

    # per-run log-log slope of steps on lambda_3, successful dof = 2 runs
    s_ok = det & rea
    slope_run = icpt_run = np.nan
    if s_ok.sum() >= 3:
        slope_run, icpt_run = np.polyfit(np.log(d["lambda3"][s_ok]),
                                         np.log(d["iter_rqi"][s_ok]), 1)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(d["lambda3"][s_ok], d["iter_rqi"][s_ok])

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))

    # (a)
    jitter = np.random.default_rng(0).uniform(-0.006, 0.006, len(d["fraction"]))
    x = d["fraction"] + jitter
    ax[0].scatter(x[rea], d["iter_rqi"][rea], s=8, color=BLUE, alpha=0.55,
                  linewidths=0, label="reached RQI > 0.95")
    ax[0].scatter(x[~rea], np.full((~rea).sum(), cap * 1.25), s=10, marker="^",
                  facecolors="none", edgecolors=ORANGE, linewidths=0.6,
                  label=f"not reached in {int(cap) + 1:,} steps")
    ax[0].plot(agg[:, 1], agg[:, 5], color=BLUE, lw=1.5, label="median")
    ax[0].axvline(0.4, color=GREY, ls="--", lw=0.7)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("training data fraction $r$")
    ax[0].set_ylabel("steps to RQI > 0.95")
    ax[0].set_title("(a) steps to RQI > 0.95", loc="left", fontsize=7.5)
    ax[0].legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.72))

    # (b)
    ax[1].plot(th["fraction"], th["p_unique"], color=GREY, lw=1.2, ls="--", label=r"theory: $P(\lambda_3>0)$")
    ax[1].plot(agg[:, 1], agg[:, 4], "o-", color=BLUE, ms=3, lw=1.5,
               label="training: P(RQI > 0.95)")
    ax[1].axvline(0.4, color=GREY, ls="--", lw=0.7)
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].set_xlabel("training data fraction $r$")
    ax[1].set_ylabel("probability")
    ax[1].set_title("(b) theory vs training", loc="left", fontsize=7.5)
    ax[1].legend(fontsize=5.5, frameon=False, loc="upper left")

    # (c) per run, as in the paper's Fig. 5(b): steps vs lambda_3 for every run
    # with lambda_3 > 0; failures at the cap. Our lambda_3 normalisation differs
    # from the paper's by a constant factor, so only the slope is comparable
    # (theory: steps ~ 1/lambda_3, i.e. slope -1 on log-log axes).
    pos = det
    s_ok = pos & rea
    ax[2].scatter(d["lambda3"][s_ok], d["iter_rqi"][s_ok], s=8, color=BLUE,
                  alpha=0.55, linewidths=0, label="reached")
    ax[2].scatter(d["lambda3"][pos & ~rea], np.full((pos & ~rea).sum(), cap * 1.25),
                  s=10, marker="^", facecolors="none", edgecolors=ORANGE,
                  linewidths=0.6, label="not reached")
    if np.isfinite(slope_run):
        xx = np.geomspace(d["lambda3"][s_ok].min(), d["lambda3"][s_ok].max(), 50)
        ax[2].plot(xx, np.exp(icpt_run) * xx**slope_run, color=GREY, lw=0.9,
                   label=f"fit: slope {slope_run:.2f}")
        ref = np.exp(np.mean(np.log(d["iter_rqi"][s_ok]) + np.log(d["lambda3"][s_ok])))
        ax[2].plot(xx, ref / xx, color=GREY, lw=0.9, ls=":", label="theory: slope $-1$")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$\lambda_3$ (this work's normalisation)")
    ax[2].set_ylabel("steps to RQI > 0.95")
    ax[2].set_title("(c) per run, cf. paper Fig. 5(b)", loc="left", fontsize=7.5)
    ax[2].legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.70))

    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig4b_{a.tag}.{ext}"), dpi=220)

    print(f"runs: {len(d['n_train'])}, step cap {int(cap) + 1}")
    print("\n".join(lines))
    print("contingency (rows: dof==2, dof>2; cols: reached, not reached):")
    print(table)
    print(f"log-log slope of median steps on median 1/lambda3: {slope:.3f} "
          f"(theory: 1 under gradient flow), sizes used: {int(ok.sum())}")
    print(f"per run (successful, lambda_3 > 0, n={int(s_ok.sum())}): log-log slope of steps "
          f"on lambda_3 = {slope_run:.3f} (theory -1); Spearman rho = {rho:.3f}, p = {pval:.3g}")
    print(f"lambda_3 > 0 runs that failed: {int((det & ~rea).sum())}, lambda_3 range "
          f"{d['lambda3'][det & ~rea].min():.4f}-{d['lambda3'][det & ~rea].max():.4f}")


if __name__ == "__main__":
    main()
