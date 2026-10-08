"""
Reproduce Fig. 4(a) of Liu et al. (2022): probability that the training set
determines a unique linear representation, as a function of training fraction.

Two independent routes:
  1. our implementation (src/effective_theory.py), many seeds;
  2. the authors' train_add(..., steps=1, eff_steps=1), as in their
     toy/Figure4ab.ipynb, on the same seeds, to check we agree per seed.

Outputs (results/fig4a/):
  fig4a.csv          per (n_train, seed): null_dim, lambda_3, |P_0(D)|
  fig4a_summary.csv  per n_train: P(unique) with Wilson 95% CI, median lambda_3
  crosscheck.csv     per (n_train, seed): our dof vs authors' dof
  fig4a.pdf / .png   the figure
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from effective_theory import full_dataset, sample_train_set, spectrum  # noqa: E402

OUT = os.path.join(ROOT, "results", "fig4a")


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    denom = 1 + z**2 / n
    centre = (ph + z**2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def run_ours(p, train_nums, seeds):
    rows = []
    for n in train_nums:
        for s in seeds:
            train = sample_train_set(p, n, s)
            eig, null_dim, n_par = spectrum(train, p)
            rows.append((n, s, null_dim, eig[2], n_par))
    return np.array(rows, dtype=float)


def run_authors(p, train_nums, seeds):
    sys.path.insert(0, os.path.join(ROOT, "third_party", "grokking-squared", "toy"))
    from train_add import train_add  # noqa: E402

    rows = []
    for n in train_nums:
        for s in seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                d = train_add(p=p, train_num=n, seed=s, steps=1, eff_steps=1)
            rows.append((n, s, int(d["dof"])))
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--crosscheck-seeds", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    p = args.p
    all_num = len(full_dataset(p))
    train_nums = np.arange(1, 19) * 3  # 3, 6, ..., 54 as in the authors' notebook

    t = time.time()
    ours = run_ours(p, train_nums, range(args.seeds))
    print(f"ours: {len(ours)} training sets in {time.time() - t:.1f}s")
    np.savetxt(os.path.join(OUT, "fig4a.csv"), ours, delimiter=",", fmt="%.10g",
               header="n_train,seed,null_dim,lambda3,n_parallelograms", comments="")

    summary = []
    for n in train_nums:
        sub = ours[ours[:, 0] == n]
        k = int(np.sum(sub[:, 2] == 2))
        lo, hi = wilson(k, len(sub))
        lam = sub[sub[:, 2] == 2, 3]
        summary.append((n, n / all_num, k / len(sub), lo, hi,
                        np.median(lam) if len(lam) else np.nan))
    summary = np.array(summary)
    np.savetxt(os.path.join(OUT, "fig4a_summary.csv"), summary, delimiter=",",
               fmt="%.6g", comments="",
               header="n_train,fraction,p_unique,ci_lo,ci_hi,median_lambda3_given_unique")

    if args.crosscheck_seeds > 0:
        t = time.time()
        auth = run_authors(p, train_nums, range(args.crosscheck_seeds))
        print(f"authors: {len(auth)} training sets in {time.time() - t:.1f}s")
        mine = ours[ours[:, 1] < args.crosscheck_seeds][:, :3]
        assert np.array_equal(mine[:, :2], auth[:, :2])
        agree = mine[:, 2] == auth[:, 2]
        print(f"per-seed agreement on dof: {agree.sum()}/{len(agree)}")
        np.savetxt(os.path.join(OUT, "crosscheck.csv"),
                   np.column_stack([auth, mine[:, 2], agree]), delimiter=",",
                   fmt="%d", comments="",
                   header="n_train,seed,dof_authors,dof_ours,agree")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = summary[:, 1]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.fill_between(frac, summary[:, 3], summary[:, 4], color="#4C72B0", alpha=0.25,
                    linewidth=0, label="95% Wilson CI")
    ax.plot(frac, summary[:, 2], "o-", color="#4C72B0", ms=3, lw=1.2,
            label=f"this work ({args.seeds} seeds)")
    if args.crosscheck_seeds > 0:
        pa = [np.mean(auth[auth[:, 0] == n, 2] == 2) for n in train_nums]
        ax.plot(frac, pa, "x", color="#C44E52", ms=4,
                label=f"authors' code ({args.crosscheck_seeds} seeds)")
    ax.axvline(0.4, color="grey", ls="--", lw=0.8)
    ax.text(0.39, 0.55, r"$r_c=0.4$ (paper)", fontsize=7, color="grey",
            ha="right")
    ax.set_xlabel("training data fraction $r$")
    ax.set_ylabel(r"$P(\lambda_3 > 0)$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4a.{ext}"), dpi=200)

    # crossing point by linear interpolation
    pu = summary[:, 2]
    i = np.argmax(pu >= 0.5)
    rc = frac[i - 1] + (0.5 - pu[i - 1]) * (frac[i] - frac[i - 1]) / (pu[i] - pu[i - 1])
    print(f"P(unique) = 0.5 at r = {rc:.3f}")
    for row in summary:
        print("n=%2d r=%.3f P=%.3f [%.3f, %.3f] median_lambda3=%.4g" % tuple(row))


if __name__ == "__main__":
    main()
