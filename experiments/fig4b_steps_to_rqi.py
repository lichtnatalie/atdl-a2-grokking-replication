"""
Reproduce Fig. 4(b) of Liu et al. (2022): training steps until RQI > 0.95,
as a function of training fraction, using the authors' train_add unmodified.

Authors' setting (toy/produce_figure_data.py): p = 10,
train_nums = [5, 10, ..., 50, 54], seeds [0, 1, 2], steps = 1e4, AdamW,
eta_reprs = 1e-3, eta_dec = 1e-4, no weight decay, MSE loss.
We keep everything except the number of seeds (default 20).

Caveat handled here: train_add returns iter_rqi = last step when RQI never
exceeds the threshold, which is indistinguishable from reaching it on the
last step. We therefore also store `reached` (any RQI > threshold) and treat
unreached runs as right-censored.

For each run we also compute lambda_3 of the SAME training split with our
own implementation (src/effective_theory.py), so steps can later be compared
with the theory's timescale 1/lambda_3.

Output: results/fig4b/runs.csv (appended run by run, so partial results
survive an interruption; finished (n, seed) pairs are skipped on restart).
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "fig4b")
FIELDS = ["n_train", "seed", "fraction", "reached", "iter_rqi", "final_rqi",
          "iter_train", "final_train_acc", "final_test_acc", "ideal_test_acc",
          "dof", "lambda3", "seconds"]


def one_run(args):
    n, seed, steps, threshold_rqi, threshold_P, wd_dec, eta_dec = args
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "third_party", "grokking-squared", "toy"))
    from effective_theory import sample_train_set, spectrum
    from train_add import train_add
    import torch
    torch.set_num_threads(1)

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        d = train_add(train_num=n, seed=seed, steps=steps, eff_steps=1,
                      threshold_rqi=threshold_rqi, threshold_P=threshold_P,
                      weight_decay_dec=wd_dec, eta_dec=eta_dec)
    assert d["eta_dec"] == eta_dec and d["weight_decay_dec"] == wd_dec
    eig, null_dim, _ = spectrum(sample_train_set(10, n, seed), 10)
    assert null_dim == d["dof"], "split mismatch with authors' code"
    rqi = np.asarray(d["rqi"])
    return dict(n_train=n, seed=seed, fraction=n / d["all_num"],
                reached=int(np.any(rqi > threshold_rqi)),
                iter_rqi=int(d["iter_rqi"]), final_rqi=float(rqi[-1]),
                iter_train=int(d["iter_train"]),
                final_train_acc=float(d["acc_train"][-1]),
                final_test_acc=float(d["acc_test"][-1]),
                ideal_test_acc=float(d["ideal_test_acc"]),
                dof=int(d["dof"]), lambda3=float(eig[2]),
                seconds=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--threshold-rqi", type=float, default=0.95)
    ap.add_argument("--threshold-P", type=float, default=0.01)
    ap.add_argument("--wd-dec", type=float, default=0.0)
    ap.add_argument("--eta-dec", type=float, default=1e-4)  # authors' default
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"runs_{a.tag}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}

    train_nums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 54]
    # interleave sizes so partial results cover the whole range
    jobs = [(n, s, a.steps, a.threshold_rqi, a.threshold_P, a.wd_dec, a.eta_dec)
            for s in range(a.seeds) for n in train_nums if (n, s) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already done) -> {path}", flush=True)

    new = not os.path.exists(path)
    with open(path, "a", newline="") as f, get_context("spawn").Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        t0 = time.time()
        for k, row in enumerate(pool.imap_unordered(one_run, jobs), 1):
            w.writerow(row)
            f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
