"""
Are the Fig. 4(b) runs that never reach RQI > 0.95 slow or stuck?

Re-runs a sample of censored runs whose training set DOES determine the
linear representation (dof = 2) with a 5x longer step budget (50,000), same
seed and split, and records whether / when RQI > 0.95 is reached and the RQI
trajectory at a few checkpoints.

Output: results/fig4b/extended_budget.csv (baseline) or
        results/fig4b/extended_budget_<tag>.csv (with --tag / --eta-dec)
"""

import argparse
import contextlib
import csv
import io
import os
import sys
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 50_000
CHECK = [1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 49_999]


def one(job):
    n, seed, eta_dec = job
    sys.path.insert(0, os.path.join(ROOT, "third_party", "grokking-squared", "toy"))
    from train_add import train_add
    import torch
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        d = train_add(train_num=n, seed=seed, steps=STEPS, eff_steps=1, eta_dec=eta_dec)
    rqi = np.asarray(d["rqi"])
    out = dict(n_train=n, seed=seed, reached=int(np.any(rqi > 0.95)),
               iter_rqi=int(d["iter_rqi"]), max_rqi=float(rqi.max()),
               final_train_acc=float(d["acc_train"][-1]))
    out.update({f"rqi_{c}": round(float(rqi[c]), 3) for c in CHECK})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--eta-dec", type=float, default=1e-4)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "fig4b",
                                                 f"runs_{a.tag}.csv"))))
    stalled = [(int(r["n_train"]), int(r["seed"])) for r in rows
               if r["dof"] == "2" and r["reached"] == "0"]
    rng = np.random.default_rng(0)
    # two per size from 30 to 50, where stalls and successes coexist
    jobs = []
    for n in (30, 35, 40, 45, 50):
        cand = [j for j in stalled if j[0] == n]
        if not cand:
            cand = [j for j in stalled if j[0] < n][-2:]  # fall back to smaller sizes
        if cand:
            pick = rng.choice(len(cand), size=min(2, len(cand)), replace=False)
            jobs += [(*cand[i], a.eta_dec) for i in pick]
    jobs = list(dict.fromkeys(jobs))  # drop duplicates from the fallback
    print("jobs:", jobs, flush=True)
    with get_context("spawn").Pool(os.cpu_count()) as pool:
        res = pool.map(one, jobs)
    name = "extended_budget.csv" if a.tag == "baseline" else f"extended_budget_{a.tag}.csv"
    path = os.path.join(ROOT, "results", "fig4b", name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(r, flush=True)


if __name__ == "__main__":
    main()
