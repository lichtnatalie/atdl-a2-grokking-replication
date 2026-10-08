# ATDL Assignment 2 — Reproducibility study of Liu et al. (2022)

**Group 73** - **Replication Track** - Advanced Topics in Deep Learning, University of Copenhagen, 2026

Paper: Z. Liu, O. Kitouni, N. Nolte, E. J. Michaud, M. Tegmark, M. Williams,
*Towards Understanding Grokking: An Effective Theory of Representation Learning*,
NeurIPS 2022. [arXiv:2205.10343](https://arxiv.org/abs/2205.10343)

**Experiment reproduced:** the critical training-set fraction (Fig. 4).

- **(a)** The effective theory's training-free prediction: the probability that the
  training set determines a unique linear representation, i.e. P(λ₃ > 0), as a
  function of the training fraction r. Paper: transition at r_c ≈ 0.4.
- **(b)** The empirical counterpart: training steps until RQI > 0.95, across r and
  seeds.

## Layout

```
src/effective_theory.py           parallelogram constraints, spectrum of H, λ₃ (our implementation)
experiments/fig4a_critical_fraction.py   Fig. 4(a): ours on 1000 seeds + authors' code on 100, per-seed cross-check
experiments/fig4b_steps_to_rqi.py        Fig. 4(b): authors' train_add, 11 sizes x 20 seeds, resumable
experiments/fig4b_analyse.py             Fig. 4(b) summary, censoring-aware medians, figure
experiments/fig4b_extended_budget.py     10 stalled runs re-run with 50,000 steps
experiments/run_wd_ablation.sh           Ablation 1: same sweep with decoder weight decay 1 and 5
experiments/run_lrdec_ablation.sh        Ablation 2: same sweep with decoder lr 1e-5 and 1e-3
experiments/ablation_compare.py          Ablations 1-2: comparison table and figure (--which wd|lrdec)
results/fig4a/                    CSVs and the figure
third_party/grokking-squared/     authors' code, unmodified (MIT), pinned at 0229df9 (2022-11-25)
report/                           ISBI LaTeX source of the report (latexmk -pdf A2_submission.tex)
```

## Reproduce

```bash
pip install -r requirements.txt
python experiments/fig4a_critical_fraction.py        # ~1 min on CPU
python experiments/fig4b_steps_to_rqi.py             # ~50 min on 2 CPU cores
python experiments/fig4b_analyse.py
python experiments/fig4b_extended_budget.py          # ~25 min
experiments/run_wd_ablation.sh                       # ~2 h
python experiments/ablation_compare.py --which wd
experiments/run_lrdec_ablation.sh                    # ~2 h
python experiments/ablation_compare.py --which lrdec
```

## Notes for part (b)

- The authors train with **AdamW**, not plain gradient descent. The theory's
  n_h = 1/(λ₃ η) is derived for gradient flow on the effective loss, so absolute step
  counts are not expected to match; compare the **scaling** of steps-to-RQI with r
  (and with 1/λ₃), not the numbers.
- H is computed as 2 A_Dᵀ A_D / (|P₀(D)| Z₀) with Z₀ = p, from ℓ_eff = ½ Rᵀ H R (eq. 8)
  and normalised 1-D embeddings. The constant only rescales λ₃; it does not affect (a).
- Appendix G (arXiv version) gives the decoder (MLP 1-200-200-30) and embedding lr 10⁻³,
  matching the released code, but no decoder lr, weight decay, seeds or step budget for
  Fig. 4(b). Table 1 defines the phases with 10⁵ steps; the released phase-diagram
  code uses 10⁴.
- In the toy task, test accuracy is capped by the *ideal* accuracy: some test sums
  cannot be inferred from the training pairs even with a perfect representation.
