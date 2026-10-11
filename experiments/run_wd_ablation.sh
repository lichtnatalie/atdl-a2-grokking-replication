#!/bin/bash
# Ablation 1: decoder weight decay, everything else as the baseline sweep.
cd "$(dirname "$0")/.."
for wd in 1 5; do
  python3 experiments/fig4b_steps_to_rqi.py --wd-dec $wd --tag wd$wd >> results/fig4b/log_wd$wd.txt 2>&1
done
