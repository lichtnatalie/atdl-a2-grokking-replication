#!/bin/bash
# Ablation 2: decoder learning rate (embedding lr fixed at 1e-3, no weight decay).
# Released default is 1e-4 (= runs_baseline.csv).
cd "$(dirname "$0")/.."
python3 experiments/fig4b_steps_to_rqi.py --eta-dec 1e-5 --tag lrdec1e-5 >> results/fig4b/log_lrdec1e-5.txt 2>&1
python3 experiments/fig4b_steps_to_rqi.py --eta-dec 1e-3 --tag lrdec1e-3 >> results/fig4b/log_lrdec1e-3.txt 2>&1
# are the slow-decoder failures stuck or slow? 10 stalled lambda_3 > 0 runs at 50,000 steps
python3 experiments/fig4b_extended_budget.py --tag lrdec1e-5 --eta-dec 1e-5 > results/fig4b/log_extended_lrdec1e-5.txt 2>&1
