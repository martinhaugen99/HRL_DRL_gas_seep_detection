#!/bin/bash
# Submits the meta agent variants of run 6 to Fox: every variant with every seed, one job each.
# The variants are the survey presets in HUGIN_gym/training/train_HRL(ppo).py. Pull the code on Fox first.
# Run on a login node from the repository root (not with sbatch):
#     bash fox_train_HRL_matrix.sh [decisions]        e.g. bash fox_train_HRL_matrix.sh 5000000
# Each job runs fox_train_HRL.sh and is saved as ../trained-agents/HRL_PPO_<tag>_seed<seed>_<job id>.
# 5M decisions take about 2.4 h at the 579 decisions/s of job 4380825. Other meta states may be slower, so the limit is 6 h.

set -o errexit
set -o nounset

DECISIONS=${1:-5000000}
TAGS="sweep6_est sweep4_visit sweep6_est_map sweep6_est_no_time space_est sweep6_est_one"
SEEDS="1 2 3"

for tag in $TAGS; do
    for seed in $SEEDS; do
        sbatch --time=06:00:00 --export=ALL,SEED="$seed",MAX_STEPS="$DECISIONS" fox_train_HRL.sh "$tag"
    done
done
