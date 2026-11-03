#!/bin/bash
# Fox (UiO) Slurm job: the results of run 6, in one job.
#   1. Evaluates the meta agents and the baselines, see HUGIN_gym/evaluation/run_evaluations.py.
#      Results go to ../eval-results/run6, one .json and one .log per evaluation.
#   2. Plots training_stats.png in each run folder, and writes ../eval-results/run6/training_summary.md.
# Submit from the repository root:   sbatch fox_eval_HRL.sh
# An evaluation with a saved result is skipped, so submit it again if the job stops or the time limit is hit.
# 118 evaluations of up to about 8 minutes each on a PC (6 at a time took 467 s), so about 1-2 h on 16 CPUs, plus the plots.
#SBATCH --account=ec12
#SBATCH --job-name=hugin-eval
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # one evaluation per CPU
#SBATCH --mem-per-cpu=2G             # 32 GiB; run_evaluations keeps every evaluation to one math thread
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -o errexit
set -o nounset

module --quiet purge
module load Python/3.12.3-GCCcore-13.3.0   # same version the .venv was created with (needs >= 3.12)

export MPLBACKEND=Agg                      # compute nodes have no display

cd "$SLURM_SUBMIT_DIR"                     # the sub-agents and run folders are read from ../trained-agents
.venv/bin/python -m HUGIN_gym.evaluation.run_evaluations --workers "$SLURM_CPUS_PER_TASK"

RUNS=(../trained-agents/HRL_PPO_*_seed?_4387[78]*)   # the jobs of run 6: 4387793 to 4387813
.venv/bin/python -m HUGIN_gym.evaluation.summarise_training_stats "${RUNS[@]}" --last 1000 > ../eval-results/run6/training_summary.md
.venv/bin/python -m HUGIN_gym.evaluation.training_data_visualiser "${RUNS[@]}"
