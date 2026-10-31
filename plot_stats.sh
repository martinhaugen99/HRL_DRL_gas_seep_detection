#!/bin/bash
# Fox (UiO) Slurm job: plots training_stats.pkl of the given run folders (training_data_visualiser.py).
# Submit from the repository root:   sbatch plot_stats.sh <run folder> [<run folder> ...]
#                                    e.g. sbatch plot_stats.sh ../trained-agents/PPO_compare_100M_4259607
#SBATCH --account=ec12
#SBATCH --job-name=plot-stats
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G            # the stats need about 5 GB in memory; the rest is margin for plotting
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -o errexit
set -o nounset

module --quiet purge
module load Python/3.12.3-GCCcore-13.3.0
export MPLBACKEND=Agg                # compute nodes have no display

cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -m HUGIN_gym.evaluation.training_data_visualiser "$@"
