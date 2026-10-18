#!/bin/bash
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
.venv/bin/python -m HUGIN_gym.evaluation.training_data_visualiser