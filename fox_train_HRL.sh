#!/bin/bash
# Fox (UiO) Slurm job: trains the HRL meta agent with PPO, on top of the trained space, plume and border agents.
# The sub-agent models are read from the paths in HUGIN_gym/training/train_HRL(ppo).py (../trained-agents/...).
# The tag is the meta agent variant: a key of PRESETS in train_HRL(ppo).py (untuned, duration, duration_mean,
# minimal, minimal_values, minimal_values_duration).
# Submit from the repository root:   sbatch fox_train_HRL.sh <tag>        e.g. sbatch fox_train_HRL.sh minimal
# The duration variants can run up to 5/3 as many env steps: sbatch --time=2-06:00:00 fox_train_HRL.sh duration
# Resume a run from its newest checkpoint: sbatch fox_train_HRL.sh <tag> <job id of the run>
#                                    e.g. sbatch fox_train_HRL.sh untuned 4300000
# The run is saved as ../trained-agents/HRL_PPO_<tag>_<job id>; the job id makes every submission unique.
#SBATCH --account=ec12              # <-- your Educloud project
#SBATCH --job-name=hugin-hrl
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # 12 env workers + the main process, with headroom
#SBATCH --mem-per-cpu=2G             # 32 GiB; every env worker holds the three sub-agent models and its own GP
#SBATCH --time=1-06:00:00            # 10M steps of untuned took about 27 h; checkpoints (every ~2M steps) survive if the limit is hit
#SBATCH --output=slurm-%x-%j.out

set -o errexit
set -o nounset

module --quiet purge
module load Python/3.12.3-GCCcore-13.3.0   # same version the .venv was created with (needs >= 3.12)

# one math thread per process: the 12 env workers already fill the cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg                      # compute nodes have no display

cd "$SLURM_SUBMIT_DIR"                     # runs write to ../trained-agents/<run name>
PY=.venv/bin/python
TAG=${1:-untuned}                          # first argument after the script name in sbatch
RESUME_ID=${2:-}                           # second argument: job id of the run to resume (none: start a new run)
if [[ -n "$RESUME_ID" ]]; then
    RUN="HRL_PPO_${TAG}_${RESUME_ID}"      # continue in the existing run folder
    RESUME_FLAG="--resume"
else
    RUN="HRL_PPO_${TAG}_${SLURM_JOB_ID}"
    RESUME_FLAG=""
fi

$PY -m "HUGIN_gym.training.train_HRL(ppo)" --preset "$TAG" --name "$RUN" $RESUME_FLAG
