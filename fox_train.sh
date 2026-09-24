#!/bin/bash
# Fox (UiO) Slurm job: trains the space agent with DQN, PPO and SAC in parallel, one array task per algorithm.
# Submit from the repository root:   sbatch fox_train.sh
# Only one algorithm, e.g. PPO:      sbatch --array=1 fox_train.sh
#SBATCH --account=ec12               # <-- your Educloud project
#SBATCH --job-name=hugin-space
#SBATCH --partition=normal
#SBATCH --array=0-2                  # 0 = DDQN, 1 = PPO (--paper), 2 = SAC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # 12 env workers + the main process, with headroom
#SBATCH --mem-per-cpu=2G             # 32 GiB per run; a DQN run is estimated to peak around 9 GiB
#SBATCH --time=16:00:00              # checkpoints (every 1-2M steps) survive if the limit is hit
#SBATCH --output=slurm-%x-%A_%a.out

set -o errexit
set -o nounset

module --quiet purge
module load Python/3.12.3-GCCcore-13.3.0   # same version the .venv was created with (needs >= 3.12)

# one math thread per process: the 12 env workers already fill the cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg                      # compute nodes have no display

cd "$SLURM_SUBMIT_DIR"                     # runs write to ../trained-agents/<run name>
PY=.venv/bin/python

case "$SLURM_ARRAY_TASK_ID" in
    0) $PY -m HUGIN_gym.training.train_DDQN ;;
    1) $PY -m HUGIN_gym.training.train_PPO --paper ;;
    2) $PY -m HUGIN_gym.training.train_SAC ;;
esac
