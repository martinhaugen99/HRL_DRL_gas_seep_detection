#!/bin/bash
# Fox (UiO) Slurm job: trains the plume agent with DQN, PPO and SAC in parallel, one array task per algorithm.
# Submit from the repository root:   sbatch fox_train.sh <tag>        e.g. sbatch fox_train.sh gp_230steps
# Only one algorithm, e.g. PPO:      sbatch --array=1 fox_train.sh <tag>
# Runs are saved as ../trained-agents/{DQN,PPO,SAC}_<tag>_<job id>; the job id makes every submission unique.
#SBATCH --account=ec12              # <-- your Educloud project
#SBATCH --job-name=hugin-plume
#SBATCH --partition=normal
#SBATCH --array=0-2                  # 0 = DQN, 1 = PPO (--compare), 2 = SAC
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
TAG=${1:-plume}                            # first argument after the script name in sbatch
RUN="${TAG}_${SLURM_ARRAY_JOB_ID}"

case "$SLURM_ARRAY_TASK_ID" in
    0) $PY -m HUGIN_gym.training.train_DDQN --name "DQN_$RUN" ;;
    1) $PY -m HUGIN_gym.training.train_PPO --compare --name "PPO_$RUN" ;;
    2) $PY -m HUGIN_gym.training.train_SAC --name "SAC_$RUN" ;;
esac
