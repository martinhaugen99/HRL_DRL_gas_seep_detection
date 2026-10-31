# Documentation of the different training runs

This file records every training run on Fox: the plume agent in runs 1–3 and the HRL meta agent from run 4 on (see [HRL meta agent](#hrl-meta-agent)). For each run it gives the settings, what changed from the previous run and why, and the results. Use it to see how a parameter change affects training.

- **Run folders:** `../trained-agents/<run>` on this machine. On Fox the training scripts write to `../trained-agents/<run>`.
- **Run names:** `<ALG>_<tag>_<Slurm job ID>`, set by [fox_train.sh](../fox_train.sh). One `sbatch fox_train.sh <tag>` starts DQN, PPO and SAC as one array job, so all three share the job ID.
- **Copies in git:** [training_runs/](training_runs/) holds `training_stats.png` and `training_config.json` for each run, and `resume_log.txt` for runs that were resumed after a failed job. The SVG plots, the models and the stats pickles stay in `../trained-agents`.
- **Why the methods are set up the way they are:** [comparing_training_methods.md](comparing_training_methods.md).

## How to read the results

The plots come from [training_data_visualiser.py](../HUGIN_gym/evaluation/training_data_visualiser.py). The numbers tables come from [summarise_training_stats.py](../HUGIN_gym/evaluation/summarise_training_stats.py):

```bash
.venv/bin/python -m HUGIN_gym.evaluation.summarise_training_stats ../trained-agents/<run> [<run> ...] --last 1000
```

| Metric | Meaning |
|---|---|
| Total reward | Sum of the rewards in the episode. |
| Episode length | Steps in the episode. The step limit is the run's episode length setting. |
| Visited (%) | Share of the map the agent visited by the end of the episode. |
| Plume coverage (%) | Share of the true above-threshold cells the agent visited by the end of the episode. In the plots this panel is labelled "Steps Above Threshold (%)". |
| Ended early (%) | Share of episodes that ended before the step limit. For the plume agent this only happens when it reaches its plume coverage goal (100 %, +40 reward). With the GP on, the goal is checked against the GP's estimate of the plume, so it can differ from the true coverage. |

- The numbers are mean ± std over the last 1,000 training episodes of a run. For repeated runs, the bold row gives the mean ± std of the run means.
- In GP runs, 1–15 % of the episodes do not report the true plume coverage, so that column covers slightly different episodes than the others.
- Training episodes include exploration (random actions for DQN, sampled actions for PPO and SAC). Use these numbers to see how a change affects one algorithm. Do not use them to rank the algorithms; that needs the deterministic evaluation described in [comparing_training_methods.md](comparing_training_methods.md#3-how-to-compare).
- In the plots, the "Steps Around Threshold (%)" panel is always empty for the plume agent. The rightmost panel shows the reward at each step of every 10th episode; the spikes to 40 are the goal bonus.

## Overview

| # | Tag | Slurm job IDs | Started | Code | Main change | Status |
|---|---|---|---|---|---|---|
| 1 | `gp_230steps` | 4218364 | 2026-09-24 22:01 | `f220184` | First runs of all three algorithms on Fox | Finished |
| 2 | `compare` | 4223550, 4223553, 4223557 | 2026-09-26 13:35–13:49 | `961f7a2` | Same task, budget and network size for all three algorithms; start near the plume; random plume size; less DQN exploration. 3 repeats | Finished |
| 3 | `compare_100M` | 4259604, 4259607, 4259610; resumed as 4272086, 4272088, 4272091 | 2026-10-04 22:31 – 2026-10-05 05:05 | `10e4023`; resumed with `bf7e7fc` | 10× the steps, 24 envs, 460-step episodes, 128 × 128 networks. 3 repeats | Finished; 8 of 9 runs resumed after a full disk quota |
| 4 | `untuned` (HRL) | 4363368; resumed as 4367840, 4371199 | 2026-10-28 15:13 | `56ba1c7` | First training of the HRL meta agent, with the settings from the thesis code | Finished 2026-10-30 01:23; hit the 12 h limit twice and was resumed |
| 5 | `minimal`, `minimal_values`, `duration`, `duration_mean`, `minimal_values_duration` (HRL) | 4371279, 4371280, 4371281, 4371282, 4371283 | 2026-10-29 22:58 | `152ba0a` | The meta agent chooses how many steps a sub-agent runs; smaller meta states; sub-agent value estimates | Finished 2026-10-30 14:29 – 2026-10-31 08:50, each in one job |

"Started" is `RUN_INFO.timestamp` in `training_config.json`. "Code" is the last commit before the run started. From run 3 on, it is the commit checked out on Fox when the job started (`git reflog --date=iso` on Fox).

## Settings shared by runs 1–3 (plume agent)

- **Agent:** plume agent (`AGENT_TYPE = "PLUME"`) with the feature extractor `AgentPlumeExplore` ([agent2_plume_explore.py](../HUGIN_gym/agents/feature_extractor/agent2_plume_explore.py)). It turns 108 inputs into 128 features (ReLU): plume and map coverage, position and heading, a 7 × 7 visited patch, a 7 × 7 plume patch, and the distance from the maximum. The hidden layers after it differ per run (see the tables).
- **Observation keys:** DQN and SAC keep 7 keys (`obs_state`, `visited_maps_downsampled`, `GT_c_over_threshold_maps_downsampled`, `local_GT`, `plume_coverage`, `space_coverage`, `distance_from_max`); PPO keeps 5 (without the two downsampled maps). The feature extractor does not read the downsampled maps, so all three get the same inputs.
- **Environment** ([HUGIN_env.py](../HUGIN_gym/envs/HUGIN_env.py)): 2D map of 41 × 41 cells, one Gaussian gas source per episode at a random position, concentration threshold 0.3, 5 discrete actions (forward, up, down, left turn, right turn). SAC acts through [continuous_wrapper.py](../HUGIN_gym/envs/wrappers/continuous_wrapper.py), which maps one number in [−1, 1] to the 5 actions.
- **Goal:** accuracy goals [0.9, 1.0, 1.0], so the plume agent's goal is 100 % plume coverage.
- **Reward** ([agent2_plume.py](../HUGIN_gym/envs/core/rewards/agent2_plume.py), identical in all runs; each `training_config.json` stores the source under `REWARD_FUNCTION_SOURCE`): per step a −0.7 time penalty, + (newly covered above-threshold cells) / 6, and a turn penalty of up to −0.3. Reaching the goal gives +40 and ends the episode.
- **GP** ([GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py), RBF kernel, length scale 3.5): with the GP on, the agent's plume patch and plume coverage come from a Gaussian process fitted to its own measurements, and the reward and the goal check use that estimate. With the GP off, the agent sees the true plume.
- **Fox:** module `Python/3.12.3-GCCcore-13.3.0`, one math thread per process, one array task per algorithm.

## Run 1: `gp_230steps` (job 4218364)

- **Goal:** first training of the plume agent with DQN, PPO and SAC on Fox, as a baseline.
- **Submitted with:** `sbatch fox_train.sh gp_230steps`
- **Code:** `f220184` (2026-09-24). PPO used the `--paper` preset (`PAPER_CONFIG`).
- **Fox resources per algorithm:** 16 CPUs, 2 GB per CPU, 16 h limit.

> **The tag is misleading.** Only DQN trained with the GP and 230-step episodes. PPO and SAC trained without the GP and with 460-step episodes, which is a different and easier task. DQN also trained for 3× as many steps. The three algorithms in this run are therefore not comparable.

### Task and budget

| Setting | DQN | PPO | SAC |
|---|---|---|---|
| Script and preset | `train_DDQN.py` | `train_PPO.py --paper` | `train_SAC.py` |
| GP / `SUB_AGENT_TRAIN_ON_GP` | on / on | off / on (no effect without the GP) | off / off |
| Episode length (steps) | 230 | 460 | 460 |
| Total steps | 30M | 10M | 10M |
| Parallel envs | 12 | 12 | 12 |
| `gamma` | 0.9945 | 0.9975 | 0.997 |
| Hidden layers (SB3 defaults) | [64, 64] ReLU | pi [64, 64], vf [64, 64] Tanh | [256, 256] ReLU |
| Checkpoint every | 2M steps | 1M steps | 2M steps |
| Start position | anywhere on the map | anywhere on the map | anywhere on the map |
| Plume size | fixed, σ = 3.5 | fixed, σ = 3.5 | fixed, σ = 3.5 |
| Wall-clock time (approx.) | 14.4 h | 1.2 h | 6.8 h |

### Algorithm settings

| Setting | DQN | PPO | SAC |
|---|---|---|---|
| `learning_rate` | 1e-4 | 3e-4 | 5e-5 |
| `batch_size` | 128 | 1024 | 256 |
| Replay buffer | 800,000 | – | 900,000 |
| `learning_starts` | 400,000 | – | 100 (SB3 default) |
| `train_freq` / `gradient_steps` | 4 / 4 | – | 4 / 4 |
| Target network update | every 50,000 steps | – | every step, `tau` 0.01 |
| Exploration | ε from 1.0 to 0.05 over the first 80 % of the run (`exploration_fraction` 0.8) | `ent_coef` 0.01 | `ent_coef` auto |
| `n_steps` / `n_epochs` | – | 512 / 4 | – |
| `gae_lambda` / `clip_range` | – | 0.95 / 0.2 | – |
| `vf_coef` / `max_grad_norm` | – | 0.5 / 0.5 | – |

The `episode_stats_*.pkl` snapshots of this run were deleted on 2026-10-05 to save space. `training_stats.pkl` (all episodes) is kept.

### Results

Mean ± std over the last 1,000 training episodes:

| Run | Episodes | Total reward | Episode length | Visited (%) | Plume coverage (%) | Ended early (%) |
|---|---|---|---|---|---|---|
| `DQN_gp_230steps_4218364` | 149,661 | -57.2 ± 69.6 | 161.9 ± 65.4 | 28.5 ± 12.3 | 96.0 ± 17.5 | 62.4 |
| `PPO_gp_230steps_4218364` | 31,613 | -60.7 ± 91.3 | 151.0 ± 108.8 | 23.3 ± 12.1 | 98.9 ± 8.5 | 93.9 |
| `SAC_gp_230steps_4218364` | 50,301 | -83.8 ± 145.0 | 138.2 ± 115.5 | 14.5 ± 4.5 | 96.1 ± 18.7 | 92.1 |

**DQN** (GP on, 230 steps, 30M steps)

![DQN gp_230steps](training_runs/DQN_gp_230steps_4218364/training_stats.png)

**PPO** (GP off, 460 steps, 10M steps)

![PPO gp_230steps](training_runs/PPO_gp_230steps_4218364/training_stats.png)

**SAC** (GP off, 460 steps, 10M steps)

![SAC gp_230steps](training_runs/SAC_gp_230steps_4218364/training_stats.png)

### Observations

- **DQN:** the reward rises slowly and steadily over the whole run, from about −500 to about −50 over 150k episodes. The episode length stays at the 230-step limit for the first ~30k episodes and is still noisy at the end. With `exploration_fraction` 0.8, ε only reaches its final 5 % after 24M of the 30M steps, so most of the run has many random actions.
- **PPO:** the episode length falls from 460 to about 150 and is still falling at the end, so the run had not converged after 10M steps. The share of the map visited falls from about 45 % to about 22 %. It was the fastest run (1.2 h).
- **SAC:** most of the gain happens in the first ~10k episodes. After that, the reward and the episode length level off at about −100 and 150 steps. It visits the smallest share of the map (about 15 %).

## Run 2: `compare` (jobs 4223550, 4223553, 4223557)

- **Goal:** a fair comparison of DQN, PPO and SAC: the same task, training budget and network size for all three, each with its own algorithm settings. Submitted three times with identical settings to see the spread between runs.
- **Submitted with:** `sbatch fox_train.sh compare` (3 times)
- **Code:** `961f7a2` (2026-09-26). PPO used the new `--compare` preset (`COMPARE_CONFIG`).
- **Fox resources per algorithm:** unchanged (16 CPUs, 2 GB per CPU, 16 h limit).

### What changed from run 1

| Change | Commit | Why |
|---|---|---|
| Start in a box around the plume (the plume's bounding box plus 3 cells) instead of anywhere on the map | `ac68318` | In the HRL setup the plume agent takes over near a plume, after the exploration agent has found it. |
| New plume size every episode: σ drawn from [2.0, 5.0] (plumes 5–15 cells wide) instead of a fixed σ = 3.5 | `ac68318` | Train on plumes of different sizes instead of one size. |
| GP on and 230-step episodes for all three (PPO and SAC had the GP off and 460 steps) | `858ee1e` | Same task for all. |
| 10M steps for all (DQN had 30M) | `858ee1e` | Same budget. |
| `gamma` 0.995 for all (was 0.9945 / 0.9975 / 0.997) | `858ee1e` | `gamma` defines what the agent optimizes, so it must be equal. |
| Hidden layers [64, 64] for all (SAC had [256, 256]) | `858ee1e` | Same network size. |
| Checkpoint every 1M steps for all | `858ee1e` | Checkpoints at the same step counts, for evaluation during training. |
| PPO: new `COMPARE_CONFIG` preset with `n_steps` 2048 (was 512 in `PAPER_CONFIG`) | `858ee1e`, `961f7a2` | A PPO preset with the GP on and 230 steps. |
| DQN `exploration_fraction` 0.8 → 0.2 | `cad73a3` | With 0.8, DQN acts mostly at random for most of the run; 0.1–0.3 is standard. |

> Many settings changed at once, so the difference between run 1 and run 2 cannot be traced to a single parameter.

### Task and budget

Values in bold changed from run 1.

| Setting | DQN | PPO | SAC |
|---|---|---|---|
| Script and preset | `train_DDQN.py` | `train_PPO.py` **`--compare`** | `train_SAC.py` |
| GP / `SUB_AGENT_TRAIN_ON_GP` | on / on | **on** / on | **on / on** |
| Episode length (steps) | 230 | **230** | **230** |
| Total steps | **10M** | 10M | 10M |
| Parallel envs | 12 | 12 | 12 |
| `gamma` | **0.995** | **0.995** | **0.995** |
| Hidden layers | [64, 64] ReLU | pi [64, 64], vf [64, 64] Tanh | **[64, 64]** ReLU |
| Checkpoint every | **1M steps** | 1M steps | **1M steps** |
| Start position | **near the plume** | **near the plume** | **near the plume** |
| Plume size | **σ random in [2.0, 5.0]** | **σ random in [2.0, 5.0]** | **σ random in [2.0, 5.0]** |
| Wall-clock time (approx., 3 runs) | 4.2–5.2 h | 2.8–7.4 h | 6.3–11.9 h |

### Algorithm settings

Values in bold changed from run 1.

| Setting | DQN | PPO | SAC |
|---|---|---|---|
| `learning_rate` | 1e-4 | 3e-4 | 5e-5 |
| `batch_size` | 128 | 1024 | 256 |
| Replay buffer | 800,000 | – | 900,000 |
| `learning_starts` | 400,000 | – | 100 (SB3 default) |
| `train_freq` / `gradient_steps` | 4 / 4 | – | 4 / 4 |
| Target network update | every 50,000 steps | – | every step, `tau` 0.01 |
| Exploration | ε from 1.0 to 0.05 over the first **20 %** of the run (`exploration_fraction` **0.2**) | `ent_coef` 0.01 | `ent_coef` auto |
| `n_steps` / `n_epochs` | – | **2048** / 4 | – |
| `gae_lambda` / `clip_range` | – | 0.95 / 0.2 | – |
| `vf_coef` / `max_grad_norm` | – | 0.5 / 0.5 | – |

### Results

Mean ± std over the last 1,000 training episodes of each run:

| Run | Episodes | Total reward | Episode length | Visited (%) | Plume coverage (%) | Ended early (%) |
|---|---|---|---|---|---|---|
| `DQN_compare_4223550` | 55,330 | -94.6 ± 80.0 | 177.8 ± 78.6 | 35.2 ± 17.1 | 93.6 ± 20.6 | 36.9 |
| `DQN_compare_4223553` | 55,211 | -110.1 ± 69.3 | 192.2 ± 67.4 | 29.1 ± 12.0 | 90.0 ± 24.9 | 31.3 |
| `DQN_compare_4223557` | 55,815 | -88.0 ± 75.0 | 171.2 ± 78.0 | 28.7 ± 15.3 | 92.5 ± 22.0 | 44.3 |
| **DQN_compare (mean of 3 runs)** |  | **-97.6 ± 9.2** | **180.4 ± 8.8** | **31.0 ± 3.0** | **92.0 ± 1.5** | **37.5 ± 5.3** |
| `PPO_compare_4223550` | 83,262 | -2.8 ± 49.8 | 77.4 ± 55.1 | 9.0 ± 4.0 | 99.4 ± 5.5 | 96.2 |
| `PPO_compare_4223553` | 83,533 | -6.0 ± 44.9 | 81.5 ± 57.8 | 9.3 ± 3.7 | 99.6 ± 3.4 | 95.5 |
| `PPO_compare_4223557` | 74,930 | -5.4 ± 42.8 | 83.2 ± 58.7 | 10.0 ± 4.5 | 99.5 ± 5.5 | 95.6 |
| **PPO_compare (mean of 3 runs)** |  | **-4.7 ± 1.4** | **80.7 ± 2.4** | **9.4 ± 0.4** | **99.5 ± 0.1** | **95.8 ± 0.3** |
| `SAC_compare_4223550` | 52,335 | -76.0 ± 92.8 | 140.1 ± 84.0 | 9.5 ± 5.5 | 91.5 ± 25.6 | 61.9 |
| `SAC_compare_4223553` | 60,498 | -69.2 ± 110.7 | 117.7 ± 75.9 | 10.9 ± 6.8 | 93.7 ± 22.4 | 80.2 |
| `SAC_compare_4223557` | 51,528 | -85.4 ± 92.9 | 142.0 ± 79.5 | 12.3 ± 8.8 | 92.0 ± 24.2 | 66.7 |
| **SAC_compare (mean of 3 runs)** |  | **-76.9 ± 6.6** | **133.3 ± 11.1** | **10.9 ± 1.2** | **92.4 ± 0.9** | **69.6 ± 7.7** |


**DQN**

![DQN compare 4223550](training_runs/DQN_compare_4223550/training_stats.png)
![DQN compare 4223553](training_runs/DQN_compare_4223553/training_stats.png)
![DQN compare 4223557](training_runs/DQN_compare_4223557/training_stats.png)

**PPO**

![PPO compare 4223550](training_runs/PPO_compare_4223550/training_stats.png)
![PPO compare 4223553](training_runs/PPO_compare_4223553/training_stats.png)
![PPO compare 4223557](training_runs/PPO_compare_4223557/training_stats.png)

**SAC**

![SAC compare 4223550](training_runs/SAC_compare_4223550/training_stats.png)
![SAC compare 4223553](training_runs/SAC_compare_4223553/training_stats.png)
![SAC compare 4223557](training_runs/SAC_compare_4223557/training_stats.png)

### Observations

- **Spread between repeats:** small for PPO (the run means of the reward range from −6.0 to −2.8), larger for SAC ("ended early" ranges from 62 % to 80 %).
- **DQN:** the reward climbs fast in the first ~10k episodes (ε reaches 0.05 after 2M steps, compared with 24M in run 1), then stays at about −100 for the remaining ~45k episodes. The true plume coverage is high (92 %), but only 37 % of the episodes reach the goal and end early. It visits about 30 % of the map, about 3× more than PPO and SAC.
- **PPO:** the reward rises steadily to about −5. Episodes get shorter (about 80 steps), and 96 % end early with 99.5 % plume coverage while visiting only about 9 % of the map. The reward is still rising slightly and the episode length still falling at 10M steps.
- **SAC:** the episode length stays at the 230-step limit for the first 12k–22k episodes, and the plume coverage drops to 20–40 % during this phase. Then it learns to finish episodes. When this happens differs between repeats (about 12k episodes in `4223553`, about 22k in `4223550`). It ends at about −77 reward, with 70 % of the episodes ending early.
- **Wall-clock time:** SAC took the longest (6.3–11.9 h), then DQN (4.2–5.2 h) and PPO (2.8–7.4 h). The times vary between repeats on the same resources, so treat them as rough.

## Run 3: `compare_100M` (jobs 4259604, 4259607, 4259610)

- **Goal:** the run 2 setup with 10× the training budget, longer episodes and larger networks.
- **Submitted with:** `sbatch fox_train.sh compare_100M` (3 times)
- **Code:** `10e4023` (2026-10-04), pulled on Fox one minute before the first submission. On top of `d80a55e`, it only deletes older episode stats snapshots, so the training settings are those of `d80a55e`. The resumed runs used `bf7e7fc` (see [Failure and resume](#failure-and-resume)).
- **Fox resources per algorithm:** 28 CPUs, 2 GB per CPU (56 GB), 60 h limit. The Slurm log of the DQN job `4259604_0` reports a memory peak of 29.3 GiB and an average CPU use of 20.5 %.
- **Status:** finished. 8 of the 9 runs failed after about 24 h because the disk quota was full, and were resumed from their newest checkpoint.

### What changed from run 2

| Change | Commit |
|---|---|
| 100M steps for all (was 10M) | `4a452da` |
| 24 parallel envs (was 12) | `4a452da` |
| 460-step episodes for all (was 230) | `4a452da` |
| Hidden layers [128, 128] for all (was [64, 64]) | `4a452da` |
| Checkpoint every 10M steps (was 1M) | `4a452da` |
| Fox: 28 CPUs, 56 GB, 60 h (was 16 CPUs, 32 GB, 16 h) | `d80a55e` |

### Task and budget

Values in bold changed from run 2. The algorithm settings are unchanged from run 2.

| Setting | DQN | PPO | SAC |
|---|---|---|---|
| Script and preset | `train_DDQN.py` | `train_PPO.py --compare` | `train_SAC.py` |
| GP / `SUB_AGENT_TRAIN_ON_GP` | on / on | on / on | on / on |
| Episode length (steps) | **460** | **460** | **460** |
| Total steps | **100M** | **100M** | **100M** |
| Parallel envs | **24** | **24** | **24** |
| `gamma` | 0.995 | 0.995 | 0.995 |
| Hidden layers | **[128, 128]** ReLU | **pi [128, 128], vf [128, 128]** Tanh | **[128, 128]** ReLU |
| Checkpoint every | **10M steps** | **10M steps** | **10M steps** |
| Start position | near the plume | near the plume | near the plume |
| Plume size | σ random in [2.0, 5.0] | σ random in [2.0, 5.0] | σ random in [2.0, 5.0] |
| Wall-clock time (approx., 3 runs, first job + resume) | 38.4–49.0 h | 20.2–27.3 h | 49.6–53.9 h |

### Points to keep in mind when reading the results

- Several settings changed at once, so the difference from run 2 cannot be traced to a single parameter.
- The step limit doubled from 230 to 460, so the agent has twice as many steps to reach the goal. "Ended early", the episode length and the total reward are therefore not directly comparable with run 2.
- `gamma` stays 0.995, which looks about 200 steps ahead (1 / (1 − 0.995)), while episodes can now last 460 steps.
- DQN's `exploration_fraction` stays 0.2, which now means 20M steps of decreasing exploration (2M in run 2). `learning_starts` (400,000) and the replay buffers (DQN 800,000, SAC 900,000) are unchanged, so they are a smaller share of the run.
- PPO now collects `n_steps` 2048 × 24 envs = 49,152 steps per update (24,576 in run 2).
- The resumed DQN and SAC runs restarted with an empty replay buffer (see [Failure and resume](#failure-and-resume)), so they are not quite the same as uninterrupted runs.

### Failure and resume

On 2026-10-06, 8 of the 9 jobs stopped with state FAILED: 7 between 04:09 and 04:10, and `PPO_compare_100M_4259610` at 04:33. Only `PPO_compare_100M_4259604` finished in one job. No job was close to the 60 h limit.

- **Cause:** the disk quota of the home directory on Fox was full. The Slurm log of `4259604_0` (DQN) ends with `OSError: [Errno 122] Disk quota exceeded`, raised while TensorBoard wrote its log file. All runs write to `../trained-agents` in the home directory, so they share one quota. The logs of the other 7 failed jobs were not checked, but they stopped in the same minute (`PPO_compare_100M_4259610` 24 min later), so the cause is very likely the same.
- **Code changes before resuming:** `b1813c6` writes the stats snapshots through a temporary file, so a failed write no longer leaves a broken snapshot. `bf7e7fc` adds the `--resume` option to the training scripts and to `fox_train.sh`, and saves the DQN and SAC stats every 10M steps (was 5M), at the same steps as the checkpoints.
- **Resumed with:** `sbatch --array=0,2 fox_train.sh compare_100M 4259604` (DQN and SAC only), `sbatch fox_train.sh compare_100M 4259607` and `sbatch fox_train.sh compare_100M 4259610`, on 2026-10-08/09 with code `bf7e7fc`.
- **What a resume keeps:** the networks, the step count and DQN's ε schedule (already at its final 0.05) continue from the newest checkpoint. The episode stats are cut at the checkpoint step, so episodes from the steps that ran twice are only counted once (`DQN_compare_100M_4259604` dropped 28,535 episodes). Each run folder has a `resume_log.txt` with the checkpoint and the stats file it used.
- **What a resume loses:** the DQN and SAC replay buffers are not saved, so they start empty. The step count is already past `learning_starts`, so training restarts at once on a nearly empty buffer. PPO has no replay buffer.

| Run | First job | First job reached | Resumed from | Steps run twice | Resume job | Total time |
|---|---|---|---|---|---|---|
| `DQN_compare_100M_4259604` | `4259604_0`, 29.6 h | 65–70M | 60M | 5–10M | `4272086_0`, 19.3 h | 49.0 h |
| `DQN_compare_100M_4259607` | `4259607_0`, 24.1 h | 65–70M | 60M | 5–10M | `4272088_0`, 14.2 h | 38.4 h |
| `DQN_compare_100M_4259610` | `4259610_0`, 23.7 h | 60–65M | 60M | 0–5M | `4272091_0`, 15.9 h | 39.6 h |
| `PPO_compare_100M_4259604` | `4259604_1`, 20.2 h | 100M (finished) | – | – | – | 20.2 h |
| `PPO_compare_100M_4259607` | `4259607_1`, 24.1 h | 100M | 90M | 10M | `4272088_1`, 3.2 h | 27.3 h |
| `PPO_compare_100M_4259610` | `4259610_1`, 23.8 h | 100M | 90M | 10M | `4272091_1`, 1.4 h | 25.2 h |
| `SAC_compare_100M_4259604` | `4259604_2`, 24.3 h | 40–45M | 40M | 0–5M | `4272086_2`, 25.3 h | 49.6 h |
| `SAC_compare_100M_4259607` | `4259607_2`, 24.1 h | 45–50M | 40M | 5–10M | `4272088_2`, 29.8 h | 53.9 h |
| `SAC_compare_100M_4259610` | `4259610_2`, 23.1 h | 40–45M | 40M | 0–5M | `4272091_2`, 30.1 h | 53.2 h |

- The times are the `Elapsed` times from `sacct`. "Total time" is the sum of both jobs and includes the steps that ran twice.
- "First job reached" follows from the stats snapshots in the resume logs: the first job wrote the newest snapshot that the resume found, but not the next one (snapshots every 5M steps for DQN and SAC, every 10M for PPO).
- Both resumed PPO runs left an unreadable snapshot at 99,999,840 steps. They failed while writing it, 160 steps before the 100M limit and before the last checkpoint was saved, so they resumed from 90M.

### Results

Mean ± std over the last 1,000 training episodes of each run:

| Run | Episodes | Total reward | Episode length | Visited (%) | Plume coverage (%) | Ended early (%) |
|---|---|---|---|---|---|---|
| `DQN_compare_100M_4259604` | 479,980 | -60.7 ± 116.8 | 160.3 ± 148.7 | 23.4 ± 17.9 | 95.3 ± 20.2 | 87.1 |
| `DQN_compare_100M_4259607` | 448,254 | -70.7 ± 120.7 | 176.0 ± 150.6 | 25.4 ± 18.9 | 97.1 ± 15.2 | 87.4 |
| `DQN_compare_100M_4259610` | 463,638 | -62.6 ± 118.0 | 161.8 ± 152.1 | 22.0 ± 17.2 | 87.3 ± 32.7 | 84.7 |
| **DQN_compare_100M (mean of 3 runs)** |  | **-64.7 ± 4.3** | **166.0 ± 7.1** | **23.6 ± 1.4** | **93.3 ± 4.3** | **86.4 ± 1.2** |
| `PPO_compare_100M_4259604` | 2,477,751 | 32.0 ± 5.2 | 33.5 ± 16.2 | 8.1 ± 3.2 | 99.8 ± 0.8 | 100.0 |
| `PPO_compare_100M_4259607` | 2,557,236 | 32.7 ± 4.8 | 32.2 ± 15.0 | 8.0 ± 3.1 | 99.8 ± 1.0 | 100.0 |
| `PPO_compare_100M_4259610` | 2,455,487 | 31.6 ± 6.3 | 33.2 ± 16.5 | 7.8 ± 2.9 | 99.9 ± 0.7 | 100.0 |
| **PPO_compare_100M (mean of 3 runs)** |  | **32.1 ± 0.4** | **33.0 ± 0.6** | **8.0 ± 0.1** | **99.8 ± 0.0** | **100.0 ± 0.0** |
| `SAC_compare_100M_4259604` | 878,218 | -22.2 ± 105.0 | 88.4 ± 84.6 | 8.8 ± 4.7 | 99.3 ± 4.7 | 98.2 |
| `SAC_compare_100M_4259607` | 1,036,931 | -6.1 ± 64.2 | 72.1 ± 68.2 | 8.4 ± 4.2 | 98.7 ± 9.4 | 98.8 |
| `SAC_compare_100M_4259610` | 938,244 | -19.9 ± 87.8 | 87.4 ± 88.7 | 8.7 ± 4.7 | 97.7 ± 13.0 | 97.1 |
| **SAC_compare_100M (mean of 3 runs)** |  | **-16.1 ± 7.1** | **82.6 ± 7.4** | **8.6 ± 0.1** | **98.6 ± 0.6** | **98.0 ± 0.7** |

**DQN**

![DQN compare_100M 4259604](training_runs/DQN_compare_100M_4259604/training_stats.png)
![DQN compare_100M 4259607](training_runs/DQN_compare_100M_4259607/training_stats.png)
![DQN compare_100M 4259610](training_runs/DQN_compare_100M_4259610/training_stats.png)

**PPO**

![PPO compare_100M 4259604](training_runs/PPO_compare_100M_4259604/training_stats.png)
![PPO compare_100M 4259607](training_runs/PPO_compare_100M_4259607/training_stats.png)
![PPO compare_100M 4259610](training_runs/PPO_compare_100M_4259610/training_stats.png)

**SAC**

![SAC compare_100M 4259604](training_runs/SAC_compare_100M_4259604/training_stats.png)
![SAC compare_100M 4259607](training_runs/SAC_compare_100M_4259607/training_stats.png)
![SAC compare_100M 4259610](training_runs/SAC_compare_100M_4259610/training_stats.png)

### Observations

- **Spread between repeats:** very small for PPO (the run means of the reward range from 31.6 to 32.7). Larger for SAC (−22.2 to −6.1) and for DQN, where `4259610` has a lower plume coverage (87.3 %) than the other two (95–97 %).
- **DQN:** the reward rises from about −1,000 to about −100 in the first ~100k episodes, then improves slowly to about −65 over the remaining ~350k. The curves stay noisy, with repeated dips in the reward and jumps in the episode length during the whole run. The episode length ends at about 165 steps (std about 150), and 86 % of the episodes end early (37.5 % in run 2). It visits about 24 % of the map, about 3× as much as PPO and SAC.
- **PPO:** the reward reaches about +30 within the first ~0.2M of about 2.5M episodes and then stays flat, while the episode length keeps falling slowly to about 33 steps. All of the last 1,000 episodes end early, with 99.8 % plume coverage and 8 % of the map visited. Most of the training budget was spent after the reward had levelled off.
- **SAC:** as in run 2, the episode length stays near the step limit during the first ~10k–20k episodes, and the plume coverage drops to about 30–35 % in this phase. Then the reward rises fast to about −50 and improves slowly to about −16 at the end. 98 % of the last 1,000 episodes end early (69.6 % in run 2), after about 83 steps with 9 % of the map visited. The episode length stays noisier than PPO's (std about 85, against 16 for PPO).
- **Resume:** `DQN_compare_100M_4259604` shows a dip in the reward and a rise in the episode length near episode 250k, close to where it resumed (episode 247,716). The DQN curves have similar dips elsewhere, so the plots alone do not show whether the empty replay buffer caused it. The SAC plots show no clear change where the runs resumed (episodes 258k–314k).
- **Wall-clock time:** PPO was the fastest, and the only algorithm with a run that finished in one job (20.2 h for 100M steps). In the about 24 h before the failure, the first jobs reached 60–70M steps for DQN and 40–50M for SAC.

## HRL meta agent

From run 4 on, the runs train the meta agent of the HRL setup. The meta agent (PPO) chooses which of three trained sub-agents moves the AUV next, and the chosen sub-agent then runs a few steps (an option). The sub-agents are frozen and act deterministically.

### Settings shared by the HRL runs

- **Script:** [train_HRL(ppo).py](<../HUGIN_gym/training/train_HRL(ppo).py>), submitted with [fox_train_HRL.sh](../fox_train_HRL.sh) as `sbatch fox_train_HRL.sh <tag>`. From run 5 on, the tag also selects the meta agent variant (`PRESETS` in the script). The runs are saved as `../trained-agents/HRL_PPO_<tag>_<job ID>`.
- **Sub-agents:**

  | Sub-agent | Model | Observation keys |
  |---|---|---|
  | Space exploration | `../trained-agents/PPO_space_30M` | `obs_state`, `visited_maps_downsampled`, `space_coverage` |
  | Plume | `../trained-agents/PPO_compare_100M_4259607` (run 3) | `obs_state`, `local_GT`, `plume_coverage`, `space_coverage`, `distance_from_max` |
  | Border | `../trained-agents/PPO_border_4280885` | `obs_state`, `local_GT_around`, `border_coverage`, `space_coverage`, `distance_from_max` |

- **Environment:** the 41 × 41 map of runs 1–3 with one Gaussian source, σ random in [2.0, 5.0]. The AUV starts in a random corner, heading along an edge into the map, like a lawnmower survey (`spawn_at_survey_start`). GP on (RBF, length scale 3.5): the meta agent and the sub-agents only see the GP's estimate of the plume, never the true plume.
- **Survey budget:** 480 AUV steps per episode.
- **Goals:** accuracy goals [0.9, 0.9, 0.9].
- **Meta reward** ([HRL_env.py](../HUGIN_gym/envs/HRL_env.py), [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)), summed over the steps of an option and discounted by 0.998 per step:
  - −1 per step, plus a reward for new cells that depends on the chosen sub-agent. Space exploration gets (new visited cells) / 6, but only while the plume and border estimates are at least 90 % covered and no new plume or border cells are found. Plume and border get (new border cells + new visited cells) / 6 when they find new border cells, otherwise (new visited cells) / 12.
  - +40 once per plume when the estimated plume and border are both at least 90 % covered.
  - +40 and the episode ends when the mean GP standard deviation over the map is 0.1 or lower and both coverages are at least 90 %.
  - −4 instead of the step reward when the AUV tries to leave the map, and −2 when an option ends after 3 steps without new cells.
- **Meta agent (PPO):** 12 parallel envs, `n_steps` 512, `batch_size` 1024, `n_epochs` 4, `learning_rate` 3e-4, `gae_lambda` 0.95, `clip_range` 0.2, `ent_coef` 0.01, `vf_coef` 0.5, `max_grad_norm` 0.5, hidden layers pi [64, 64] and vf [64, 64] Tanh (SB3 default). `gamma` 0.998 per AUV step, which is 0.998³ ≈ 0.994 per 3-step decision. 10M decisions, with a checkpoint and a stats snapshot every 2,049,996 decisions.
- **Fox:** 16 CPUs, 2 GB per CPU (32 GB). Run 4 made 103–125 decisions per second and used at most 7.1 GiB of memory.
- **Counting:** the meta agent's step count, the episode length in the stats and `rollout/ep_len_mean` count meta decisions, not AUV steps.

## Run 4: HRL `untuned` (job 4363368)

- **Goal:** first training of the meta agent, with the settings from the thesis code. It is the baseline for run 5.
- **Submitted with:** `sbatch fox_train_HRL.sh untuned`, resumed twice with `sbatch fox_train_HRL.sh untuned 4363368` (jobs 4367840 and 4371199).
- **Code:** `56ba1c7` (2026-10-28 14:57), pulled on Fox at 14:58, for all three jobs. The next pull on Fox was 2026-10-29 22:55, after the last job had started.
- **Fox resources:** 16 CPUs, 32 GB, 12 h limit.
- **Failed first submission:** job 4363336 (2026-10-28 14:59) failed after 19 s with a `FileNotFoundError`: `PPO.load` did not find the space sub-agent `../trained-agents/PPO_space_30M/PPO_scratch.zip` (`slurm-hugin-hrl-4363336.out` on Fox). The error shows the path as `PPO_scratch.zip.zip`, because SB3 also tries the name with `.zip` added. Job 4363368 started 13 min later with the same code and loaded the model, so the file was in place by then.
- **Meta agent:** 3 actions, one per sub-agent, and every option runs 3 steps, so an episode has at most 160 decisions. The meta state is 158 inputs, turned into 128 features (ReLU) by `AgentHRLExplore` ([agent0_HRL.py](../HUGIN_gym/agents/feature_extractor/agent0_HRL.py)): the three coverages, position and heading, 7 × 7 patches of the visited map and of the plume and border estimates, and `distance_from_max`. With the GP on, `distance_from_max` is always 1.0.
- **Status:** finished. The third job completed on 2026-10-30 01:23 after 10,000,176 decisions (76,478 episodes). The final model is `HRL_PPO_scratch.zip` in the run folder.

### Time limit and resumes

The 12 h limit was too short for 10M decisions, so the run needed three jobs. Each job lost the decisions after its last checkpoint: about 1 h in the first job and about 1.5 h in the second.

| Job | Node | Slurm start | Slurm end | Elapsed | State | Starts from decision | Saved last | Saved at | Speed | Peak memory |
|---|---|---|---|---|---|---|---|---|---|---|
| 4363368 | c1-19 | 2026-10-28 15:12 | 2026-10-29 03:12 | 12:00:18 | TIMEOUT | 0 | checkpoint 4,099,992 | 2026-10-29 02:16 | about 103 decisions/s | 5.7 GiB |
| 4367840 | c1-13 | 2026-10-29 08:57 | 2026-10-29 20:57 | 12:00:06 | TIMEOUT | 4,099,992 | checkpoint 8,199,984 | 2026-10-29 19:24 | about 109 decisions/s | 7.1 GiB |
| 4371199 | c1-9 | 2026-10-29 21:21 | 2026-10-30 01:23 | 04:01:22 | COMPLETED | 8,199,984 | final model 10,000,176 | 2026-10-30 01:23 | about 125 decisions/s | 7.1 GiB |

- The Slurm times, the states and the peak memory (`MaxRSS` of the batch step) come from `sacct`. The nodes come from the TensorBoard event files and the Slurm log. The run's own start times (`RUN_INFO.timestamp` in `training_config.json` and the times in `resume_log.txt`) are about 30 s later, after the envs are set up. The speeds are measured from those times to the time the last model was saved.
- **Total:** 28.0 h in three jobs. About 2.5 h of it repeated decisions that were lost after the last checkpoint of the first two jobs, so the 10M decisions took about 25.5 h. At the measured speeds, 10M decisions take 22–27 h in one job, so `fox_train_HRL.sh` now asks for 30 h.
- **Third job:** faster than the first two (about 125 against 103 and 109 decisions/s) and on another node. Its Slurm log reports "Job exited normally". On average it used 37 % of its 16 CPUs (23.7 CPU hours in 4.0 h).
- **Memory:** at most 7.1 GiB of the 32 GiB requested. 16 GB (1 GB per CPU) would still leave more than twice the peak.

### Learning curve

From `training_stats.pkl` (76,478 episodes, 10M decisions), split into 20 groups of equal size, in order. "Plume coverage" is the true coverage. It is missing for 1,109 episodes (1.5 %), so its groups line up only roughly with the decisions.

| Up to decision | Total reward | Episode length (decisions) | Ended early (%) | Visited (%) | Plume coverage (%) |
|---|---|---|---|---|---|
| 554,383 | -374.0 | 145.0 | 50.6 | 79.6 | 88.8 |
| 1,097,952 | -353.7 | 142.1 | 55.6 | 80.6 | 88.9 |
| 1,611,045 | -300.9 | 134.2 | 68.9 | 81.6 | 91.3 |
| 2,110,621 | -297.8 | 130.6 | 80.7 | 87.5 | 92.4 |
| 2,611,017 | -302.9 | 130.9 | 79.1 | 87.3 | 92.2 |
| 3,111,572 | -303.2 | 130.9 | 79.8 | 87.3 | 92.1 |
| 3,613,518 | -305.5 | 131.3 | 78.7 | 87.4 | 92.1 |
| 4,117,004 | -308.9 | 131.7 | 77.5 | 87.3 | 92.0 |
| 4,619,507 | -308.3 | 131.4 | 77.7 | 87.3 | 92.2 |
| 5,118,084 | -301.6 | 130.4 | 79.6 | 87.1 | 92.3 |
| 5,623,881 | -313.0 | 132.3 | 75.0 | 86.8 | 91.6 |
| 6,132,296 | -291.4 | 133.0 | 69.9 | 82.0 | 91.5 |
| 6,635,249 | -249.6 | 131.5 | 75.1 | 78.3 | 93.3 |
| 7,131,877 | -230.4 | 129.9 | 78.5 | 77.8 | 94.8 |
| 7,619,857 | -212.8 | 127.6 | 80.0 | 78.1 | 95.2 |
| 8,099,902 | -195.6 | 125.5 | 83.6 | 77.5 | 95.7 |
| 8,580,825 | -189.2 | 125.8 | 83.8 | 77.2 | 95.7 |
| 9,057,668 | -184.9 | 124.7 | 85.0 | 77.1 | 96.2 |
| 9,528,872 | -177.4 | 123.3 | 85.5 | 77.1 | 96.4 |
| 9,997,920 | -171.0 | 122.7 | 85.2 | 77.3 | 96.2 |

- "Ended early" means the episode reached the termination goal before the limit of 160 decisions.
- The last row ends at 9,997,920 decisions: the other 2,256 of the 10,000,176 decisions belong to episodes that had not finished when the run stopped.
- While the run was going, this table used the snapshot `episode_stats_8199984.pkl` (61,967 episodes up to 8.2M decisions). The method is the same, but the group boundaries have moved because there are more episodes now.

### Results

Mean ± std over the last 1,000 training episodes:

| Run | Episodes | Total reward | Episode length (decisions) | Visited (%) | Plume coverage (%) | Ended early (%) |
|---|---|---|---|---|---|---|
| `HRL_PPO_untuned_4363368` | 76,478 | -162.8 ± 163.2 | 122.0 ± 22.0 | 77.0 ± 5.5 | 96.0 ± 10.8 | 86.2 |

![HRL PPO untuned 4363368](training_runs/HRL_PPO_untuned_4363368/training_stats.png)

- **Plot panels:** "Episode Length" counts decisions. "Steps Around Threshold (%)" is the true border coverage, and "Steps Above Threshold (%)" the true plume coverage.
- **Reward panel:** the rightmost panel shows the reward of each decision in every 10th episode. The spikes to about 80 are decisions in which both +40 bonuses fall, the coverage bonus and the end bonus. 99 % of the decisions with a reward above 60 are the last decision of their episode. Such a decision occurs in 25 % of the episodes in the first fifth of the training and in 9 % in the last fifth.

### Observations

- **Three phases:**
  - Up to about 2.1M decisions, fast improvement: the reward goes from −374 to −298, and the share that ends early from 51 % to 81 %.
  - From 2.1M to about 5.6M, flat at about −305.
  - From about 5.8M to the end, a second improvement: the reward goes from −313 to −171, and the plume coverage from 91.6 % to 96.2 %. The share that ends early first dips to 70 % at 6.1M and then rises to 85 %. The improvement slows down towards the end (−185 to −171 over the last 0.9M decisions), but it has not stopped.
- **Less of the map visited:** in the second improvement the visited share drops from 87 % to 77 %. The agent reaches the termination goal with less of the map visited.
- **Policy entropy:** from `train/entropy_loss` in TensorBoard.
  - **Start:** 1.10, a uniform choice among the three sub-agents.
  - **Flat phase:** it falls to almost 0 (0.007 at its lowest, at 1.9M, and 0.02–0.05 from 2M to 5M). The agent then made almost the same choice every time in a given situation.
  - **Second improvement:** it rises again to about 0.5 at 6.5–7M.
  - **End:** 0.33.

  The run alone does not show whether the rise caused the improvement. Because the final policy is still fairly random, its deterministic actions do much worse than its sampled ones (see [Evaluation of the trained agent](#evaluation-of-the-trained-agent)).
- **Value function:** the explained variance (`train/explained_variance`) goes from about 0 at the start to 0.80 at 4–5M and 0.63 at the end.
- **Run length:** a run of 5M decisions would have stopped in the flat phase. Run 5 therefore also uses 10M decisions.
- **Resumes:** the second improvement starts at about 5.8M, not at the resume at 4.1M. The curve shows no change at the resume at 8.2M.

### Evaluation of the trained agent

On 2026-10-30 the final model was tested on 1,000 new episodes with [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>). The code is `a73297d` plus that day's evaluation changes, which were not yet committed.

- **Test env:** the same as in training. The AUV starts in a corner, there is one source with σ in [2.0, 5.0], the GP is on (RBF, length scale 3.5), the goals are [0.9, 0.9, 0.9] and the survey budget is 480 steps (160 decisions). The script reads the run settings from `training_config.json`. Run 4 has no `HRL_VARIANT` there, so the script uses the `untuned` settings.
- **Same episodes:** seed 0. The env only draws the plumes and start positions when it resets, so every mode gets the same 1,000 episodes in the same order.
- **Two ways to choose:** at each decision, the meta agent's network gives a probability for each sub-agent.
  - **Sampled (mode 4):** draws the sub-agent with these probabilities, as PPO does in training. This shows the policy as it was trained.
  - **Deterministic (mode 1):** always takes the most likely sub-agent.

  The sub-agents always act deterministically.
- **Run with** the following command from the repository root. Modes 1 and 4 took about 44 min on a Mac.

  ```bash
  PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py"
  ```

  Enter `1` (deterministic), `4` (sampled) or `5` (watch in Meshcat, see [Watching episodes](#watching-episodes-mode-5)).

Mean ± std over the 1,000 test episodes, next to the last 1,000 training episodes:

| Metric | Output line | Training | Sampled (mode 4) | Deterministic (mode 1) |
|---|---|---|---|---|
| Ended early (%) | `SUCCESS_RATE` | 86.2 | 86 | 51 |
| Total reward | `REWARD` | -162.8 ± 163.2 | -173.9 ± 158.1 | -396.6 ± 418.2 |
| Decisions | `Options_CHOSEN` | 122.0 ± 22.0 | 125.9 ± 20.5 | 134.7 ± 27.7 |
| AUV steps | `STEPS` | – | 376.7 ± 61.6 | 403.6 ± 83.5 |
| Turns | `TURNS` | – | 113.8 ± 39.8 | 194.1 ± 140.6 |
| Visited (%) | `SPACE` | 77.0 ± 5.5 | 76 ± 6 | 55 ± 27 |
| True plume coverage (%) | `ACTUAL PLUME` | 96.0 ± 10.8 | 97.1 ± 9.4 | 95.5 ± 12.8 |
| True border coverage (%) | `ACTUAL BORDER` | – | 95.7 ± 8.7 | 93.6 ± 12.0 |
| Plume coverage of the GP map (%) | `PLUME` | – | 98 ± 6 | 95 ± 13 |
| Border coverage of the GP map (%) | `BORDER` | – | 96 ± 6 | 93 ± 13 |
| Choices: space / plume / border | `CHOICES` | – | 0.74 / 0.12 / 0.14 | 0.48 / 0.34 / 0.18 |
| RMSE of the GP concentration map | `RMSE` | – | 0.0023 ± 0.0080 | 0.0050 ± 0.0157 |
| True positives (share of the 1,681 cells) | `TP` | – | 0.0540 | 0.0534 |
| False positives | `FP` | – | 0.0024 | 0.0025 |
| False negatives | `FN` | – | 0.0002 | 0.0004 |

- **`SUCCESS_RATE`** is the share of episodes that reached the termination goal before the survey budget ran out, the same as "Ended early" in training.
- **`TP`, `FP` and `FN`** compare the GP's plume map (GP mean above 0.3) with the true plume at the end of the episode, as shares of all 1,681 cells. `TN` (0.943 sampled, 0.944 deterministic) is left out.
- **The true coverages** are missing for episodes whose last step hit the map edge, so those two rows cover slightly fewer episodes.

**Observations:**

- **Sampled actions reproduce training:** 86 % end early (86.2 % in training), with 125.9 decisions (122.0), 76 % of the map visited (77 %) and 97.1 % true plume coverage (96.0 %). So the test env matches the training conditions.
- **Deterministic actions do much worse:** only half the episodes end early. The visited share varies widely (55 ± 27 %), and the agent turns 70 % more.
  - **Why:** in the same situation the meta agent always picks the same sub-agent, and the sub-agents are deterministic too. When the AUV comes back to a situation it has been in before, it repeats the same moves and can circle until the budget runs out. Sampling breaks such loops by chance.
  - **Example:** deterministic episode 3 of seed 0 visits 1.8 % of the map in 480 steps.
  - This fits the entropy of 0.33 at the end of training: the policy still depends on its randomness.
- **The plume is found in both modes:** the true plume coverage is 97.1 % and 95.5 %. The GP map finds over 99 % of the true plume cells (TP / (TP + FN): 99.6 % sampled, 99.3 % deterministic). The deterministic agent mostly fails at covering the map, which the end goal needs (mean GP standard deviation 0.1 or lower).
- **The false positives are the map corners:** `FP` (0.0024) is almost exactly the 4 corner cells (4 / 1,681 = 0.0024), which the GP map always marks as plume (`_confusion_matrix` in [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)).
- **Which to report:** the sampled results show the policy as it was trained. Report the deterministic results next to them, and say which is which.

#### Reported coverage against true coverage

- **Two measures:** `PLUME` and `BORDER` are measured against the GP map. This is the coverage the meta agent sees, and the one its coverage goal and bonus use. It is the share of the cells the GP marks as plume (GP mean above 0.3) or border (0.15 to 0.45) that the AUV has visited. `ACTUAL PLUME`, `ACTUAL BORDER` and the plume coverage in the training stats are measured against the true plume.
- **Usually they agree:** in training, the GP-based plume coverage at the end of an episode was within 1 point of the true one in 91 % of the 75,369 episodes with both values (border: 87 %). Over the last 1,000 training episodes the means are 96.5 % against 96.0 % (plume) and 95.5 % against 94.9 % (border).
- **Exception:** while no cell of the GP map is above 0.3, the GP threshold for the plume drops to 0.15 (`step` in [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)). The "plume" is then the weak gas the AUV has measured around itself, and circling there gives a high reported coverage.
  - **Example:** deterministic episode 3 of seed 0. The true plume has 141 cells and the GP plume 19 (plus the 4 corners). The reported plume coverage is 87 %, the true one 0 %. The AUV circled in the weak gas outside the plume for all 480 steps.
- **How often:** at the end of the episode, the GP-based plume coverage was at least 90 % while the true one was below 50 % in 0.15 % of the training episodes (0.3 % of the last 1,000).
- **Effect on the reward:** the +40 coverage bonus is paid when the GP-based plume and border coverages both reach 90 %, so it can also be paid for such a patch of weak gas. How often this happened in training was not counted.
- **For the article:** use the true coverages (`ACTUAL PLUME`, `ACTUAL BORDER` and the plume coverage in the training stats).

#### Watching episodes (mode 5)

- **What it shows:** the first 5 episodes of seed 0 in the Meshcat browser view, slowed down to 0.1 s per AUV step, with a pause after each episode. Mode 5 asks for deterministic (`d`) or sampled (`s`) actions. The settings are at the top of the script (`STEP_DELAY`, `WATCH_EPISODES`, `OPTION_COLORS`).
- **Top left:** the sub-agent in control. The AUV's trail has the same color: green for space exploration, pink for plume, yellow for border.
- **Gas cloud:** colored by what the coverage counts as plume and border, with a legend in the bottom left:
  - pale cyan: plume only (above 0.45);
  - light purple: plume and border (0.30 to 0.45);
  - dark purple: border only (0.15 to 0.30).
- **Window shape:** keep the browser window at least 1.5 times as wide as it is high, or the labels are cut off.

## Run 5: HRL meta agent variants

- **Goal:** test two questions from the project supervisor.
  1. Can the meta agent learn how long to use a sub-agent in different situations? It chooses between a 2-step and a 5-step variant of each sub-agent. We hope to see it choose the 5-step space exploration early in the survey.
  2. How little information does the meta agent need? The meta state goes from 158 inputs to 6 numbers, with and without each sub-agent's own estimate of how well it will do.
- **Submitted:** 2026-10-29 22:58, all five at once. All five started at 22:58:04 (`sacct`), while the last job of run 4 was still running.

  | Tag | Job | Submitted with | Run folder | Started (`RUN_INFO.timestamp`) |
  |---|---|---|---|---|
  | `minimal` | 4371279 | `sbatch fox_train_HRL.sh minimal` | `HRL_PPO_minimal_4371279` | 2026-10-29 22:58:23 |
  | `minimal_values` | 4371280 | `sbatch fox_train_HRL.sh minimal_values` | `HRL_PPO_minimal_values_4371280` | 2026-10-29 22:58:39 |
  | `duration` | 4371281 | `sbatch --time=2-06:00:00 fox_train_HRL.sh duration` | `HRL_PPO_duration_4371281` | 2026-10-29 22:58:39 |
  | `duration_mean` | 4371282 | `sbatch --time=2-06:00:00 fox_train_HRL.sh duration_mean` | `HRL_PPO_duration_mean_4371282` | 2026-10-29 22:58:39 |
  | `minimal_values_duration` | 4371283 | `sbatch --time=2-06:00:00 fox_train_HRL.sh minimal_values_duration` | `HRL_PPO_minimal_values_duration_4371283` | 2026-10-29 22:58:39 |

- **Code:** `152ba0a` (2026-10-29 22:54), pulled on Fox at 22:55. The code changes are in the four commits before it (`92c4352`, `af15f11`, `36c770f`, `28d07d3`). `152ba0a` itself only adds this documentation.
- **Fox resources:** 16 CPUs, 32 GB per run. 30 h limit for the 3-step variants. 54 h for the duration variants, because their 10M decisions can be up to 50M AUV steps.
- **Status:** finished. All five jobs completed in one job each, without a resume, after 10,002,432 decisions. The first finished on 2026-10-30 14:29 (`minimal_values_duration`), the last on 2026-10-31 08:50 (`duration_mean`). The final models are `HRL_PPO_scratch.zip` in the run folders. See [Jobs](#jobs) and the results below.

### In short: what each variant does

All variants use the same sub-agents, map, survey start, 480-step survey budget and goals as run 4 (`untuned`). Only the meta agent changes.

- **`minimal`:** as `untuned`, but the meta agent sees only 6 numbers. They are: how much of the survey time is used, how much of the map, the plume and the plume border is covered, whether a plume has been found, and how uncertain the GP map still is. *Question:* is that enough to choose the sub-agents well, and does it learn in fewer decisions?
- **`minimal_values`:** as `minimal`, plus each sub-agent's own estimate of how well it would do if chosen now. *Question:* can these estimates stand in for the local maps that `minimal` leaves out?
- **`duration`:** as `untuned`, but each choice also sets how long the sub-agent runs: 2 or 5 steps. A small cost per choice makes the long variant pay off when it does as well as two short ones. *Question:* does the meta agent learn when to commit for longer, for example long space exploration early in the survey?
- **`duration_mean`:** as `duration`, but a choice is rewarded with the average of its step rewards, as the supervisor suggested. *Question:* does averaging bias the choice of length? We expect it to favor 5 steps whatever the situation.
- **`minimal_values_duration`:** both ideas together: the 6 numbers and the sub-agent estimates, and the choice of 2 or 5 steps.

### The variants

| Tag | Actions | Reward per decision | Discount per decision | Cost per decision | Meta state |
|---|---|---|---|---|---|
| `untuned` (run 4) | 3 sub-agents × 3 steps | Sum of the step rewards | 0.998³ | 0 | Full (158 inputs) |
| `duration` | 3 sub-agents × {2, 5} steps | Sum | 0.998^k, k = steps run | 0.1 | Full |
| `duration_mean` | 3 sub-agents × {2, 5} steps | Mean of the step rewards; bonuses and penalties in full | 0.998³ | 0.1 | Full |
| `minimal` | 3 sub-agents × 3 steps | Sum | 0.998³ | 0 | Minimal (6 numbers) |
| `minimal_values` | 3 sub-agents × 3 steps | Sum | 0.998³ | 0 | Minimal + 3 sub-agent values |
| `minimal_values_duration` | 3 sub-agents × {2, 5} steps | Sum | 0.998^k | 0.1 | Minimal + 3 sub-agent values |

Each comparison changes one thing:

| Compare | To see the effect of |
|---|---|
| `duration` with `untuned` | Choosing the option length |
| `duration_mean` with `duration` | The mean reward suggested by the supervisor, against the sum with a discount per decision |
| `minimal` with `untuned` | The small meta state |
| `minimal_values` with `minimal` | The sub-agent values |
| `minimal_values_duration` with `minimal_values` and `duration` | Both ideas together |

### Choosing the option length (`duration`, `duration_mean`, `minimal_values_duration`)

- **Actions:** 6, one per sub-agent and length: `space_2`, `space_5`, `plume_2`, `plume_5`, `border_2`, `border_5`. An option still ends after 3 steps without new cells (−2), so a 5-step option can stop after 3 steps.
- **Survey budget in AUV steps:** the episode ends after 480 AUV steps, not after a number of decisions. Otherwise the 5-step variants would give a longer survey, and the agent would prefer them for that reason alone. The decision limit is 240 (480 / 2), so it never ends an episode first.
- **Sum and discount per decision (`duration`, `minimal_values_duration`):** a 5-step decision covers more time than a 2-step decision. Its step rewards are summed, and the next decision is discounted by 0.998⁵ instead of 0.998². This is the semi-MDP form of the options framework (Sutton, Precup and Singh, 1999). A 2-step and a 5-step choice are then compared over the same time, and the reward scale favors neither.
  - SB3's PPO uses one discount for every step. These runs use `SMDPDictRolloutBuffer` ([smdp_rollout_buffer.py](../HUGIN_gym/agents/smdp_rollout_buffer.py)), which gets the discount of each decision from the env through `SMDPDiscountCallback`.
  - One small approximation remains. When an episode hits the step limit, SB3 bootstraps it with 0.998³ instead of the last decision's own discount. The difference is below 1 %.
- **Cost per decision (0.1):** this is the "slightly higher reward for the variant with more actions". A 5-step option makes fewer decisions, so it pays less cost per AUV step. 0.1 is 10 % of the time penalty of one step. Harb et al. (2018) call this a deliberation cost.
- **Mean reward (`duration_mean`):** the supervisor's suggestion. The step rewards are averaged over the option, so that the 5-step variant does not win by adding up more rewards. The bonuses and penalties count in full, and every decision has the same discount, 0.998³.
  - The episode's return is still a sum over decisions, and long options make fewer decisions.
  - When the step rewards are mostly negative, fewer decisions give a less negative return, whatever the situation. In run 4 the reward averages about −0.5 per AUV step (−194 per episode of about 377 steps at 8.2M).
  - We therefore expect `duration_mean` to choose the 5-step variants more often than `duration`. The run tests this.
- **Related methods:** choosing an action together with how long to repeat it is also used in Dynamic Action Repetition (Lakshminarayanan, Sharma and Ravindran, 2017) and FiGAR (Sharma, Lakshminarayanan and Ravindran, 2017).

### Minimal meta states (`minimal`, `minimal_values`, `minimal_values_duration`)

| Input | Meaning |
|---|---|
| `time_used` | Share of the 480-step survey used |
| `space_coverage` | Share of the map visited |
| `plume_coverage` | Share of the estimated plume visited |
| `border_coverage` | Share of the estimated plume border visited |
| `plume_found` | 1 when the GP estimate holds a plume, otherwise 0 |
| `gp_uncertainty` | Mean GP standard deviation over the map. The termination goal needs 0.1 or lower |

- **Why `plume_found`:** before a plume is found, the plume and border estimates only hold the four corner cells, which count as visited. The coverages then show 1.0, the same value as a fully covered plume.
- **Why `time_used`:** the run 4 meta state has no time input, although the episodes have a fixed limit.
- **Sub-agent values (`minimal_values`, `minimal_values_duration`):** each sub-agent's own PPO critic estimates its expected return if it is chosen now. The three values are scaled with symlog, sign(v) · log(1 + |v|), because the sub-agents have different reward scales. The critics learned in the sub-agents' own training situations, for example the plume agent always started near the plume. So the values may be off in a full survey, which these runs also show.
- **Network:** SB3's default for dict observations, which concatenates the inputs, followed by [64, 64] Tanh. `AgentHRLExplore` is not used.
- **Expected effect:** fewer decisions needed to learn, not faster decisions. Most of the time per decision most likely goes to the GP update after every AUV step and to the three sub-agents, not to the meta network. This has not been profiled.

### Changes for all HRL runs from run 5 on

| Change | Why |
|---|---|
| The tag selects the variant (`--preset`) | One job script for all variants |
| The survey budget is counted in AUV steps (480) | Needed for the duration variants. For 3-step options it ends the episode at 160 decisions, as in run 4 |
| `VecMonitor` | `rollout/ep_rew_mean` and `rollout/ep_len_mean` in TensorBoard |
| `OptionUsageCallback` ([OptionUsageCallback.py](../HUGIN_gym/callbacks/OptionUsageCallback.py)) | The meta agent's choices per survey phase in TensorBoard |
| `HRL_VARIANT` in `training_config.json` | Records the variant settings |
| 30 h limit in `fox_train_HRL.sh` (was 12 h) | 10M decisions take about 27 h |

### How to compare the variants

- **TensorBoard** (`tensorboard --logdir ../trained-agents`), logged once per PPO update:

  | Tag | Meaning |
  |---|---|
  | `hrl_choice_early/<action>`, `hrl_choice_middle/<action>`, `hrl_choice_late/<action>` | Share of the decisions in the first, middle and last third of the survey that chose the action |
  | `hrl/option_length` | Mean AUV steps per decision |
  | `hrl/episode_steps` | Mean AUV steps of the episodes that ended during the update |
  | `hrl/ended_early` | Share of those episodes that reached the termination goal |
  | `rollout/ep_rew_mean`, `rollout/ep_len_mean` | Mean reward and decisions of the last 100 episodes |

- **Do not compare the reward between variants.** The cost per decision and the mean reward change its scale. Compare `hrl/ended_early`, `hrl/episode_steps`, and the visited and plume coverage from the stats.
- **Episode length:** it counts decisions, and in the duration variants a decision is 2–5 AUV steps, so use `hrl/episode_steps`. For the same reason, the episode length does not show whether a duration variant ended early, so `summarise_training_stats.py` prints "-" for "Ended early" in runs with several option lengths. Use `hrl/ended_early`.
- **The x-axis in TensorBoard counts decisions.** The duration variants make fewer decisions per AUV step, and `hrl/option_length` converts between the two.
- **One run per variant:** a difference smaller than the spread between repeats cannot be trusted. Repeat the promising variants with 2–3 runs.
- **Do not stop a run because its curve is flat.** Run 4 was flat for almost 4M decisions before it improved again.

### Jobs

All five jobs started on 2026-10-29 at 22:58:04 and completed without a resume.

| Variant | Job | Node | Slurm end | Elapsed | Decisions/s | AUV steps | AUV steps/s | CPU use | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| `minimal` | 4371279 | c1-9 | 2026-10-30 22:32 | 23:34:10 | 118 | 29.9M | 353 | 34 % | 7.0 GiB |
| `minimal_values` | 4371280 | c1-8 | 2026-10-30 21:52 | 22:54:31 | 121 | 29.9M | 363 | 36 % | 7.1 GiB |
| `duration` | 4371281 | c1-8 | 2026-10-30 21:23 | 22:25:28 | 124 | 24.0M | 298 | 31 % | 7.1 GiB |
| `duration_mean` | 4371282 | c1-8 | 2026-10-31 08:50 | 1-09:52:03 | 82 | 40.0M | 328 | 36 % | 7.1 GiB |
| `minimal_values_duration` | 4371283 | c1-8 | 2026-10-30 14:29 | 15:31:14 | 179 | 20.7M | 371 | 39 % | 7.0 GiB |

- **Sources:** the Slurm times, the state, the node, the CPU time (`TotalCPU`) and the peak memory (`MaxRSS` of the batch step) come from `sacct`. Each Slurm log ends with "Job <ID> completed at …".
- **How the columns are computed:**
  - **Decisions/s:** from `RUN_INFO.timestamp` to the time the final model was saved. Each run made 10,002,432 decisions.
  - **AUV steps:** `hrl/option_length` times the 6,144 decisions of each PPO update (12 envs × 512), summed over the updates.
  - **CPU use:** `TotalCPU` divided by 16 × the elapsed time.
- **The time follows the AUV steps, not the decisions:** all five ran 300–370 AUV steps per second.
  - `duration_mean` took the longest (33.9 h), because its options ran the most AUV steps (40.0M, about 4 per decision).
  - `minimal_values_duration` was the fastest (15.5 h), because it chose almost only 2-step options (20.7M AUV steps).
  - Before the runs, the duration variants were expected to take 18–45 h. They took 15.5–33.9 h, well within their 54 h limit.
- **The minimal meta state does not make the decisions faster:** `minimal` and `minimal_values` made 118 and 121 decisions per second, within the 103–125 of run 4. This was expected, because most of the time per decision goes to the GP update and the sub-agents.
- **Resources:** 31–39 % of the 16 CPUs and at most 7.1 GiB of the 32 GiB requested, as in run 4.

### Results

Mean ± std over the last 1,000 training episodes. `untuned` is run 4.

| Variant | Episodes | Ended early (%) | AUV steps | Decisions | Visited (%) | True plume coverage (%) | Total reward |
|---|---|---|---|---|---|---|---|
| `untuned` | 76,478 | 86.2 | ≈ 366 | 122.0 ± 22.0 | 77.0 ± 5.5 | 96.0 ± 10.8 | -162.8 ± 163.2 |
| `minimal` | 78,498 | 83.5 | 371 | 123.7 ± 22.1 | 76.3 ± 7.8 | 95.2 ± 9.3 | -190.2 ± 230.1 |
| `minimal_values` | 82,307 | 89.2 | 346 | 115.7 ± 21.8 | 76.5 ± 7.8 | 96.7 ± 6.9 | -141.6 ± 241.3 |
| `duration` | 60,578 | 86.5 | 367 | 124.7 ± 27.1 | 76.7 ± 6.6 | 96.3 ± 10.7 | -160.7 ± 168.8 |
| `duration_mean` | 101,103 | 81.1 | 389 | 93.9 ± 20.3 | 87.5 ± 6.4 | 92.2 ± 9.0 | -40.1 ± 73.8 |
| `minimal_values_duration` | 55,320 | 89.5 | 352 | 168.7 ± 31.8 | 75.9 ± 8.5 | 96.9 ± 8.2 | -136.4 ± 217.6 |

- **Ended early (%)** and **AUV steps** (per episode) come from `hrl/ended_early` and `hrl/episode_steps` in TensorBoard. Each value is the mean over the PPO updates of the last 1,000 episodes (16–28 updates), so it has no std.
  - Run 4 logged neither. Its "Ended early" comes from the stats, which is correct for 3-step options. Its AUV steps are 3 × the decisions. For `minimal`, 3 × the decisions gives 371.1, against 370.7 from TensorBoard.
  - For `minimal` and `minimal_values`, the stats give 83.5 % and 89.1 % ended early, against 83.5 % and 89.2 % from TensorBoard.
- **Decisions, Visited, True plume coverage and Total reward** come from `summarise_training_stats.py`.
- **Total reward** can only be compared between variants with the same reward:
  - `untuned`, `minimal` and `minimal_values`: sum of the step rewards, no cost per decision.
  - `duration` and `minimal_values_duration`: sum, with a cost of 0.1 per decision.
  - `duration_mean`: mean of the step rewards. It cannot be compared with any other variant.

![HRL PPO minimal 4371279](training_runs/HRL_PPO_minimal_4371279/training_stats.png)

![HRL PPO minimal_values 4371280](training_runs/HRL_PPO_minimal_values_4371280/training_stats.png)

![HRL PPO duration 4371281](training_runs/HRL_PPO_duration_4371281/training_stats.png)

![HRL PPO duration_mean 4371282](training_runs/HRL_PPO_duration_mean_4371282/training_stats.png)

![HRL PPO minimal_values_duration 4371283](training_runs/HRL_PPO_minimal_values_duration_4371283/training_stats.png)

- **Plot panels:** as in run 4. "Episode Length" counts decisions, which are 2–5 AUV steps in the duration variants. "Steps Around Threshold (%)" is the true border coverage, and "Steps Above Threshold (%)" the true plume coverage.

#### Learning curves

Each row is the mean over the episodes that ended in that million decisions: row 1M covers decisions 0 to 1M, row 2M covers 1M to 2M, and so on. The values from TensorBoard are the mean over the PPO updates in that million.

- **Rows by decisions:** the duration variants run more AUV steps per decision (see the last table), so at the same row they have trained on more AUV steps.
- **`untuned`:** its column uses the same method, so it differs slightly from the run 4 table, which splits the episodes into 20 groups of equal size.

**Ended early (%)**, from `hrl/ended_early` (`untuned`: from the stats):

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 52.6 | 57.5 | 62.4 | 33.9 | 56.9 | 56.4 |
| 2M | 72.6 | 79.1 | 77.4 | 58.4 | 72.0 | 69.8 |
| 3M | 79.4 | 82.1 | 85.1 | 64.4 | 75.1 | 79.3 |
| 4M | 78.3 | 81.1 | 86.7 | 66.0 | 77.7 | 83.8 |
| 5M | 78.5 | 83.3 | 88.4 | 73.9 | 78.3 | 86.6 |
| 6M | 73.5 | 83.6 | 88.7 | 79.9 | 79.2 | 87.4 |
| 7M | 75.7 | 82.7 | 89.3 | 82.9 | 80.5 | 87.7 |
| 8M | 81.3 | 84.2 | 89.3 | 84.3 | 80.8 | 88.9 |
| 9M | 84.3 | 84.0 | 89.7 | 83.9 | 80.2 | 89.3 |
| 10M | 85.3 | 83.4 | 89.8 | 85.8 | 81.1 | 89.3 |

**AUV steps per episode**, from `hrl/episode_steps` (`untuned`: 3 × the decisions):

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 432 | 426 | 422 | 458 | 425 | 432 |
| 2M | 400 | 388 | 392 | 428 | 400 | 406 |
| 3M | 392 | 381 | 370 | 415 | 398 | 392 |
| 4M | 394 | 377 | 364 | 418 | 395 | 380 |
| 5M | 393 | 376 | 357 | 404 | 394 | 366 |
| 6M | 397 | 376 | 354 | 388 | 393 | 364 |
| 7M | 393 | 377 | 351 | 381 | 392 | 362 |
| 8M | 381 | 375 | 350 | 377 | 392 | 357 |
| 9M | 376 | 374 | 348 | 376 | 392 | 351 |
| 10M | 369 | 374 | 346 | 371 | 391 | 352 |

**Visited (%)**:

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 79.8 | 78.3 | 78.4 | 74.2 | 82.1 | 76.9 |
| 2M | 84.0 | 77.9 | 78.3 | 76.9 | 86.3 | 79.0 |
| 3M | 87.3 | 77.1 | 77.4 | 79.1 | 86.9 | 77.2 |
| 4M | 87.4 | 77.2 | 76.8 | 77.2 | 87.3 | 76.6 |
| 5M | 87.2 | 77.0 | 76.6 | 76.9 | 87.4 | 76.3 |
| 6M | 85.5 | 76.7 | 76.4 | 76.7 | 87.4 | 76.2 |
| 7M | 78.2 | 76.6 | 76.4 | 76.6 | 87.6 | 76.0 |
| 8M | 77.8 | 76.3 | 76.4 | 76.5 | 87.5 | 76.1 |
| 9M | 77.1 | 76.4 | 76.4 | 76.6 | 87.6 | 76.1 |
| 10M | 77.2 | 76.3 | 76.4 | 76.5 | 87.5 | 76.0 |

**True plume coverage (%)**. It is missing for 0.6–2.5 % of the episodes, so its rows line up only roughly with the decisions:

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 88.9 | 89.8 | 91.9 | 88.4 | 89.5 | 91.7 |
| 2M | 91.5 | 93.9 | 93.7 | 91.2 | 91.2 | 92.9 |
| 3M | 92.1 | 95.1 | 95.6 | 90.6 | 91.6 | 94.8 |
| 4M | 92.1 | 94.5 | 96.3 | 92.0 | 91.9 | 95.7 |
| 5M | 92.2 | 95.1 | 96.6 | 93.8 | 92.0 | 96.8 |
| 6M | 91.6 | 95.1 | 96.8 | 95.3 | 92.1 | 96.8 |
| 7M | 93.7 | 94.8 | 97.0 | 96.1 | 92.3 | 96.8 |
| 8M | 95.3 | 95.5 | 97.1 | 96.5 | 92.3 | 97.0 |
| 9M | 96.0 | 95.3 | 97.0 | 96.2 | 92.2 | 97.1 |
| 10M | 96.3 | 95.5 | 96.9 | 96.4 | 92.3 | 97.1 |

**Total reward** (compare only between variants with the same reward, see above):

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | -366.5 | -339.1 | -328.2 | -371.4 | -135.1 | -321.0 |
| 2M | -303.1 | -227.2 | -241.7 | -296.9 | -83.2 | -266.1 |
| 3M | -302.9 | -206.7 | -182.4 | -298.4 | -65.0 | -210.6 |
| 4M | -306.9 | -204.2 | -166.0 | -277.8 | -54.6 | -178.4 |
| 5M | -305.6 | -194.0 | -152.2 | -242.9 | -51.7 | -163.6 |
| 6M | -306.9 | -190.6 | -149.5 | -205.0 | -47.1 | -156.3 |
| 7M | -244.9 | -197.0 | -143.6 | -193.0 | -41.4 | -151.2 |
| 8M | -207.9 | -191.3 | -142.5 | -184.6 | -43.0 | -139.7 |
| 9M | -187.2 | -192.9 | -138.0 | -185.8 | -41.5 | -132.4 |
| 10M | -175.4 | -193.3 | -137.7 | -169.9 | -39.3 | -134.6 |

**Policy entropy**, minus `train/entropy_loss`. A uniform choice gives 1.10 with 3 actions and 1.79 with the 6 actions of the duration variants:

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 0.75 | 0.80 | 0.74 | 1.08 | 1.16 | 0.93 |
| 2M | 0.29 | 0.48 | 0.55 | 0.87 | 0.88 | 0.56 |
| 3M | 0.03 | 0.42 | 0.37 | 0.95 | 0.65 | 0.57 |
| 4M | 0.04 | 0.38 | 0.31 | 0.99 | 0.48 | 0.48 |
| 5M | 0.05 | 0.38 | 0.27 | 0.92 | 0.45 | 0.39 |
| 6M | 0.16 | 0.39 | 0.24 | 0.80 | 0.39 | 0.37 |
| 7M | 0.52 | 0.39 | 0.23 | 0.93 | 0.26 | 0.35 |
| 8M | 0.45 | 0.39 | 0.20 | 0.98 | 0.37 | 0.30 |
| 9M | 0.40 | 0.38 | 0.19 | 1.04 | 0.29 | 0.26 |
| 10M | 0.36 | 0.37 | 0.18 | 1.01 | 0.37 | 0.25 |

**Explained variance** of the value function, from `train/explained_variance`:

| Up to decision | `untuned` | `minimal` | `minimal_values` | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|---|---|---|
| 1M | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| 2M | 0.00 | 0.00 | 0.28 | 0.00 | 0.41 | 0.52 |
| 3M | 0.27 | 0.00 | 0.55 | 0.29 | 0.54 | 0.74 |
| 4M | 0.80 | 0.00 | 0.58 | 0.84 | 0.57 | 0.74 |
| 5M | 0.80 | 0.00 | 0.58 | 0.82 | 0.59 | 0.71 |
| 6M | 0.75 | 0.00 | 0.59 | 0.78 | 0.63 | 0.72 |
| 7M | 0.66 | 0.00 | 0.59 | 0.73 | 0.65 | 0.73 |
| 8M | 0.64 | 0.00 | 0.62 | 0.71 | 0.65 | 0.73 |
| 9M | 0.63 | 0.00 | 0.63 | 0.69 | 0.66 | 0.75 |
| 10M | 0.63 | 0.00 | 0.64 | 0.68 | 0.69 | 0.75 |

**Option length** (mean AUV steps per decision), from `hrl/option_length`. It is 3.0 for the other variants:

| Up to decision | `duration` | `duration_mean` | `minimal_values_duration` |
|---|---|---|---|
| 1M | 2.09 | 2.79 | 2.15 |
| 2M | 2.05 | 3.54 | 2.03 |
| 3M | 2.23 | 4.08 | 2.04 |
| 4M | 2.16 | 4.25 | 2.05 |
| 5M | 2.19 | 4.21 | 2.06 |
| 6M | 2.21 | 4.21 | 2.07 |
| 7M | 2.50 | 4.33 | 2.07 |
| 8M | 2.69 | 4.18 | 2.07 |
| 9M | 2.95 | 4.27 | 2.08 |
| 10M | 2.94 | 4.13 | 2.08 |

#### Choices of the meta agent

Share of the decisions in each third of the survey that chose each action, from `hrl_choice_<phase>/<action>`. The values are means over the PPO updates of the last 1,000 episodes. Run 4 did not log the choices.

3-step variants, as space / plume / border:

| Variant | First third | Middle third | Last third |
|---|---|---|---|
| `minimal` | 0.55 / 0.16 / 0.30 | 0.90 / 0.03 / 0.06 | 0.94 / 0.02 / 0.05 |
| `minimal_values` | 0.68 / 0.15 / 0.17 | 0.97 / 0.02 / 0.01 | 1.00 / 0.00 / 0.00 |

Duration variants:

| Variant | Third | `space_2` | `space_5` | `plume_2` | `plume_5` | `border_2` | `border_5` |
|---|---|---|---|---|---|---|---|
| `duration` | First | 0.27 | 0.23 | 0.18 | 0.04 | 0.23 | 0.05 |
| | Middle | 0.51 | 0.36 | 0.05 | 0.01 | 0.06 | 0.01 |
| | Last | 0.58 | 0.36 | 0.02 | 0.00 | 0.03 | 0.01 |
| `duration_mean` | First | 0.18 | 0.82 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Middle | 0.21 | 0.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Last | 0.37 | 0.63 | 0.00 | 0.00 | 0.00 | 0.00 |
| `minimal_values_duration` | First | 0.68 | 0.00 | 0.05 | 0.06 | 0.20 | 0.01 |
| | Middle | 0.96 | 0.00 | 0.01 | 0.01 | 0.02 | 0.00 |
| | Last | 0.98 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |

#### The value function of `minimal` did not learn

- **In training:** the explained variance of `minimal` stayed between −0.004 and 0.001 for all 10M decisions, and its value loss did not go down (2,814 at the first update, 2,753 at the last).
- **In the saved model:**
  - The critic gives the same value, −122.3, for every meta state. Over 2,000 random meta states its std is 0.001, while the action probabilities change with the state.
  - All 64 units of the critic's second hidden layer are saturated (|tanh| > 0.999) in every one of these states, so no gradient passes through them.
  - In `minimal_values`, which has the same network, no unit is saturated.
- **Effect:** PPO trained the `minimal` policy with a constant baseline instead of a learned value. So `minimal` shows what the 6 numbers reach without a working critic, not what they can reach.
- **Cause:** the runs do not show why this happened in `minimal` and not in `minimal_values`. A repeat of `minimal` would show whether it happens again.

#### Observations

All results are from one run per variant and from training episodes with sampled actions. A difference of a few points can be within the spread between repeats. The comparisons follow the table in [The variants](#the-variants).

- **`minimal` against `untuned` (the small meta state):**
  - **At the end:** `minimal` came close to `untuned` without reaching it: 83.5 % against 86.2 % ended early, about the same number of AUV steps (371 and about 366) and plume coverage (95.2 % and 96.0 %).
  - **Speed of learning:** `minimal` learned faster at first (79.1 % against 72.6 % ended early at 2M). It was flat at 81–84 % from 3M on. `untuned` passed it only after 8M, in its second improvement.
  - **Caveat:** its critic did not learn (see above), so this comparison is not fair to the minimal state.
- **`minimal_values` against `minimal` (the sub-agent values):**
  - **At the end:** `minimal_values` was clearly better: 89.2 % against 83.5 % ended early, 346 against 371 AUV steps, and 96.7 % against 95.2 % plume coverage. It is also better than `untuned` (86.2 %, about 366 AUV steps) and the best variant in training, together with `minimal_values_duration`.
  - **Caveat:** its critic learned (explained variance 0.64 at the end) and the one of `minimal` did not. The runs cannot separate the effect of the sub-agent values from the effect of a working critic.
  - **Policy:** it ends with the lowest entropy (0.18). In the last third of the survey it always chooses space exploration.
- **`duration` against `untuned` (choosing the option length):**
  - **At the end:** the same result as `untuned`: 86.5 % against 86.2 % ended early, 367 against about 366 AUV steps, 96.3 % against 96.0 % plume coverage.
  - **Speed of learning:** `duration` learned more slowly at first (58.4 % against 72.6 % ended early at 2M), and caught up at about 6M.
  - **Long options:** in the first 1M decisions it chose a 5-step option in 4 % of the decisions. At the end it chose one in 32 % of the decisions in the first third of the survey, 38 % in the middle and 37 % in the last third, almost all `space_5`. The option length went up from 2.2 AUV steps at 6M to 2.9 at the end.
  - **Question 1:** the meta agent does learn to run space exploration for 5 steps, but more in the middle and the end of the survey (0.36 of the decisions) than at the start (0.23). In the first third, half of its decisions go to the plume and border sub-agents. The full meta state has no time input (see [Minimal meta states](#minimal-meta-states-minimal-minimal_values-minimal_values_duration)), so the agent can only tell the survey phase from the coverages.
  - **Policy:** it is still fairly random at the end: entropy 1.01, the highest of all variants (the maximum is 1.79). Its deterministic actions do much worse than its sampled ones: 33 % against 86 % ended early (see [Evaluation of the trained agents](#evaluation-of-the-trained-agents)).
- **`duration_mean` against `duration` (the mean reward):**
  - **Long options, as expected:** it chose the 5-step options much more (4.1 against 2.9 AUV steps per decision).
  - **Only the space agent, which was not expected:** at the end it used only the space exploration sub-agent. Plume and border got 0.00 of the decisions in all three thirds, against 0.24 in the first 1M decisions.
  - **Result:** it visited the most of the map (87.5 %), but had the lowest true plume coverage (92.3 %) and the lowest share that ended early (81.1 %). Its episodes were the longest (389 AUV steps).
  - **Cause:** the runs do not show why it stopped using the plume and border agents.
- **`minimal_values_duration` against `minimal_values` and `duration` (both ideas together):**
  - **Hardly any long options:** at the end it chose a 5-step option in 7 % of the decisions in the first third (mostly `plume_5`) and in 1 % or less later. The option length stayed at 2.1 AUV steps. In effect it is `minimal_values` with 2-step options.
  - **Result:** the same as `minimal_values`: 89.5 % against 89.2 % ended early, 352 against 346 AUV steps, 97.1 % against 96.7 % plume coverage. It is better than `duration` (86.5 %, 367 AUV steps).
  - **Value function:** the highest explained variance of all variants (0.75 at the end).
  - **Cause:** the runs do not show why, with the minimal state, the meta agent did not use the long options that `duration` uses.
- **Summary for the two questions:**
  1. With the sum reward and the full meta state, the meta agent learns to run space exploration for 5 steps in about a third of its decisions, mostly after the start of the survey. It does no better than `untuned`. With the minimal state it hardly uses the long options. With the mean reward it uses long space options almost only, and does worse.
  2. Six numbers and the three sub-agent values were enough to do better than the full meta state of 158 inputs (89.2 % against 86.2 % ended early). Six numbers alone reached 83.5 %, but with a critic that did not learn.
- **Before ranking the variants:** repeats of the promising variants are needed. The evaluation with sampled and deterministic actions is below.

### Evaluation of the trained agents

On 2026-10-31 the five final models were tested on 1,000 new episodes each with [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>), with sampled (mode 4) and deterministic (mode 1) actions, as in run 4 (see [Evaluation of the trained agent](#evaluation-of-the-trained-agent)).

- **Test env:** the same as in run 4. The script reads `HRL_VARIANT` from each run's `training_config.json` (`hrl_variant_from_config` in [io.py](../HUGIN_gym/utils/io.py)), so each model is tested with its own actions, meta state and reward, and with the 480-step survey budget.
- **Same episodes:** seed 0, as in run 4, so all models and modes get the same 1,000 episodes.
- **Code:** `822156c`, plus the `--model` and `--mode` arguments of the script, which are not yet committed. `822156c` disables the turn penalty in the border sub-agent's reward. This does not change the results: the sub-agents are already trained, and the meta agent's reward only takes the goal bonus (+40) and the penalty for leaving the map (−4) from the base env ([HRL_env.py](../HUGIN_gym/envs/HRL_env.py)).
- **Run with** the following command from the repository root, once per run folder and mode (on Windows: `.venv/Scripts/python`):

  ```bash
  PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --model ../trained-agents/<run>/HRL_PPO_scratch.zip --mode 4
  ```

**Ended early (%)** (`SUCCESS_RATE`), next to the last 1,000 training episodes and the policy entropy in the last 1M training decisions (see [Learning curves](#learning-curves)):

| Variant | Training | Sampled (mode 4) | Deterministic (mode 1) | Sampled − deterministic | Entropy at the end of training |
|---|---|---|---|---|---|
| `untuned` | 86.2 | 86 | 51 | 35 | 0.36 |
| `minimal` | 83.5 | 85 | 59 | 26 | 0.37 |
| `minimal_values` | 89.2 | 89 | 86 | 3 | 0.18 |
| `duration` | 86.5 | 86 | 33 | 53 | 1.01 |
| `duration_mean` | 81.1 | 82 | 83 | -1 | 0.37 |
| `minimal_values_duration` | 89.5 | 91 | 77 | 14 | 0.25 |

**Sampled actions (mode 4)**, mean ± std over the 1,000 test episodes:

| Variant | Ended early (%) | AUV steps | Decisions | Turns | Visited (%) | True plume coverage (%) | True border coverage (%) | Total reward |
|---|---|---|---|---|---|---|---|---|
| `untuned` | 86 | 376.7 ± 61.6 | 125.9 ± 20.5 | 113.8 ± 39.8 | 76 ± 6 | 97.1 ± 9.4 | 95.7 ± 8.7 | -173.9 ± 158.1 |
| `minimal` | 85 | 365.2 ± 65.6 | 122.0 ± 21.8 | 106.7 ± 38.1 | 76 ± 7 | 94.7 ± 11.2 | 94.9 ± 10.2 | -173.9 ± 213.2 |
| `minimal_values` | 89 | 344.8 ± 65.1 | 115.2 ± 21.6 | 87.4 ± 40.3 | 76 ± 8 | 96.9 ± 7.3 | 96.1 ± 6.8 | -145.1 ± 261.6 |
| `duration` | 86 | 368.4 ± 64.6 | 125.2 ± 26.6 | 106.6 ± 38.8 | 77 ± 6 | 96.6 ± 9.3 | 95.3 ± 8.1 | -166.0 ± 169.9 |
| `duration_mean` | 82 | 392.3 ± 69.5 | 93.8 ± 20.3 | 68.7 ± 19.6 | 88 ± 6 | 92.6 ± 8.7 | 91.7 ± 7.6 | -39.3 ± 65.6 |
| `minimal_values_duration` | 91 | 344.7 ± 62.3 | 165.1 ± 31.0 | 89.4 ± 34.0 | 76 ± 7 | 97.2 ± 6.3 | 96.1 ± 6.2 | -122.0 ± 188.1 |

**Deterministic actions (mode 1)**, mean ± std over the 1,000 test episodes:

| Variant | Ended early (%) | AUV steps | Decisions | Turns | Visited (%) | True plume coverage (%) | True border coverage (%) | Total reward |
|---|---|---|---|---|---|---|---|---|
| `untuned` | 51 | 403.6 ± 83.5 | 134.7 ± 27.7 | 194.1 ± 140.6 | 55 ± 27 | 95.5 ± 12.8 | 93.6 ± 12.0 | -396.6 ± 418.2 |
| `minimal` | 59 | 411.6 ± 73.4 | 137.4 ± 24.3 | 133.4 ± 77.0 | 77 ± 13 | 88.1 ± 13.7 | 96.5 ± 11.6 | -347.6 ± 290.2 |
| `minimal_values` | 86 | 340.6 ± 73.7 | 113.8 ± 24.5 | 95.3 ± 80.8 | 73 ± 13 | 96.5 ± 9.5 | 95.7 ± 8.7 | -152.9 ± 323.5 |
| `duration` | 33 | 428.4 ± 79.0 | 205.1 ± 43.0 | 230.9 ± 140.0 | 44 ± 27 | 93.3 ± 18.0 | 89.2 ± 16.1 | -375.9 ± 359.9 |
| `duration_mean` | 83 | 391.7 ± 69.2 | 90.8 ± 20.0 | 68.4 ± 19.6 | 88 ± 6 | 92.9 ± 8.3 | 91.9 ± 7.4 | -31.7 ± 59.8 |
| `minimal_values_duration` | 77 | 361.7 ± 81.7 | 178.7 ± 40.8 | 109.7 ± 82.9 | 75 ± 12 | 91.7 ± 13.4 | 96.6 ± 9.5 | -163.4 ± 226.8 |

**The GP map at the end of the episode**, as sampled / deterministic:

| Variant | Plume coverage of the GP map (%) | Border coverage of the GP map (%) | RMSE of the GP concentration map | True plume cells found by the GP map (%) |
|---|---|---|---|---|
| `untuned` | 98 / 95 | 96 / 93 | 0.0023 / 0.0050 | 99.6 / 99.3 |
| `minimal` | 95 / 87 | 95 / 95 | 0.0029 / 0.0048 | 99.5 / 99.4 |
| `minimal_values` | 97 / 97 | 96 / 95 | 0.0021 / 0.0036 | 99.9 / 99.8 |
| `duration` | 97 / 93 | 95 / 88 | 0.0025 / 0.0072 | 99.9 / 98.4 |
| `duration_mean` | 93 / 93 | 92 / 93 | 0.0015 / 0.0014 | 99.8 / 99.9 |
| `minimal_values_duration` | 97 / 92 | 96 / 96 | 0.0021 / 0.0039 | 99.9 / 99.6 |

**Choices** (`CHOICES`), the share of all decisions in the 1,000 episodes. 3-step variants, as space / plume / border:

| Variant | Sampled (mode 4) | Deterministic (mode 1) |
|---|---|---|
| `untuned` | 0.74 / 0.12 / 0.14 | 0.48 / 0.34 / 0.18 |
| `minimal` | 0.75 / 0.09 / 0.16 | 0.71 / 0.01 / 0.29 |
| `minimal_values` | 0.83 / 0.08 / 0.09 | 0.79 / 0.11 / 0.10 |

Duration variants:

| Variant | Actions | `space_2` | `space_5` | `plume_2` | `plume_5` | `border_2` | `border_5` |
|---|---|---|---|---|---|---|---|
| `duration` | Sampled | 0.41 | 0.31 | 0.11 | 0.02 | 0.12 | 0.03 |
| | Deterministic | 0.27 | 0.03 | 0.53 | 0.00 | 0.16 | 0.00 |
| `duration_mean` | Sampled | 0.22 | 0.78 | 0.00 | 0.00 | 0.00 | 0.00 |
| | Deterministic | 0.18 | 0.82 | 0.00 | 0.00 | 0.00 | 0.00 |
| `minimal_values_duration` | Sampled | 0.85 | 0.00 | 0.03 | 0.03 | 0.09 | 0.01 |
| | Deterministic | 0.77 | 0.00 | 0.00 | 0.01 | 0.21 | 0.00 |

- **`untuned`:** the numbers come from the run 4 evaluation.
- **Output lines:** as in the run 4 table. "Ended early" is `SUCCESS_RATE`, "AUV steps" is `STEPS`, "Decisions" is `Options_CHOSEN` and "Visited" is `SPACE`. The plume and border coverages of the GP map are `PLUME` and `BORDER`, and the true coverages are `ACTUAL PLUME` and `ACTUAL BORDER`.
- **True plume cells found by the GP map:** `TP` / (`TP` + `FN`).
- **Precision of "Ended early":** the script prints it as a share with two decimals, so it is rounded to 1 point. Its standard error over 1,000 episodes (std / √1,000) is 0.9–1.5 points.
- **Total reward:** compare it only between variants with the same reward, as listed under the run 5 results above.

**Observations:**

- **Sampled actions reproduce training for all five variants:**
  - **Ended early:** within 1.5 points of training.
  - **Episodes:** within 8 AUV steps and 4 decisions.
  - **Coverage:** within 1 point for the visited share, and within 0.5 points for the true plume coverage.
  - **Option length** of the duration variants (AUV steps / decisions): 2.94, 4.18 and 2.09 AUV steps per decision, against 2.94, 4.14 and 2.09 from `hrl/option_length` over the last 1,000 training episodes.
- **Deterministic actions:** the gap to the sampled results is small where the policy was almost deterministic at the end of training, and large where it was still random:
  - `minimal_values` (entropy 0.18): 3 points;
  - `minimal_values_duration` (0.25): 14 points;
  - `minimal` (0.37): 26 points;
  - `untuned` (0.36): 35 points;
  - `duration` (1.01): 53 points.

  `duration_mean` (0.37) is the exception, with 83 % against 82 %. In both modes it uses only the space exploration sub-agent, so its randomness is only in the choice between 2 and 5 steps.
- **`duration` with deterministic actions is the worst result of all:**
  - **Ended early:** 33 %.
  - **Coverage:** 44 ± 27 % of the map visited.
  - **Turns:** 231, against 107 with sampled actions.
  - **Choices:** `plume_2` in 53 % of the decisions (sampled: 11 %), and a 5-step option in only 3 % (sampled: 36 %).

  This fits the explanation in run 4: in the same situation the deterministic agent always makes the same choice and can repeat the same moves until the budget runs out.
- **`minimal` and `minimal_values_duration` with deterministic actions hardly use the plume sub-agent:**
  - **Plume options:** 1 % of the decisions, against 9 % and 6 % with sampled actions.
  - **True plume coverage:** 88.1 % and 91.7 %, against 94.7 % and 97.2 % with sampled actions.
  - **Visited share:** almost unchanged (77 % and 75 %).

  So these episodes mostly fail at the plume coverage, not at the map coverage.
- **`minimal_values` with deterministic actions is the best deterministic result:**
  - **Ended early:** 86 %, the same as `untuned` and `duration` with sampled actions.
  - **Episodes:** the fewest AUV steps (341).
  - **Plume options:** 11 % of the decisions.
- **`duration_mean`:** the same in both modes.
  - **Ended early:** 82–83 %.
  - **Coverage:** the most of the map visited (88 %) and the lowest RMSE (0.0014–0.0015), but the lowest true plume coverage with sampled actions (92.6 %).
  - **Turns:** the fewest (68).
- **The GP map finds the plume in all variants and modes:** at least 98.4 % of the true plume cells. `FP` is 0.0024–0.0026, the 4 corner cells, as in run 4.
- **For the two questions:**
  1. **Choosing the option length:** the long options of `duration` come from sampling. With sampled actions, 36 % of its decisions are 5-step options. With deterministic actions, only 3 % are, so in the situations the deterministic agent reaches, a 5-step option is rarely the most likely choice. `duration_mean` chooses `space_5` in about 80 % of its decisions in both modes.
  2. **The minimal meta state:** `minimal_values` is the only variant whose deterministic actions do about as well as its sampled ones (86 % against 89 %). With sampled actions, `minimal_values_duration` (91 %) and `minimal_values` (89 %) are the best. The 2-point difference between them is too small to separate them with 1,000 test episodes and one run per variant.
- **Which to report:** as in run 4, report the sampled results with the deterministic results next to them, and say which is which.

### References

- Harb, J., Bacon, P.-L., Klissarov, M. and Precup, D. (2018). When waiting is not an option: learning options with a deliberation cost. *AAAI Conference on Artificial Intelligence*.
- Lakshminarayanan, A. S., Sharma, S. and Ravindran, B. (2017). Dynamic action repetition for deep reinforcement learning. *AAAI Conference on Artificial Intelligence*.
- Sharma, S., Lakshminarayanan, A. S. and Ravindran, B. (2017). Learning to repeat: fine grained action repetition for deep reinforcement learning. *International Conference on Learning Representations (ICLR)*.
- Sutton, R. S., Precup, D. and Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1–2), 181–211.

## Adding a new run

1. Before submitting: commit the changes and note the commit. Submit with `sbatch fox_train.sh <tag>` (HRL: `sbatch fox_train_HRL.sh <tag>`) and note the job ID(s).
2. Add a row to the overview table and a section with the goal, what changed (with commits and why), the settings tables and the Fox resources.
3. If a job fails: find the cause in `slurm-hugin-plume-<job ID>_<array index>.out` on Fox (HRL: `slurm-hugin-hrl-<job ID>.out`). Resume the runs from their newest checkpoint with `sbatch fox_train.sh <tag> <job ID>` (add `--array=...` to resume only some algorithms) and note the resume job IDs. Get the start, end and elapsed times with `sacct -j <job IDs> --format=JobID,Start,End,Elapsed,State`.
4. After downloading: make the plot with [training_data_visualiser.py](../HUGIN_gym/evaluation/training_data_visualiser.py). Copy `training_stats.png`, `training_config.json` and, for a resumed run, `resume_log.txt` to `docs/training_runs/<run>/`. Run [summarise_training_stats.py](../HUGIN_gym/evaluation/summarise_training_stats.py) for the numbers table.
5. Write down what the curves and the numbers show.
