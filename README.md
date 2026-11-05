# HUGIN HRL Gymnasium Environment for gas seepage detection (HRL-GSD)

A Gymnasium-based environment for simulating and training reinforcement learning agents on the HUGIN AUV. This environment provides a simplified simulation of the HUGIN.

This repository holds the code and results of the paper *Multi-objective robotic environmental monitoring with hierarchical reinforcement learning* by Martin Haugen, Erik Johnsrud, and Øyvind L. Rekdal (University of Oslo). It is a fork of the [repository of Matthias Schmitt's Master's thesis](https://github.com/MatthiasHansHeinrichSchmitt/HRL_DRL_gas_seep_detection), which the paper builds on.

![Simulation visualization](Sim_visualiser.png)

*The Meshcat view from the thesis, with a 3D domain. The paper uses a 2D map.*

## 🌊 Features

- **Concentration Distribution**: Visualizing a concentration distribution in the form of opacity-dependent bubbles
- **3D Visualization**: Real-time rendering using Meshcat. The simulation in the paper is 2D, shown in the same 3D view.
- **Custom Rewards**: Configurable reward functions for different tasks
- **Feature Extraction**: Individual feature extraction for the different sub-agents (detection, characterization, and delineation)
- **Stable-Baselines3 Compatible**: Ready to use with popular RL frameworks (e.g. PPO, DQN, SAC) + logging
- **Survey Evaluation**: A scripted sweep and handover, lawnmower, rule, and random baselines, and map metrics

## 🛠️ Installation

### Prerequisites
- Python ≥3.12
- pip

### Using pip
```bash
# Clone the repository
git clone https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection.git
cd HRL_DRL_gas_seep_detection

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

Use pip rather than `uv run`. `uv.lock` pins torch 2.5.1, which has no wheel for Python 3.14.

### Running the tests
```bash
pip install pytest
python -m pytest tests -q
```

Pass `tests` explicitly. A bare `pytest` also collects the scripts `HUGIN_gym/evaluation/test_*.py`, which are not unit tests.

## 📊 Examples

The `evaluation` and `training` directories contain several scripts demonstrating different uses:

- `test_XYZ.py` (e.g. `test_PPO.py`): Manual control of the environment, a lawnmower pattern, or evaluation of a trained model
- `train_XYZ.py` (e.g. `train_PPO.py`): Training script using PPO, DQN, or SAC

### Running Examples
```bash
# Test the environment. Enter 2 at the prompt for manual control, or 3 for a lawnmower.
python -m HUGIN_gym.evaluation.test_PPO

# Test the training script with a short run
python -m HUGIN_gym.training.train_PPO --smoke

# Train the characterization expert with the preset of the paper (100M steps)
python -m HUGIN_gym.training.train_PPO --compare
```

Mode 1 of `test_PPO` evaluates a trained model, so set `model_path_global` in the script to your model first. Training output goes to `../trained-agents/<run name>`, next to the repository.

## 🖼️ Visualization

The environment uses Meshcat for 3D visualization. When running with `render_mode="human"`, a web browser window will open automatically showing the simulation. The visualization includes:
- AUV model (pose aligned)
- trajectory trackers (green pointers)
- gas plume (grey bubbles: concentration-dependent opacity and radius)
- visited cell marking (green cubes with low opacity)

## 📚 Project Structure
```
HRL_DRL_gas_seep_detection/
├── HUGIN_gym/                  The (H)RL Python package
│   ├── envs/                       Environments (base & HRL env)
│   │   ├── core/                       Dynamics, rewards, GP update, gas maps, visualization
│   │   └── wrappers/                   Env wrappers (obs, GP)
│   ├── agents/                     RL agents & HRL components, including the scripted options
│   │   ├── feature_extractor/          Feature extractors
│   │   └── term_fn/                    Termination conditions (by default off)
│   ├── training/                   Training scripts (DQN, PPO, SAC, HRL)
│   ├── evaluation/                 Evaluation, baselines, tables, figures, and debugging scripts
│   ├── callbacks/                  Logging training data
│   ├── utils/                      Saving/Loading training config
│   └── assets/                     AUV model (.dae)
├── docs/                       Training documentation, run configs, and evaluation results
├── tests/                      Unit tests
├── examples/                   GP and search algorithm examples from the thesis
├── fox_train.sh                Fox (Slurm) job: characterization expert with DQN, PPO, and SAC
├── fox_train_HRL.sh            Fox job: one meta agent run
├── fox_train_HRL_matrix.sh     Submits all meta agent runs of run 6
├── fox_eval_HRL.sh             Fox job: the evaluations of run 6
├── plot_stats.sh               Fox job: plots the training statistics of a run
├── job.sh                      Slurm job from the thesis, not used in the paper
├── pyproject.toml, setup.py    Build config (SB3, gym, meshcat, scikit, ... )
├── LICENSE
└── README.md
```

The central environment is HUGIN_gym/envs/HUGIN_env.py. It models:
- A 2D map of 41 x 41 cells. A `z` limit of 0 in `domain_limit` makes the map 2D, and the paper does not use the 3D option left in the code.
- AUV position and orientation
- Five discrete movement actions: forward (two cells), left and right turn, and up and down. On the 2D map, up and down always hit the boundary, which costs -4.
- One or more Gaussian gas sources: 1–3 per survey for the meta agent, each with its own width σ in [2, 5] cells. A cell belongs to a plume at a concentration of at least 0.3, and overlapping sources count as one plume.
- Visited, above-threshold, and plume-border maps
- Task-specific observations, rewards, and termination conditions

Three low-level tasks are supported:

| Expert in the paper | `AGENT_TYPE` in the code | Task |
|---|---|---|
| Detection | `SPACE` | Maximize general map coverage |
| Characterization | `PLUME` | Map regions above the concentration threshold |
| Delineation | `BORDER` | Trace the plume's threshold boundary |

Reward implementations are under HUGIN_gym/envs/core/rewards/.

### Hierarchical RL

HUGIN_gym/envs/HRL_env.py provides a meta-environment. In the surveys of the paper, a meta agent (the supervisor) makes one choice every 3 AUV steps, for at most 480 AUV steps (160 decisions):
1. Search: a scripted lawnmower sweep (`SweepOption`). In the `space_est` variant, the trained detection expert searches instead.
2. Characterize the plume: the trained expert with the handover
3. Delineate its border: the trained expert with the handover

The handover (`HandoverOption`) is scripted. The characterization and delineation experts only see a 7 x 7 patch around the AUV, so while no target is in that patch, a shortest path brings the AUV to the nearest unfinished cell. Then the trained expert decides. The options are defined in HUGIN_gym/agents/options.py.

The meta agent and the experts are learned. The sweep, the handover, and the rule baseline are scripted. In run 6, the meta agent is trained with PPO for 5M decisions, on 1–3 sources with σ in [2, 5], starting in a random corner. It is not told the number of plumes. Its variants are in `PRESETS` in [train_HRL(ppo).py](<HUGIN_gym/training/train_HRL(ppo).py>).

**Meta reward.** The meta reward is not one of the classes in HUGIN_gym/envs/core/rewards/. The time cost and the pay for new cells are in `HierarchicalEnv.step` and `HierarchicalEnv._progress_reward` in [HRL_env.py](HUGIN_gym/envs/HRL_env.py). The goal and patch bonuses are in `GPWrapper._compute_meta_subgoal_reward` in [GPWrapper.py](HUGIN_gym/envs/wrappers/GPWrapper.py). `HRL_VARIANT.reward_mode` in each run's `training_config.json` names the reward of that run, and `REWARD_FUNCTION_SOURCE` is empty for these runs. The changes document explains [the reward](docs/261103_hrl-vs-lawnmower-changes_draft-v7.md#change-9-a-meta-reward-that-follows-the-goal) and [the bonus](docs/261103_hrl-vs-lawnmower-changes_draft-v7.md#change-10-a-bonus-per-plume).

### End Goals

A survey ends at its goal, or when the 480 AUV steps are spent. The goals are checked in [GPWrapper.py](HUGIN_gym/envs/wrappers/GPWrapper.py).

- **Estimate (the main goal):** the mean GP standard deviation over the map is at most 0.26, at least 90% of the estimated plume and border cells have a GP standard deviation of at most 0.1, visited or not, and no connected area of more than 10 cells has a GP standard deviation above 0.8.
- **Visit:** the mean GP standard deviation over the map is at most 0.1, and the AUV has visited at least 90% of the estimated plume and border cells.

### Baselines

- **Lawnmower:** always chooses the sweep.
- **Rule:** `RuleChooser` in [baselines.py](HUGIN_gym/evaluation/baselines.py) sweeps unless a found plume is unfinished. It then picks the gas expert with the lower share done.
- **Random:** chooses each option with equal probability.
- **Scripted gas:** the rule with `GasFillOption`, a shortest path to the nearest unfinished gas cell in place of the two gas experts. It shows how good gas mapping can get without learning.

### Mapping

HUGIN_gym/envs/wrappers/GPWrapper.py adds belief-based mapping:
- Collects position and concentration measurements
- Updates an exact Gaussian process with an RBF kernel (fixed length scale 3.5) one measurement at a time, in [incremental_gp.py](HUGIN_gym/envs/core/incremental_gp.py)
- Predicts concentrations throughout the domain
- Builds estimated plume and border maps
- Calculates coverage, RMSE, and classification metrics

The incremental update gives the same result as refitting a scikit-learn GP after every step, and it trained about 6 times faster on Fox ([numbers](docs/261103_hrl-vs-lawnmower-changes_draft-v7.md#change-2-a-faster-gp-with-the-same-result)). scikit-learn is still used for other kernels and past 2,000 measurements.

### Training

Training scripts are in HUGIN_gym/training/:
- train_PPO.py, with the presets `--smoke`, `--paper`, `--full`, and `--compare`
- train_DDQN.py, which actually uses Stable-Baselines3's plain DQN, not Double DQN
- train_SAC.py
- train_HRL(ppo).py, the meta agent, with `--preset`, `--seed`, `--max-steps`, and `--smoke`. Quote the module name: `python -m "HUGIN_gym.training.train_HRL(ppo)"`.
- train_HRL(ddqn).py, from the thesis and not used in the paper

They use custom feature extractors from HUGIN_gym/agents/feature_extractor/, vectorized environments, checkpoints, and episode-statistics callbacks.

On `main`, the expert scripts are set to `AGENT_TYPE = "PLUME"`. The detection and delineation experts were trained on separate branches, and their code was not copied into `main`. It is in the tags [`oyvind-final`](https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection/tree/oyvind-final) and [`erik-final`](https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection/tree/erik-final).

### Evaluation And Visualization

HUGIN_gym/evaluation/ contains:
- Survey evaluations and baselines: `run_evaluations.py`, `test_HRL(ppo).py`, `baselines.py`
- Map and plume metrics: `survey_metrics.py`
- The tables and figures of the paper: `summarise_evaluations.py`, `plot_evaluations.py`, `summarise_training_stats.py`
- What the gas experts see and do inside a survey: `log_gas_agents.py`
- The true map against the estimated map: `compare_maps.py`
- Trained-policy rollouts
- Manual control
- Vectorized validation
- Lawnmower-pattern baselines
- Training-curve and confusion-matrix analysis

Meshcat visualization under HUGIN_gym/envs/core/visualisation/ renders the AUV, trajectory, domain, and concentration field. Matplotlib utilities visualize GP predictions and map coverage.

## 📈 Results

The numbers use 1–3 sources, σ in [2, 5], the estimate goal, and sampled actions for the meta agent, as a mean over 3 training seeds. Each cell gives the first/second set of 1,000 episodes. The AUV steps are the measure of distance.

| Survey | Ended by goal (%) | AUV steps | Plume overlap | Border overlap | Turns |
|---|---|---|---|---|---|
| Lawnmower, sweep 3 | 100.0/99.6 | 281.6/283.7 | 0.994/0.993 | 0.983/0.983 | 25.2/25.9 |
| Rule + sweep 6 + handover | 99.9/100.0 | 230.7/229.4 | 0.997/0.998 | 0.990/0.989 | 61.1/60.5 |
| Main meta-agent | 100.0/99.7 | 225.3/226.1 | 0.998/0.997 | 0.991/0.989 | 58.1/59.2 |

- **Against the lawnmower:** the main meta-agent uses about 20% fewer steps with a better map, but more than twice the turns.
- **Against the rule:** it matches the rule rather than clearly beating it. Most of the saving comes from the survey design, because the rule alone already uses 18–19% fewer steps than the lawnmower.
- **The full test:** it passes against the lawnmower on 3 of 3 training seeds on the first set and on 1 of 3 on the second. Against the rule, it passes on 3 of 3 and on 0 of 3.
- **The second set:** we added it after seeing the results of the first, because the margin over the rule was small and the limits of the survey were chosen while looking at the first set.
- **Plume size:** the lead holds only near the plume sizes the meta agent trained on.

Where to find everything:
- [Tables of the first set](docs/evaluation_results/run6/summary.md) and [of the second set](docs/evaluation_results/run6_seed1/summary.md)
- [Figures of the paper](docs/evaluation_results/run6/)
- [Training documentation](docs/training_documentation.md), with the setup and results of every run
- [Changes document](docs/261103_hrl-vs-lawnmower-changes_draft-v7.md), which explains why each part of the survey exists

### Names in the paper and in the code

| Paper | Code |
|---|---|
| Main meta-agent | `sweep6_est` |
| Control | `sweep6_est_control` |
| Map gain | `sweep6_est_map` |
| No time input | `sweep6_est_no_time` |
| One-source training | `sweep6_est_one` |
| Learned search | `space_est` |
| Meta-agent + sweep 4 (visit goal) | `sweep4_visit` |
| First meta-agent | `untuned` (run 4) |
| Best earlier meta-agent | `minimal_values` (run 5) |
| Lawnmower, rule, random | `lawnmower_*`, `rule_*`, `random_*` |
| First set, second set | Evaluation seed 0 (`run6`), evaluation seed 1 (`run6_seed1`) |

In the per-seed tables of `summary.md` and in `evaluations.csv`, "seed" means the training seed. Only the folder shows the evaluation seed.

## 🔁 Reproducing the Results

### What is recorded for each training run

| Runs | Configuration | Reward | Commit | Job script |
|---|---|---|---|---|
| Characterization expert, DQN, PPO, and SAC (runs 1–3) | `training_config.json` in [docs/training_runs](docs/training_runs/) | `REWARD_FUNCTION_SOURCE` in the config, [agent2_plume.py](HUGIN_gym/envs/core/rewards/agent2_plume.py) | [Overview table](docs/training_documentation.md#overview) | [fox_train.sh](fox_train.sh) |
| Meta agent (runs 4–6) | `training_config.json` in [docs/training_runs](docs/training_runs/) | [Meta reward](#hierarchical-rl) | [Overview table](docs/training_documentation.md#overview) | [fox_train_HRL.sh](fox_train_HRL.sh), and [fox_train_HRL_matrix.sh](fox_train_HRL_matrix.sh) for run 6 |
| Detection expert, DQN, PPO, and SAC | Open, see below | [agent1_exploration.py](HUGIN_gym/envs/core/rewards/agent1_exploration.py) | Tag `oyvind-final` | [fox_train.sh at `oyvind-final`](https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection/blob/oyvind-final/fox_train.sh) |
| Delineation expert, DQN, PPO, and SAC | Open, see below | [agent3_border.py](HUGIN_gym/envs/core/rewards/agent3_border.py) | Tag `erik-final` | [fox_train.sh at `erik-final`](https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection/blob/erik-final/fox_train.sh) |

**Open:** the full configurations and training results of the detection and delineation runs are not in the repository yet. The settings of the two PPO experts kept for the paper are under `SUB_POLICIES` in any run 6 config, for example [this one](docs/training_runs/HRL_PPO_sweep6_est_seed1_4387793/training_config.json).

The Overview table lists the commit of runs 1–6, which does not include the detection and delineation runs. The two tags hold the final state of those branches, which is not necessarily the commit a run started from.

### Steps

The trained models are not released at this time, and the results per episode are not in the repository. The scripts read and write two folders next to the repository: `../trained-agents` and `../eval-results`. The Fox scripts use the account `ec12`, so change it to your own.

1. **Train the three experts.** Train the characterization expert on `main`, and the other two from the tags. Meta agent training and every evaluation need these models, even the lawnmower evaluation. The scripts load them from these paths, set in [train_HRL(ppo).py](<HUGIN_gym/training/train_HRL(ppo).py>) and [test_HRL(ppo).py](<HUGIN_gym/evaluation/test_HRL(ppo).py>):
   ```
   ../trained-agents/PPO_space_30M/PPO_scratch.zip              Detection
   ../trained-agents/PPO_compare_100M_4259607/PPO_scratch.zip   Characterization
   ../trained-agents/PPO_border_4280885/PPO_scratch.zip         Delineation
   ```
2. **Train the meta agents.** `run_evaluations.py` finds a run by its folder name, so a local run needs a name in this form:
   ```bash
   # All 21 runs of run 6, on Fox
   bash fox_train_HRL_matrix.sh 5000000

   # One run on your own machine
   python -m "HUGIN_gym.training.train_HRL(ppo)" --preset sweep6_est --seed 1 --max-steps 5000000 --name HRL_PPO_sweep6_est_seed1_local
   ```
3. **Evaluate the surveys.**
   ```bash
   # First set: all 118 evaluations (on Fox: sbatch fox_eval_HRL.sh)
   python -m HUGIN_gym.evaluation.run_evaluations --workers 8

   # Second set
   python -m HUGIN_gym.evaluation.run_evaluations --evaluation-seed 1 --output-dir ../eval-results/run6_seed1 \
       --only '^sources_1_3__(sweep6_est|sweep4_visit|lawnmower_sweep[23]_|rule_sweep6_(handover|scripted_gas)_estimate|rule_sweep4_handover_visit)'
   ```
4. **Build the tables and figures.**
   ```bash
   python -m HUGIN_gym.evaluation.summarise_evaluations ../eval-results/run6 --output docs/evaluation_results/run6
   python -m HUGIN_gym.evaluation.summarise_evaluations ../eval-results/run6_seed1 --output docs/evaluation_results/run6_seed1
   python -m HUGIN_gym.evaluation.plot_evaluations ../eval-results/run6 --agents-dir ../trained-agents --output docs/evaluation_results/run6
   ```

## 🔧 Configuration

The environment can be configured through various parameters:
- Training parameters in `HUGIN_gym/training`
- Meta agent variants in `PRESETS` in `HUGIN_gym/training/train_HRL(ppo).py`
- Reward functions in `HUGIN_gym/envs/core/rewards`
- Observable features in `HUGIN_gym/agents/feature_extractor`
- End goals in `HUGIN_gym/envs/wrappers/GPWrapper.py`
- AUV movement in `HUGIN_gym/envs/HUGIN_env.py` and `HUGIN_gym/envs/core/dynamics.py`
- Visualization settings in `HUGIN_gym/envs/core/visualisation/renderer.py`

## ⚠️ Intended Use and Limitations

This repository is a research tool. It has not been validated for real leak detection and must not be used as one. This applies to the code and to any trained models we may release later.

**Intended use**
- Research and teaching on (hierarchical) reinforcement learning for autonomous search and gas plume mapping.
- Comparing algorithms, reward functions, and observation designs in simulation.
- A starting point for methods that will later be validated with real sensor data and field tests.

**Out of scope**
- Detecting, or ruling out, real gas leaks at pipelines, CCS sites, or other subsea installations.
- Providing evidence of "no leak" for regulators, audits, or environmental reporting.
- Controlling a real AUV without further development, testing, and safety review.
- Military use, or mapping subsea infrastructure for hostile purposes.

**Known limitations**
- The simulator is simplified. It uses a 2D map of 41 x 41 cells, one to three Gaussian gas sources, five discrete movement actions, and no ocean currents or sensor noise. Real plumes can be stretched by currents or spread out as diffuse seepage, and the trained agents may miss them.
- When the Gaussian process (GP) wrapper is on, the goal check uses the GP estimate of the plume. An episode can count as complete even though the true plume coverage is lower, and with the estimate goal the AUV does not need to visit the cells it counts as done. Treat reported coverage as an estimate, not as ground truth.
- The lead over the baselines holds only near the plume sizes the meta agent trained on (σ 2–5). With small plumes (σ 1–2), surveys miss more plumes, and the main meta-agent finds 92.5% of them.
- The expert runs and the meta agent runs 4–5 are not seeded, and the meta agent runs of run 6 use training seeds 1–3. Results vary between runs, so compare the mean and spread over several repeats.
- No result in this repository has been tested on real data.

**If you build on this work**
- Validate on real or realistic data before any field use.
- Report the uncertainty of the plume estimate together with any coverage result.
- Use the agent to support human monitoring, not to replace it, and keep a human in the review loop.
- Treat leak maps and survey data near pipelines and cables as sensitive.
- Follow the laws and permits that apply where you operate. Operating in another state's waters requires that state's consent.

**Why we release the code**
The repository contains no real survey data, no locations of real infrastructure, and no interface to a real vehicle. It uses standard algorithms from Stable-Baselines3 in a simplified simulator. Releasing it therefore adds little capability for misuse, while it makes the results possible to check and reproduce.

The MIT License permits any use. This section does not change the license. It states what the code was built and tested for, and how we ask others to use it.

## 📄 License

This project is licensed under the MIT License, see [LICENSE](LICENSE). The copyright holders are Matthias Schmitt, Martin Haugen, Erik Johnsrud, and Øyvind L. Rekdal.

## 📧 Contact

- Martin Haugen - martin27@ifi.uio.no
- Erik Johnsrud - erijohn@uio.no
- Øyvind L. Rekdal - oyvindlr@ifi.uio.no

Author of the thesis code: Matthias Schmitt - [@MatthiasHansHeinrichSchmitt](https://github.com/MatthiasHansHeinrichSchmitt)

Project Link: [https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection.git](https://github.com/martinhaugen99/HRL_DRL_gas_seep_detection.git)
