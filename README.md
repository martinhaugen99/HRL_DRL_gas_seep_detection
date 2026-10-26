# HUGIN HRL Gymnasium Environment for gas seepage detection (HRL-GSD)

A gymnasium-based environment for simulating and training reinforcement learning agents on the HUGIN AUV. This environment provides a simplified simulation of the HUGIN.

![Simulation visualisation](Sim_visualiser.png)
## 🌊 Features

- **Concentration Distribution**: Visualising a concentration distribution in form of opacity dependent bubbles 
- **3D Visualization**: Real-time 3D rendering using Meshcat
- **Custom Rewards**: Configurable reward functions for different tasks
- **Feature Extraction**: Individual feature extraction for the different sub-agents (detection, characterisation, and delineation)
- **Stable-Baselines3 Compatible**: Ready to use with popular RL frameworks (e.g. PPO, DDQN, SAC) + logging


## 🛠️ Installation

### Prerequisites
- Python ≥3.10
- pip



### Using pip
```bash
# Clone the repository
git clone https://github.com/MatthiasHansHeinrichSchmitt/HRL_DRL_gas_seep_detection.git
cd HUGIN_gym

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## 📊 Examples

The `evaluation` and `training` directory contains several scripts demonstrating different uses:

- `test_XYZ.py` (e.g. `test_PPO.py`): Basic environment testing with manual control and evaluation of a trained model or a lawnmower pattern
- `train_XYZ.py` (e.g. `train_PPO.py`): Training script using PPO,DDQN or SAC

### Running Examples
```bash
# Test environment with manual control
python -m HUGIN_gym.evaluation.test_PPO

# Train an expert agent with PPO
python -m HUGIN_gym.training.train_PPO
```

## 🖼️ Visualization

The environment uses Meshcat for 3D visualization. When running with `render_mode="human"`, a web browser window will open automatically showing the simulation. The visualisation includes:
- AUV model (pose aligned)
- trajectory trackers (green pointers)
- gas plume (grey bubbles: concentration dependent oppacity and radius)
- visited cell marking (green cubes with low opacity)

## 📚 Project Structure
```
the_project/
├── HUGIN_gym/                  The (H)RL python package
│   ├── envs/                       Environments (base & HRL env)
│   │   ├── core/                       Dynamics, rewards, visualisation
│   │   └── wrappers/                   Env wrappers (obs, GP)
│   ├── agents/                     RL agents & HRL components
│   │   ├── feature_extractor/          Feature extractors 
│   │   └── term_fn/                    Termination conditions (by default off)
│   ├── training/                   Training scripts (DDQN, PPO, SAC, HRL)
│   ├── evaluation/                 Evaluation, analysis, and debugging scripts
│   ├── callbacks/                  Logging training data
│   ├── utils/                      Saving/Loading training config
│   └── assets/                     AUV model (.dae)
├── HUGIN_gym.egg-info/         Packaging metadata
├── pyproject.toml, setup.py    Build config (SB3, gym, meshcat, scikit, ... )
├── job.sh                      SLURM cluster job
└── README.md,  
```

The central environment is HUGIN_gym/envs/HUGIN_env.py. It models:
- A bounded 3D voxel domain, nominally 41 x 41 x 41
- AUV position and orientation
- Five discrete movement actions
- One or more Gaussian gas sources
- Visited, above-threshold, and plume-border maps
- Task-specific observations, rewards, and termination conditions

Three low-level tasks are supported:
- Space exploration: maximize general map coverage
- Plume exploration: map regions above the concentration threshold
- Border delineation: trace the plume's threshold boundary

Reward implementations are under HUGIN_gym/envs/core/rewards/.

### Hierarchical RL

HUGIN_gym/envs/HRL_env.py provides a meta-environment. A high-level controller chooses among pretrained low-level policies:
1. Explore the space
2. Characterize the plume
3. Delineate its border

The selected policy executes primitive movements for several steps before control returns to the meta-controller. Option wrappers are defined in HUGIN_gym/agents/options.py.

### Mapping

HUGIN_gym/envs/wrappers/GPWrapper.py adds belief-based mapping:
- Collects position and concentration measurements
- Fits a scikit-learn Gaussian process
- Predicts concentrations throughout the domain
- Builds estimated plume and border maps
- Calculates coverage, RMSE, and classification metrics

This is computationally expensive in 3D because the GP may be refitted and evaluated over the full grid after many actions.

### Training

Training scripts are in HUGIN_gym/training/:
- train_PPO.py
- train_DDQN.py, which actually uses Stable-Baselines3's DQN
- train_SAC.py
- train_HRL(ppo).py
- train_HRL(ddqn).py

They use custom feature extractors from HUGIN_gym/agents/feature_extractor/, vectorized environments, checkpoints, and episode-statistics callbacks.

### Evaluation And Visualization

HUGIN_gym/evaluation/ contains:
- Trained-policy rollouts
- Manual control
- Vectorized validation
- HRL evaluation
- Lawnmower-pattern baselines
- Training-curve and confusion-matrix analysis

Meshcat visualization under HUGIN_gym/envs/core/visualisation/ renders the AUV, trajectory, domain, and concentration field. Matplotlib utilities visualize GP predictions and map coverage.

## 🔧 Configuration

The environment can be configured through various parameters:
- Training parameters in `HUGIN_gym\training`
- Reward functions in `HUGIN_gym\envs\core\rewards`
- Observable features in `HUGIN_gym\agents\feature_extractor`
- Physics parameters in `HUGIN_gym\envs\core\dynamics.py`
- Visualization settings in `HUGIN_gym\envs\core\visualisation\renderer.py`

## ⚠️ Intended Use and Limitations

This repository is a research tool. It has not been validated for real leak detection and must not be used as one. This applies to the code and to any trained models released with it.

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
- The simulator is simplified. It uses a 41 x 41 x 41 voxel domain, Gaussian gas sources, five discrete movement actions, and no ocean currents or sensor noise. Real plumes can be stretched by currents, spread out, or come from several sources, and the trained agents may miss them.
- When the Gaussian process (GP) wrapper is on, the goal check uses the GP estimate of the plume. An episode can count as complete even though the true plume coverage is lower. Treat reported coverage as an estimate, not as ground truth.
- Training runs are not seeded. Results vary between runs, so compare the mean and spread over several repeats.
- No result in this repository has been tested on real data.

**If you build on this work**
- Validate on real or realistic data before any field use.
- Report the uncertainty of the plume estimate together with any coverage result.
- Use the agent to support human monitoring, not to replace it.
- Treat leak maps and survey data near pipelines and cables as sensitive.
- Follow the laws and permits that apply where you operate. Operating in another state's waters requires that state's consent.

**Why we release the code**
The repository contains no real survey data, no locations of real infrastructure, and no interface to a real vehicle. It uses standard algorithms from Stable-Baselines3 in a simplified simulator. Releasing it therefore adds little capability for misuse, while it makes the results possible to check and reproduce.

The MIT License permits any use. This section does not change the license. It states what the code was built and tested for, and how we ask others to use it.

## 📄 License

This project is licensed under the MIT License

## 📧 Contact

Matthias Schmitt - [@MatthiasHansHeinrichSchmitt](https://github.com/MatthiasHansHeinrichSchmitt) 

Project Link: [https://github.com/MatthiasHansHeinrichSchmitt/HRL_DRL_gas_seep_detection.git](https://github.com/MatthiasHansHeinrichSchmitt/HRL_DRL_gas_seep_detection.git)
