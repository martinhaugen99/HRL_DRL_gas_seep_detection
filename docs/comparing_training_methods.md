# Comparing DQN, PPO and SAC for the plume agent

*Notes from 2026-09-26, branch `martin`.*

**Question:** the previous thesis focused on SAC because it was found to work best. Is that really the case? To find out, all three methods (DQN, PPO, SAC) need to be trained and compared fairly.

**Short answer:** make the **task, the training budget, the network size and the evaluation identical**. Do **not** force the algorithm-specific hyperparameters to be equal: they mean different things in each method, so equal values don't make the comparison fairer. Give each method reasonable settings of its own, with the same amount of tuning effort for each.

---

## 1. Settings that must be the same

These define the task, the budget and the network capacity. Values as of 2026-09-26:

| Setting | DQN ([train_DDQN.py](../HUGIN_gym/training/train_DDQN.py)) | PPO `--paper` ([train_PPO_config.py](../HUGIN_gym/training/train_PPO_config.py)) | SAC ([train_SAC.py](../HUGIN_gym/training/train_SAC.py)) | Suggested for all |
|---|---|---|---|---|
| `GP` / `SUB_AGENT_TRAIN_ON_GP` | True / True | **False** (from the preset) | **False / False** | True / True (the HRL setup uses the GP) |
| `MAX_EPS_LEN` (steps per episode) | 230 | **460** | **460** | 230 |
| Training steps (`max_steps`) | **30M** | 10M | 10M | same for all, e.g. 10M |
| `gamma` (discount factor) | 0.9945 | 0.9975 | 0.997 | one value, e.g. 0.995 |
| Hidden layers after the feature extractor | [64, 64] | [64, 64] | **[256, 256]** (SB3 default) | same, e.g. [64, 64] |
| Checkpoint interval | 2M | 1M | 2M | same, e.g. 1M |
| Parallel environments (`NUM_ENVS`) | 12 | 12 | 12 | already the same |
| Accuracy goals, near-plume start, plume sizes | same | same | same | already the same |

Notes:

- **The first Fox comparison was not like-for-like.** Unless the scripts were changed on Fox, DQN trained with the GP and 230-step episodes, while PPO and SAC trained without the GP and with 460-step episodes: an easier, different task.
- **`gamma` defines what the agent optimises**, so different values mean different goals. It is usually tied to episode length: 0.995 corresponds to roughly 200 steps.
- **Network size:** SB3 quietly gives SAC networks 4× wider than DQN and PPO get. Set `net_arch` in `policy_kwargs` for all three:
  - DQN: `policy_kwargs = dict(features_extractor_class=Agent, net_arch=[64, 64])`
  - PPO: `policy_kwargs = dict(features_extractor_class=Agent, net_arch=dict(pi=[64, 64], vf=[64, 64]))`
  - SAC: `policy_kwargs = dict(features_extractor_class=Agent, net_arch=[64, 64])` (used for the actor and both critics)
- **Observation keys:** the lists differ slightly between the scripts, but the plume network (`AgentPlumeExplore`) reads the same five inputs in all three, so this does not matter.
- **PPO:** the `--paper` preset has `gp: False` and `max_episode_length: 460`. A new preset (e.g. `COMPARE_CONFIG`) with GP on and 230 steps is needed, and `fox_train.sh` should use it instead of `--paper`.

## 2. Settings to leave different

| Method | Its own settings |
|---|---|
| DQN | `learning_rate`, `buffer_size`, `learning_starts`, `batch_size`, `target_update_interval`, `train_freq` / `gradient_steps`, `exploration_fraction` / `exploration_final_eps` |
| PPO | `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm` |
| SAC | `learning_rate`, `buffer_size`, `batch_size`, `tau`, `train_freq` / `gradient_steps`, `ent_coef` |

- Even names they share mean different things. PPO's `batch_size` is a slice of freshly collected data; for DQN and SAC it is a sample from stored past experience.
- **Fair means equal tuning effort:** either standard values for every method, or the same small search for each (for example 3 learning rates per method, run short).
- **One value to change:** DQN's `exploration_fraction=0.8` keeps it acting mostly at random for most of training (the chance of a random action only reaches its final 5% after 80% of the run). That handicaps DQN; 0.1–0.3 is standard.

## 3. How to compare

1. **Several runs per method.** RL results vary a lot between runs with the same settings, so one run each cannot show that SAC is best. Use 3–5 runs per method and report the mean and spread. With `fox_train.sh`, just submit several times; each submission gets its own job ID, so run names never collide.
2. **Compare with the evaluation script, not the training curves.** During training DQN takes random actions on purpose, and PPO and SAC sample their actions, so their training rewards are not comparable. Evaluate each method:
   - without randomness in the actions (`deterministic=True`);
   - on the same plumes and start positions: call `np.random.seed(0)` once before the evaluation loop, and every method gets the identical sequence of plumes and starts;
   - with the same number of episodes, e.g. 200.
3. **Metrics:**
   - **True plume coverage:** `info["actual_current_above_threshold"]` in GP mode. Not `obs["plume_coverage"]`, which is the GP's estimate.
   - **Success rate:** episodes that finished before the step limit.
   - **Steps needed.**
4. **Learning speed:** evaluate the saved checkpoints at the same step counts (this is why the checkpoint intervals should match) and plot evaluation performance against training steps. Also report wall-clock time on the same Fox resources: SAC and DQN do many more network updates per step than PPO.

## 4. Two things to state in the thesis

- **SAC runs through a workaround.** SAC needs continuous actions, so [continuous_wrapper.py](../HUGIN_gym/envs/wrappers/continuous_wrapper.py) cuts a single number between −1 and 1 into the 5 discrete actions:

  | Number | Action |
  |---|---|
  | −1.0 to −0.6 | 0 forward |
  | −0.6 to −0.2 | 1 up (always hits the boundary penalty in 2D) |
  | −0.2 to 0.2 | 2 down (always hits the boundary penalty in 2D) |
  | 0.2 to 0.6 | 3 left turn |
  | 0.6 to 1.0 | 4 right turn |

  Forward and right turn sit at opposite ends, and the two useless actions sit in the middle, between forward and left turn. Any SAC result, good or bad, is really "SAC with this action encoding".
- **SB3's DQN is plain DQN, not Double DQN.** The installed SB3 computes its targets with the maximum of the target network (`dqn/dqn.py`). The script name `train_DDQN.py` suggests otherwise, so call it DQN in the thesis.

## 5. Next steps

- [ ] Choose the shared values (GP on, 230 steps, `gamma`, training budget, `net_arch`, checkpoint interval).
- [ ] Put them in all three scripts, including a new PPO preset with GP on and 230 steps, and make `fox_train.sh` use it.
- [ ] Change DQN's `exploration_fraction` to 0.1–0.3.
- [ ] Decide how each method's own settings are chosen: standard values, or the same small search for each.
- [ ] Write one evaluation script that loads all three models, runs them on the same plumes (`np.random.seed(0)`) and prints a comparison table.
- [ ] Run 3–5 runs per method on Fox and evaluate the checkpoints at matching step counts.
