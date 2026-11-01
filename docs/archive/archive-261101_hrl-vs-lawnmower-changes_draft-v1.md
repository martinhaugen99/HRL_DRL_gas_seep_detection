# Making the HRL survey better than a lawnmower: the first five changes

**Date:** 2026-11-01. **Branch:** `martin`. **Status:** implemented and tested in simulation. Nothing has been retrained yet.

## Why we made these changes

A fixed lawnmower with tracks 2 cells apart reaches the end goal in every episode, in 366 AUV steps. Our best trained meta agent reaches it in 89 % of the episodes, in 345 AUV steps on average. So we have not yet shown that HRL gives more than a fixed path.

The idea to test: sweep the map with wide tracks and spend extra steps only where gas has been found. A lawnmower cannot do this, because it follows the same path whatever it measures. No agent is told how many plumes there are.

These five changes let us test the idea without training anything.

| Term | Meaning |
|---|---|
| AUV step | One move of the AUV. |
| Decision | One choice by the meta agent. It covers 3 AUV steps in these tests. |
| GP map | The plume and border cells as the GP estimates them from the measurements. |
| GP std | How uncertain the GP is about a cell. "Mean GP std" is the average over the map. |
| Map overlap | Cells in both the GP map and the true map, divided by cells in either. 1.0 means identical. |
| Plume | A connected region of true cells above the threshold. Sources that overlap count as one plume. |
| Sweep *N* | A lawnmower with tracks *N* cells apart. |

## Change 1: Baselines and map metrics

**Files:** [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>), [baselines.py](../HUGIN_gym/evaluation/baselines.py), [survey_metrics.py](../HUGIN_gym/evaluation/survey_metrics.py)

**How it was:** the evaluation script always loaded a trained model and always used one source.

```python
MULTIPLE_GAUSSIANS = [1,1]
model = PPO.load(model_path)
```

**What that meant:** trained variants could only be compared with each other. The results showed how much the AUV visited, not how correct the final map was. With several plumes, one global coverage number could hide a plume that was hardly mapped.

**How it is now:** the script can run a meta agent without learning, with a random number of sources, and it reports the quality of the map.

```bash
python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 2 --sources 1 3
```

| Baseline | What it does |
|---|---|
| `lawnmower` | Always chooses the sweep. |
| `space` | Always chooses the trained space agent. |
| `rule` | Chooses with a simple rule (see change 5). |
| `random` | Chooses each sub-agent with probability 1/3. |

**What that means:** all agents and baselines get the same 1,000 episodes (seed 0) and the same metrics: map overlap, true cells visited, plumes found, plumes at least 90 % visited, and a table per number of plumes.

**What we hope to see:** a fair benchmark that a learned agent must beat. An agent that adapts should use more steps with more plumes, while a lawnmower should stay flat.

**First result:** the steps do behave this way (see [Results](#results)). `minimal_values` and `untuned` also reproduce their earlier evaluations (344.8 and 376.7 AUV steps), so the script still measures the same thing.

## Change 2: A faster GP with the same result

**Files:** [incremental_gp.py](../HUGIN_gym/envs/core/incremental_gp.py), [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** the GP was refitted from scratch after every AUV step.

```python
kernel.fit(X, y)
mean, std = kernel.predict(self.grid_positions, return_std=True)
```

**What that meant:** every step redid the whole calculation for all measurements and all 1,681 cells, even though the kernel settings never change. The GP took 88 % of the time per step on a PC, and an update grew from about 1 ms early in an episode to about 20 ms late. Only the mean GP std was kept, not the std per cell.

**How it is now:** a new measurement adds one row to the existing calculation, and the std per cell is stored.

```python
self.incremental_gp.add(position, concentration)
self.GP_std = self.incremental_gp.std.reshape(self.grid_shape)
```

**What that means:** the mathematics is the same, with no refit. sklearn is still used for other kernels and if the limit of 2,000 measurements is passed. Training uses the fast GP by default.

**What we hope to see:** the same results in much less time, so we can run more repeats and more training steps. It also makes it practical to train agents that read the GP at every step.

**First result:** the printed output was identical to sklearn on 40 episodes of `minimal_values` and 20 of `duration`, and it ran 7–8 times faster on a PC. The speed on Fox has not been measured yet.

## Change 3: A second end goal, "estimate"

**Files:** [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** an episode ended when the mean GP std was at most 0.1 and the AUV had visited 90 % of the estimated plume and border cells.

```python
elif MEAN_uncertainty <= (1 - accuracy_goals[0]) and plume_coverage >= accuracy_goals[1] and border_coverage >= accuracy_goals[2]:
```

**What that meant:** the AUV had to pass over cells that the GP may already know well, and the 0.1 limit needs tight tracks over the whole map. This forces a dense survey and leaves little room to beat a lawnmower. For example, sweep 4 already gives a map overlap of 0.97 / 0.94, but it fails this goal because only 75 % of the plume cells are visited.

**How it is now:** `goal_mode` chooses the goal, and `"visit"` is still the default. With `"estimate"`, a cell counts as done when the GP knows it well, visited or not, and `goal_mean_std` sets the mean GP std that ends the episode.

```python
known = self.GP_std <= self.known_std  # 0.1
shares.append(np.sum(cells & known) / np.sum(cells) if cells.any() else 1.0)
```

**What that means:** the goal asks "do we know the plume and its border?" instead of "has the AUV been on each cell?".

**What we hope to see:** the same map quality in fewer steps, with most of the extra measurements near gas.

**First result:** with one source, the rule with sweep 6 uses 243 AUV steps against 366 for the lawnmower, with the same plume overlap (0.996). See [What is not solved yet](#what-is-not-solved-yet) for 1–3 sources.

## Change 4: The lawnmower as an option

**Files:** [options.py](../HUGIN_gym/agents/options.py), [HRL_env.py](../HUGIN_gym/envs/HRL_env.py), [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>)

**How it was:** the meta agent could only choose among the three trained sub-agents.

```python
option1 = Option(policy1, termination_fn1, "spatial exploration", obs_keys=[...])
```

**What that meant:** the meta agent could never fall back on a clean lawnmower. The space agent trained without the GP, to visit 90 % of the map, and it hit the map edge 4–10 times per episode in a rough check of 60 episodes.

**How it is now:** a scripted sweep has the same interface as a sub-agent. It can replace the space agent, and the track spacing is a setting.

```python
if SPACING is not None:
    option1 = SweepOption(base_env, gp_env, SPACING, name=f"lawnmower sweep, tracks {SPACING} cells apart")
```

The sweep plans waypoints from the start corner and finds the shortest path with the AUV's real moves. After another option has moved the AUV, it goes back to its next waypoint. It skips waypoints the GP already knows.

**What that means:** always choosing this option is a lawnmower, so the lawnmower becomes the worst case the meta agent can fall back on. Anything better comes from choosing when to leave the sweep.

**What we hope to see:** the meta agent learns when to leave the sweep for gas and when to return, and wide tracks save steps where there is no gas.

**First result:** always choosing sweep 2 ends all 1,000 episodes in 366 AUV steps. The tests also show no edge hits from any of the 8 start positions.

## Change 5: A simple rule as a benchmark for the meta agent

**Files:** [baselines.py](../HUGIN_gym/evaluation/baselines.py)

**How it was:** there was no simple meta agent to compare with.

**What that meant:** we could not tell whether a trained meta agent had learned more than the obvious choice, or whether the limit was in the meta agent or in the sub-agents.

**How it is now:** a few lines choose the sub-agent from what the GP has found so far.

```python
if not unfinished or self.stuck >= 2:
    choice = 0  # sweep or space agent
elif self.stuck == 1 and self.last_choice in (1, 2):
    choice = 3 - self.last_choice  # the last gas agent made no progress: try the other
else:
    choice = 1 if plume_share < border_share else 2
```

**What that means:** it continues the sweep unless a found plume is unfinished, and then it uses the plume or border agent with the lower share. It gets neither the number of plumes nor the true map, and it needs no training.

**What we hope to see:** a trained meta agent must beat this rule, otherwise training has added nothing.

**First result:** the rule with the trained sub-agents ends 85 % of the episodes in 344 AUV steps, and the trained `minimal_values` ends 89 % in 345. So the meta agent is not the main limit today. The goal and the sub-agents matter more.

## Results

1,000 episodes per row, seed 0. Trained agents use sampled actions. Each row shows ended by goal, AUV steps, map overlap, true cells visited and plumes found. "G" is the mean GP std that ends the episode.

**One source:**

| Survey | Goal | Ended by goal | AUV steps | Overlap plume / border | Visited plume / border | Plumes found |
|---|---|---|---|---|---|---|
| Lawnmower, sweep 2 | Visit | 100 % | 366 | 0.996 / 0.991 | 99.5 / 98.2 % | 99.8 % |
| Lawnmower, sweep 4 | Stops at G 0.115 | 100 % | 217 | 0.970 / 0.937 | 74.9 / 75.2 % | 100 % |
| `minimal_values` | Visit | 89 % | 345 | 0.995 / 0.988 | 96.8 / 96.1 % | 99.8 % |
| `untuned` | Visit | 86 % | 377 | 0.992 / 0.985 | 97.0 / 95.6 % | 99.6 % |
| Space agent only | Visit | 83 % | 392 | 0.995 / 0.987 | 92.9 / 91.9 % | 99.8 % |
| Random | Visit | 18 % | 471 | 0.954 / 0.938 | 88.1 / 86.1 % | 96.7 % |
| Rule + trained sub-agents | Visit | 85 % | 344 | 0.992 / 0.979 | 94.7 / 94.2 % | 99.7 % |
| Rule + sweep 4 + plume, border | Visit | 82 % | 335 | 0.993 / 0.983 | 94.8 / 93.5 % | 100 % |
| **Rule + sweep 6 + plume, border** | **Estimate, G 0.22** | **94 %** | **243** | **0.996 / 0.986** | 93.5 / 94.1 % | **100 %** |

**A random 1–3 sources, not told to the agent:**

| Survey | Goal | Ended by goal | AUV steps | Overlap plume / border | Visited plume / border | Plumes found |
|---|---|---|---|---|---|---|
| Lawnmower, sweep 2 | Visit | 100 % | 370 | 0.997 / 0.992 | 98.5 / 96.8 % | 99.9 % |
| Lawnmower, sweep 4 | Stops at G 0.115 | 100 % | 216 | 0.974 / 0.941 | 74.8 / 74.4 % | 100 % |
| `minimal_values` | Visit | 85 % | 374 | 0.991 / 0.982 | 95.2 / 94.2 % | 98.8 % |
| `untuned` | Visit | 75 % | 405 | 0.994 / 0.985 | 96.1 / 94.6 % | 99.3 % |
| Space agent only | Visit | 83 % | 400 | 0.995 / 0.989 | 91.8 / 91.2 % | 99.9 % |
| Random | Visit | 13 % | 474 | 0.957 / 0.939 | 87.4 / 86.3 % | 97.1 % |
| Rule + trained sub-agents | Visit | 75 % | 377 | 0.985 / 0.971 | 93.0 / 92.8 % | 98.5 % |
| Rule + sweep 4 + plume, border | Visit | 75 % | 369 | 0.987 / 0.976 | 93.3 / 92.3 % | 99.1 % |
| **Rule + sweep 6 + plume, border** | **Estimate, G 0.22** | **91 %** | **280** | **0.991 / 0.979** | 91.6 / 92.3 % | **98.8 %** |

**Does the survey adapt to the number of plumes?** AUV steps with 1 / 2 / 3 plumes:

| Survey | AUV steps | Plumes found |
|---|---|---|
| Lawnmower, sweep 2 | 368 / 373 / 372 | 100 / 99.9 / 100 % |
| `minimal_values` | 356 / 389 / 404 | 99.8 / 99.2 / 96.3 % |
| Rule + sweep 6, estimate, G 0.22 | 266 / 291 / 305 | 100 / 98.8 / 96.9 % |

The rule uses more steps with more plumes and still stays well below the lawnmower, but it finds fewer of the plumes as their number grows.

## What is not solved yet

- **1–3 sources are not a clean win yet.** The rule with sweep 6 saves 24 % of the steps against sweep 2, but its map is a little worse, it misses 1.2 % of the plumes, and 9 % of its episodes do not reach the goal. Our targets were at least 97 % ended, overlap of at least 0.99 / 0.98, at least 99 % of plumes found and at most 260 steps. It does not meet them yet.
- **Sweep 4 is hard to beat on speed.** It takes only 216 steps and finds all plumes, with a lower overlap (0.974 / 0.941).
- **The mean GP std can hide a gap.** A likely cause of the missed plumes is that an average can reach its limit while a strip of the map is still unsurveyed. Lowering G from 0.26 to 0.22 raised plumes found from 97.4 % to 98.8 %, for 10 more steps.
- **The plume and border agents get stuck.** They made no progress in 16–33 % of their decisions in a rough check of 60 episodes, and this probably explains many of the episodes that do not end.
- **The meta reward still pays for visited cells.** It must be changed before training a meta agent on the estimate goal.
- **The rule is not learning.** These results show what is possible with the current sub-agents, not what a trained agent reaches.
- **Limits of these numbers:** the speed was measured on a PC, not on Fox.

## Next steps

1. Check the speed of the fast GP on Fox.
2. Add a check to the estimate goal for large unknown areas, so the mean GP std cannot hide a gap.
3. Change the meta reward and the meta state, and train the meta agent on a random 1–3 sources.
4. If the plume and border agents remain the limit, test a scripted gas option and then retrain them.

## How to reproduce the results

Run from the repository root (on Windows: `.venv/Scripts/python`).

```bash
# lawnmower, sweep 2
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 2 --sources 1 3

# lawnmower, sweep 4, which stops at the end of its path
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 4 --sources 1 3 --goal estimate --goal-mean-std 0.115 --known-std 0.5

# rule with sweep 6, estimate goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 6 --sources 1 3 --goal estimate --goal-mean-std 0.22 --known-std 0.1

# a trained agent with sampled actions
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --model ../trained-agents/HRL_PPO_minimal_values_4371280/HRL_PPO_scratch.zip --mode 4 --sources 1 3
```

Use `--sources 1 1` for one source.
