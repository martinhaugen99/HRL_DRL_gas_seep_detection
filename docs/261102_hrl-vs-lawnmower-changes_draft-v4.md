# Making the HRL survey better than a lawnmower: the first seven changes

**Date:** 2026-11-02. **Branch:** `martin`. **Status:** implemented and tested in simulation, and the speed is measured on Fox. Nothing has been retrained yet.

## Why we made these changes

A fixed lawnmower with tracks 2 cells apart reaches today's end goal in every episode, in 366 AUV steps. Our best trained meta agent reaches it in 89 % of the episodes, in 345 AUV steps on average. So we have not yet shown that HRL gives more than a fixed path.

The idea to test: sweep the map with wide tracks and spend extra steps only where gas has been found. A lawnmower cannot do this, because it follows the same path whatever it measures. No agent is told how many plumes there are.

These seven changes let us test the idea without training anything.

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

**First result:** the printed output was identical to sklearn on 40 episodes of `minimal_values` and 20 of `duration`. Training of `untuned` with 12 environments also became much faster:

| Where and when | Old GP | Fast GP | Faster by |
|---|---|---|---|
| Fox, at 100,000 decisions | 99 decisions/s | 592 decisions/s | 6.0 times |
| Fox, at 200,000 decisions | 99 decisions/s | 574 decisions/s | 5.8 times |
| Fox, at 500,000 decisions | 91 decisions/s | 582 decisions/s | 6.4 times |
| PC, at 61,440 decisions | 109 decisions/s | 1,562 decisions/s | 14 times |

At the 579 decisions/s of the Fox job, 10M decisions take about 4.8 h, against 22–27 h in run 4, so `fox_train_HRL.sh` now asks for 10 h instead of 30 h. The Fox numbers compare a 20-minute job (4380825, node c1-28) with the start of run 4 (job 4363368, node c1-19). The job used about 29 % of its 16 CPUs, though that figure may be incomplete because the job hit its time limit. So something other than the GP now limits the speed, and we have not profiled what.

## Change 3: A second end goal, "estimate"

**Files:** [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** an episode ended when the mean GP std was at most 0.1 and the AUV had visited 90 % of the estimated plume and border cells.

```python
elif MEAN_uncertainty <= (1 - accuracy_goals[0]) and plume_coverage >= accuracy_goals[1] and border_coverage >= accuracy_goals[2]:
```

**What that meant:** the AUV had to pass over cells that the GP may already know well, and the 0.1 limit needs tight tracks over the whole map. This forces a dense survey.

**How it is now:** `goal_mode` chooses the goal, and `"visit"` is still the default. With `"estimate"`, a cell counts as done when the GP knows it well, visited or not, and `goal_mean_std` sets the mean GP std that ends the episode.

```python
known = self.GP_std <= self.known_std  # 0.1
shares.append(np.sum(cells & known) / np.sum(cells) if cells.any() else 1.0)
```

**What that means:** the goal asks "do we know the plume and its border?" instead of "has the AUV been on each cell?".

**What we hope to see:** the same map quality in fewer steps, with most of the extra measurements near gas.

**First result:** the goal saves steps for every survey, including a plain lawnmower. With 1–3 sources, sweep 3 reaches it in 282 AUV steps, against 370 for sweep 2 on the visit goal, with a nearly equal map (0.994 / 0.984 against 0.997 / 0.992). So on this goal, sweep 3 is the lawnmower to beat.

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

**First result:** the rule with the trained sub-agents ends 85 % of the episodes in 344 AUV steps, and the trained `minimal_values` ends 89 % in 345. So the meta agent is not the main limit today.

## Change 6: A check for large unknown areas

**Files:** [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** the estimate goal only looked at the mean GP std over the map.

```python
elif MEAN_uncertainty <= max_mean_std and plume_share >= accuracy_goals[1] and border_share >= accuracy_goals[2]:
```

**What that meant:** an average can reach its limit while part of the map has no measurements. Extra measurements near gas lower the mean, so the survey stopped early: 99 % of the episodes that ended by the goal still had sweep waypoints left. The rule with sweep 6 missed 2.6 % of the plumes, and 38 of those 42 plumes were in episodes that ended by the goal. The GP std inside them averaged 0.91, so almost nothing had been measured there.

**How it is now:** with the estimate goal, the episode can only end when no connected area with a GP std above 0.8 has more than 10 cells.

```python
def largest_unknown_area(self):
    labels, n_areas = ndimage.label(self.GP_std > self.unknown_std)
    return int(np.max(np.bincount(labels.ravel())[1:])) if n_areas else 0
```

**What that means:** the survey cannot stop while a gap large enough to hold a plume is left. The two limits come from measurements: a complete sweep with tracks up to 6 apart leaves no cell above 0.8, and the smallest plume in 5,000 sampled maps has 23 cells. The visit goal does not change.

**What we hope to see:** no plumes missed in surveys that end by the goal, and a mean GP std limit that no longer has to be tuned.

**First result:**

| Rule + sweep 6, 1–3 sources | Ended by goal | AUV steps | Overlap plume / border | Plumes found |
|---|---|---|---|---|
| Mean GP std limit 0.26, no check | 92 % | 270 | 0.983 / 0.971 | 97.4 % |
| Mean GP std limit 0.22, no check | 91 % | 280 | 0.991 / 0.979 | 98.8 % |
| **Limit 0.26, with the check** | **90 %** | **291** | **0.995 / 0.984** | **99.7 %** |

No plume was missed in any episode that ended by the goal. The 5 plumes still missed (of 1,614) are all in episodes that ran out of the 480-step budget, with about 74 waypoints left. A limit of 0.30 gives the same result as 0.26 at a stricter setting of 0 cells, so the mean limit hardly matters any more. Allowing 0, 10 or 25 cells gave the same plumes found (99.7 %) and 293, 291 and 289 steps.

## Change 7: A scripted gas option

**Files:** [options.py](../HUGIN_gym/agents/options.py), [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>)

**How it was:** only the trained plume and border agents could map gas.

```python
option2 = Option(policy2, termination_fn2, "gas plume exploration", obs_keys=[...])
option3 = Option(policy3, termination_fn2, "gas border exploration", obs_keys=[...])
```

**What that meant:** we could not tell how good an adaptive survey can be, or how much of today's result is limited by those two agents.

**How it is now:** `--scripted-gas` replaces both agents with a scripted option that takes the shortest path to the nearest unfinished cell of the GP map.

```python
gas = (base.GT_c_over_threshold_maps | base.GT_c_around_threshold_maps)[:, :, 0] & ~corners
if self.gp_env.goal_mode == "visit":
    unfinished = gas & ~base.state["visited_map"][:, :, 0, 0]
else:
    unfinished = gas & (std > self.gp_env.known_std)
```

**What that means:** it is a benchmark, not the final system. It uses only the GP map, so it gets neither the number of plumes nor the true map.

**What we hope to see:** whether an adaptive survey can beat the lawnmower at all, and how many steps better gas handling is worth.

**First result:** it beats the lawnmower on both goals, with a map at least as good and all plumes found (see [Results](#results)). With 1–3 sources it uses 277 steps on the visit goal, against 370 for sweep 2, and 226 on the estimate goal, against 282 for sweep 3. The trained plume and border agents reach 369 steps with 75 % ended and 291 steps with 90 % ended.

## Results

1,000 episodes per row, seed 0, unless noted. Trained agents use sampled actions. "Estimate" uses a mean GP std limit of 0.26, known cells at GP std 0.1 or lower, and the check from change 6. "Scripted gas" is change 7.

**One source:**

| Survey | Goal | Ended by goal | AUV steps | Overlap plume / border | Visited plume / border | Plumes found |
|---|---|---|---|---|---|---|
| Lawnmower, sweep 2 | Visit | 100 % | 366 | 0.996 / 0.991 | 99.5 / 98.2 % | 99.8 % |
| Lawnmower, sweep 3 | Estimate | 99.5 % | 283 | 0.993 / 0.982 | 85.1 / 85.1 % | 100 % |
| Lawnmower, sweep 4 | Stops at its path end | 100 % | 217 | 0.970 / 0.937 | 74.9 / 75.2 % | 100 % |
| `minimal_values` | Visit | 89 % | 345 | 0.995 / 0.988 | 96.8 / 96.1 % | 99.8 % |
| `untuned` | Visit | 86 % | 377 | 0.992 / 0.985 | 97.0 / 95.6 % | 99.6 % |
| Space agent only | Visit | 83 % | 392 | 0.995 / 0.987 | 92.9 / 91.9 % | 99.8 % |
| Random | Visit | 18 % | 471 | 0.954 / 0.938 | 88.1 / 86.1 % | 96.7 % |
| Rule + trained sub-agents | Visit | 85 % | 344 | 0.992 / 0.979 | 94.7 / 94.2 % | 99.7 % |
| Rule + sweep 4 + plume, border | Visit | 82 % | 335 | 0.993 / 0.983 | 94.8 / 93.5 % | 100 % |
| Rule + sweep 6 + plume, border | Estimate | 93 % | 244 | 0.996 / 0.987 | 93.6 / 94.2 % | 100 % |
| **Rule + sweep 4 + scripted gas** | **Visit** | **100 %** | **252** | **0.998 / 0.993** | 98.8 / 97.0 % | **100 %** |
| **Rule + sweep 6 + scripted gas** | **Estimate** | **100 %** | **195** | **0.997 / 0.989** | 93.9 / 93.9 % | **100 %** |

**A random 1–3 sources, not told to the agent:**

| Survey | Goal | Ended by goal | AUV steps | Overlap plume / border | Visited plume / border | Plumes found |
|---|---|---|---|---|---|---|
| Lawnmower, sweep 2 | Visit | 100 % | 370 | 0.997 / 0.992 | 98.5 / 96.8 % | 99.9 % |
| Lawnmower, sweep 3 | Estimate | 100 % | 282 | 0.994 / 0.984 | 84.6 / 84.1 % | 100 % |
| Lawnmower, sweep 4 | Stops at its path end | 100 % | 216 | 0.974 / 0.941 | 74.8 / 74.4 % | 100 % |
| `minimal_values` | Visit | 85 % | 374 | 0.991 / 0.982 | 95.2 / 94.2 % | 98.8 % |
| `untuned` | Visit | 75 % | 405 | 0.994 / 0.985 | 96.1 / 94.6 % | 99.3 % |
| Space agent only | Visit | 83 % | 400 | 0.995 / 0.989 | 91.8 / 91.2 % | 99.9 % |
| Random | Visit | 13 % | 474 | 0.957 / 0.939 | 87.4 / 86.3 % | 97.1 % |
| Rule + trained sub-agents | Visit | 75 % | 377 | 0.985 / 0.971 | 93.0 / 92.8 % | 98.5 % |
| Rule + sweep 4 + plume, border | Visit | 75 % | 369 | 0.987 / 0.976 | 93.3 / 92.3 % | 99.1 % |
| Rule + sweep 6 + plume, border | Estimate | 90 % | 291 | 0.995 / 0.984 | 92.4 / 93.1 % | 99.7 % |
| **Rule + sweep 4 + scripted gas** | **Visit** | **99.9 %** | **277** | **0.998 / 0.992** | 98.1 / 96.5 % | **100 %** |
| **Rule + sweep 6 + scripted gas** | **Estimate** | **100 %** | **226** | **0.997 / 0.989** | 92.9 / 93.1 % | **100 %** |

**Does the survey adapt to the number of plumes?**

| Survey | AUV steps with 1 / 2 / 3 plumes | Plumes found |
|---|---|---|
| Lawnmower, sweep 2 (visit) | 368 / 373 / 372 | 100 / 99.9 / 100 % |
| Lawnmower, sweep 3 (estimate) | 283 / 281 / 280 | 100 / 100 / 100 % |
| `minimal_values` (visit) | 356 / 389 / 404 | 99.8 / 99.2 / 96.3 % |
| Rule + sweep 6 (estimate) | 272 / 305 / 326 | 100 / 99.7 / 99.1 % |
| Rule + sweep 4 + scripted gas (visit) | 267 / 286 / 296 | 100 / 100 / 100 % |
| Rule + sweep 6 + scripted gas (estimate) | 209 / 240 / 257 | 100 / 100 / 100 % |

## What this shows

- **An adaptive survey beats the lawnmower on both goals.** With 1–3 sources, the rule with scripted gas uses 25 % fewer steps on the visit goal and 20 % fewer on the estimate goal, with a map at least as good and all plumes found. It is not told how many plumes there are, and its steps grow with the plumes it finds.
- **Gas handling is the main limit.** The same rule and sweep with the trained plume and border agents do worse than the lawnmower. The difference is worth 92 steps on the visit goal and 65 on the estimate goal.
- **The trained agents do well alone.** In its own training setup, the border agent finishes in about 22 steps with more than 90 % of the true border visited (see below). The problem appears when it is used inside a survey, and we have not yet found out why.
- **With trained agents, nothing beats the lawnmower yet with 1–3 sources.** Sweep 3 (282 steps, 100 % ended) beats the rule with sweep 6 and the trained agents (291 steps, 90 % ended).

**Targets for a learned agent, 1–3 sources:**

| Goal | Lawnmower to beat | The agent needs |
|---|---|---|
| Visit | Sweep 2: 370 steps; scripted gas reaches 277 | Fewer steps, at least 99 % ended, at least 98 / 96 % visited, no more plumes missed |
| Estimate | Sweep 3: 282 steps; scripted gas reaches 226 | Fewer steps, at least 99 % ended, overlap of at least 0.994 / 0.984, all plumes found |

Sweep 2 stays the benchmark on the visit goal because sweep 3 leaves gaps between its tracks and reaches that goal in only 6 % of the episodes.

**Limits of these numbers:** the survey results come from one seed on one PC. The rule and the scripted options do not learn, and the limits of the estimate goal were chosen by hand. The meta reward still pays for visited cells, so it must change before training on the estimate goal.

## The plume size of the border agent

The `erik` branch fixes σ at 3.5 in all its commits. The saved model `PPO_border_4280885` also stores its last 100 training episodes, so we ran the model in an export of that branch's code with different plume sizes and compared the episodes.

| Plume size in the replay | Episode length: median (middle 50 %) | KS test p: length / return |
|---|---|---|
| Stored in the model | 22 (20–24) | |
| Fixed σ 3.5 | 22 (20–24) | 0.82 / 0.39 |
| σ 3–4 | 22 (20–25) | 0.49 / 0.52 |
| σ 2.5–4.5 | 23 (20–27) | 0.03 / 0.003 |
| σ 2–5 | 24 (20–34) | below 0.0001 / below 0.0001 |

So this model ended its training on a plume size close to 3.5. The test cannot rule out a narrow range, or a different setting earlier in the training. We should confirm with Erik, who remembers training on a variable size, which job that was.

The size matters little in practice. With σ from 2.5 to 5 the model ends at least 99.7 % of the episodes with 90–93 % of the true border visited. Only σ 2 is weaker: 92 % ended and 107 steps on average, probably because the start 6–10 cells from the source is outside such a small plume.

## Next steps

1. Agree with the supervisor which goal we report on, or whether we report both.
2. Confirm with Erik which job trained on a variable plume size.
3. Find out why the plume and border agents get stuck inside a survey, by logging what they see and do, with no training.
4. Fix that, and retrain only if it is needed. These agents train on one plume only: they map the current plume, and the meta agent handles the search for more plumes.
5. Change the meta reward and meta state, and train the meta agent on a random 1–3 sources, with 3 seeds per variant.
6. Later: a learned space agent with wide tracks, a stop action, and the λ and gamma runs.

## How to reproduce the results

Run from the repository root (on Windows: `.venv/Scripts/python`).

```bash
# lawnmower, sweep 2, visit goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 2 --sources 1 3

# lawnmower, sweep 3, estimate goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 3 --sources 1 3 --goal estimate --goal-mean-std 0.26 --known-std 0.1

# lawnmower, sweep 4, which stops at the end of its path
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline lawnmower --spacing 4 --sources 1 3 --goal estimate --goal-mean-std 0.115 --known-std 0.5

# rule with sweep 6, estimate goal with the check for unknown areas
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 6 --sources 1 3 --goal estimate --goal-mean-std 0.26 --known-std 0.1

# rule with sweep 4 and scripted gas, visit goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 4 --scripted-gas --sources 1 3

# rule with sweep 6 and scripted gas, estimate goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 6 --scripted-gas --sources 1 3 --goal estimate --goal-mean-std 0.26 --known-std 0.1

# a trained agent with sampled actions
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --model ../trained-agents/HRL_PPO_minimal_values_4371280/HRL_PPO_scratch.zip --mode 4 --sources 1 3
```

Use `--sources 1 1` for one source.
