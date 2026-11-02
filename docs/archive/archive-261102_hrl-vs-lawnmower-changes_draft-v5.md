# Making the HRL survey better than a lawnmower: the changes so far

**Date:** 2026-11-02. **Branch:** `martin`. **Status:** implemented and tested in simulation, and the speed is measured on Fox. The meta agent has only had short training runs on a PC (1M decisions, one seed). The full training on Fox has not started. The plume, border and space agents have not been retrained.

## Why we made these changes

A fixed lawnmower with tracks 2 cells apart reaches today's end goal in every episode, in 366 AUV steps. Our best trained meta agent reaches it in 89 % of the episodes, in 345 AUV steps on average. So we have not yet shown that HRL gives more than a fixed path.

The idea to test: sweep the map with wide tracks and spend extra steps only where gas has been found. A lawnmower cannot do this, because it follows the same path whatever it measures. No agent is told how many plumes there are.

Changes 1–8 test the idea without training anything. Changes 9–12 prepare the meta agent to learn it.

| Term | Meaning |
|---|---|
| AUV step | One move of the AUV. |
| Decision | One choice by the meta agent. It covers 3 AUV steps in these tests. |
| GP map | The plume and border cells as the GP estimates them from the measurements. |
| GP std | How uncertain the GP is about a cell. "Mean GP std" is the average over the map. |
| Map overlap | Cells in both the GP map and the true map, divided by cells in either. 1.0 means identical. |
| Plume | A connected region of true cells above the threshold. Sources that overlap count as one plume. |
| Sweep *N* | A lawnmower with tracks *N* cells apart. |
| Handover | A scripted path brings the AUV to gas, and the trained agent takes over when it can see the gas (change 8). |
| Sampled / deterministic | Sampled: the meta agent draws its choice from its probabilities. Deterministic: it always takes the most likely choice. |
| σ | The width of a plume, in cells. The meta agents train on σ from 2 to 5. |

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

## Change 8: A scripted path to the gas (handover)

**Files:** [options.py](../HUGIN_gym/agents/options.py), [gas_cells.py](../HUGIN_gym/envs/core/gas_cells.py), [log_gas_agents.py](../HUGIN_gym/evaluation/log_gas_agents.py), [test_HRL(ppo).py](<../HUGIN_gym/evaluation/test_HRL(ppo).py>)

**How it was:** the plume and border agents controlled the AUV for the whole option, whatever they could see.

```python
action, _ = self.policy.predict(filtered_state, deterministic=True)
```

**What that meant:** these agents see a 7 × 7 patch around the AUV. We logged what they see and do inside a survey (rule + sweep 4 + trained agents, 1–3 sources). About 57 % of their decisions start with no target in that patch. How often they finish a target cell depends on how far away it is:

| Nearest target cell | Plume agent finishes a cell | Border agent finishes a cell |
|---|---|---|
| Within 3 cells | 91.7 % | 87.2 % |
| 3–6 cells | 37.8 % | 33.2 % |
| 6–10 cells | 1.3 % | 2.0 % |
| More than 10 cells | 0.1 % | 0.4 % |

So they do their job when they can see the target, but they cannot find a target far away. This answers the open question from the last version of this document. A smaller effect is also visible: with a target in view, success falls as more of the map is visited (plume from 95 % to 73 %, border from 91 % to 67 %).

**How it is now:** `--handover` wraps each trained agent. If an unfinished cell of its own map is in the patch, the trained agent decides. If not, a shortest path moves the AUV to the nearest such cell.

```python
if not target.any() or (target & patch_mask(target.shape, ix, iy, base.patch_radius)).any():
    return self.trained.act(obs)
return self._first_action_to(*self._pose(), target)
```

**What that means:** nothing is retrained, and the plume and border agents still only care about the current plume. The search for more plumes stays with the meta agent. It is a scripted part of the system: on the estimate goal, 18 % of the steps of these agents follow the scripted path (13.6 of 75.9 per episode), and the trained agents decide the rest.

**What we hope to see:** the trained agents work inside a survey, without retraining, so that a choice by the meta agent is no longer wasted on an agent that cannot see anything.

**First result:** with 1–3 sources:

| Survey | Goal | Ended by goal | AUV steps | Overlap plume / border | Plumes found |
|---|---|---|---|---|---|
| Rule + sweep 4 + trained agents | Visit | 75 % | 369 | 0.987 / 0.976 | 99.1 % |
| **Rule + sweep 4 + handover** | **Visit** | **99.1 %** | **304** | **0.998 / 0.994** | **100 %** |
| Rule + sweep 4 + scripted gas | Visit | 99.9 % | 277 | 0.998 / 0.992 | 100 % |
| Rule + sweep 6 + trained agents | Estimate | 90 % | 291 | 0.995 / 0.984 | 99.7 % |
| **Rule + sweep 6 + handover** | **Estimate** | **99.9 %** | **231** | **0.998 / 0.990** | **100 %** |
| Rule + sweep 6 + scripted gas | Estimate | 100 % | 226 | 0.997 / 0.989 | 100 % |

Both surveys with handover now use 18 % fewer steps than the lawnmower, with trained agents doing the gas mapping. The handover alone does not fix the learned space agent: the rule with that agent reaches 93.7 % in 337 steps on the visit goal, and `minimal_values` reaches 89 % in 363.

## Change 9: A meta reward that follows the goal

**Files:** [HRL_env.py](../HUGIN_gym/envs/HRL_env.py)

**How it was:** the reward paid for newly visited cells. The plume agent was paid for new border cells.

```python
elif option_idx == 1:  # plume agent
    if (current_c_around_sum - old_c_around_sum) > 0.0:  # c_around: the border cells
        reward = (current_c_around_sum - old_c_around_sum) / 6 + new_visited / 6
```

**What that meant:**

- The plume agent earned no more for a new plume cell than for any other new cell.
- On the estimate goal, a cell that the GP knows well earns nothing unless the AUV visits it, so the reward and the goal point in different directions.
- The check that gas is waiting used visited cells too, so on the estimate goal the sweep could stay unpaid after the gas was already known.
- The −2 penalty for no progress only came after 3 idle steps in a row.

**How it is now:** `reward_mode` chooses the reward, and `"cells"` keeps the old one. All modes keep the cost of −1 per AUV step.

```python
# "cells_fixed": the same sizes, but it counts the cells as the end goal counts them
case 1:  # plume agent: pays for plume cells
    reward = (new_plume + new_visited) / 6 if new_plume > 0 else new_visited / 12
```

```python
# "map_gain": how much better the map became
gain = MAP_GAIN_WEIGHT * (score - last_score) + STD_GAIN_WEIGHT * (last_mean_std - mean_std)
```

Every idle AUV step now costs 2/3, so 3 idle steps still cost 2.

**What that means:** `cells_fixed` repairs the old idea. `map_gain` is closer to the supervisor's suggestion of a reward based on relative change: it pays for how much the map improves, so it does not use the bonus per plume from change 10. Its score uses the true map, but only for the reward during training. The meta agent never sees the true map.

**What we hope to see:** the meta agent learns to leave the sweep when gas is waiting and to return when it is done, with less of its time spent on options that make no progress.

**First result:** see change 12. The fixes were trained together, so we can compare `cells_fixed` with `map_gain`, but we have not measured `cells_fixed` alone against the old reward.

## Change 10: A bonus per plume

**Files:** [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** the bonus of 40 was paid when the shares over all gas found passed 90 %, and it could be paid again after they fell below 90 %.

```python
if self.once_per_episode == False and (plume_share<self.accuracy_goals[1] or border_share< self.accuracy_goals[2]):
    self.once_per_episode = True # allowing to be rewarded for 2 or more plumes
```

**What that meant:** the bonus is tied to global shares, not to plumes. A second plume found before the first is finished adds no bonus of its own, and shares that move around 90 % can pay again for the same gas.

**How it is now:** with `bonus_mode="patch"`, the bonus is paid once for each connected patch of the GP map with at least 20 cells, when that patch reaches the goal.

```python
if n_cells >= self.MIN_BONUS_PATCH_CELLS and np.sum(patch & self.paid_gas) < n_cells / 2 and np.sum(patch & done) / n_cells >= self.accuracy_goals[1]:
    self.paid_gas |= patch
    return True
```

**What that means:** finishing a plume always pays, and more plumes pay more, without the agent being told how many there are. A patch that has mostly been paid counts as the same patch, even if it grows or shrinks. The limit of 20 cells is below the smallest plume we measured (23 cells in 5,000 maps).

**What we hope to see:** a clear reason to search for another plume after finishing one.

**First result:** not tested alone (see change 12).

## Change 11: A meta state about the gas

**Files:** [HRL_env.py](../HUGIN_gym/envs/HRL_env.py), [GPWrapper.py](../HUGIN_gym/envs/wrappers/GPWrapper.py)

**How it was:** `minimal_values` gave the meta agent 6 numbers and the values of the 3 sub-agents.

```python
keys = ("time_used", "space_coverage", "plume_coverage", "border_coverage", "plume_found", "gp_uncertainty")
```

**What that meant:** one global coverage number can hide a plume that is hardly mapped. The state also did not show whether a large unknown area was left, how far away the unfinished gas was, or whether the last choice had made progress. The rule uses that last signal, so the meta agent had less to work with than the rule.

**How it is now:** the meta state `gas` adds these, all from the GP estimate and the survey so far, never from the true map.

| Number | What it tells the meta agent |
|---|---|
| `plume_share`, `border_share` | How much of the gas found is done, as the end goal counts it |
| `lowest_patch_share` | How far the least finished patch has come, so a second plume is not hidden |
| `nearest_unfinished` | How far away the nearest unfinished cell is |
| `unknown_share` | The largest unknown area, which the end goal requires to be small |
| `last_idle`, `last_option` | The last choice, and whether it made no progress |
| `time_used`, `space_coverage`, `plume_found`, `gp_uncertainty` | As before |

**What that means:** the size of the state does not depend on the number of plumes, so one agent can handle any number of plumes. This is our proposed answer to the supervisor's question of whether the representation must change: add numbers about the least finished patch, and keep the size fixed. `gas_no_time` is the same state without the time.

**What we hope to see:** the meta agent gets at least what the rule uses, so it should be able to match the rule.

**First result:** not tested alone (see change 12).

## Change 12: Meta agent training on a random 1–3 sources

**Files:** [train_HRL(ppo).py](<../HUGIN_gym/training/train_HRL(ppo).py>), [fox_train_HRL.sh](../fox_train_HRL.sh), [fox_train_HRL_matrix.sh](../fox_train_HRL_matrix.sh), [io.py](../HUGIN_gym/utils/io.py)

**How it was:** a variant set the option lengths and the meta state. Runs had no set seed, and always trained on the same setup.

```python
"minimal_values": {"option_lengths": [3], "smdp_discount": False, "mean_option_reward": False, "decision_cost": 0.0, "meta_state": "minimal_values"},
```

**What that meant:** training with a sweep, the estimate goal or several plumes needed edits in the script, and we could not repeat a run to see how much of a difference comes from chance.

**How it is now:** a variant also sets the survey. `--seed` and `--max-steps` set the seed and the length of the training. The evaluation reads the survey from the saved training config, so a model is always tested with its own options and goal.

```python
SWEEP6_ESTIMATE = {"spacing": 6, "handover": True, "goal_mode": "estimate", "goal_mean_std": 0.26, "known_std": 0.1, "bonus_mode": "patch", "sources": [1, 3]}
"sweep6_est": {**THREE_STEPS, "meta_state": "gas", "reward_mode": "cells_fixed", "survey": SWEEP6_ESTIMATE},
```

`bash fox_train_HRL_matrix.sh 5000000` submits six variants with 3 seeds each: 18 jobs of about 2.4 h.

| Variant | What it tests |
|---|---|
| `sweep6_est` | The main test: can it beat the rule on the estimate goal? |
| `sweep4_visit` | The same on the visit goal |
| `sweep6_est_map` | Is a reward for the map better than a reward per cell? |
| `sweep6_est_no_time` | Does the agent need to know the time used? |
| `space_est` | The learned space agent instead of the scripted sweep: how much must be scripted? |
| `sweep6_est_one` | Training on one source only: does training on several plumes matter? |

**What that means:** only the meta agent trains. The sub-agents keep their current models and focus on the current plume, and the meta agent is the one that must find out whether there are more plumes.

**What we hope to see:** a trained meta agent that beats the lawnmower and then the rule, with little spread between the seeds, and that holds up on plume sizes it did not train on.

**First result:** we trained on a PC for 1M decisions (10 minutes) with seed 1, to check that learning works before using Fox. The control has the survey of `sweep6_est`, but the reward, bonus and meta state of run 5. It is not one of the Fox variants. Results are 1,000 episodes with 1–3 sources on the estimate goal:

| Meta agent | Actions | Ended by goal | AUV steps | Overlap plume / border | Plumes found |
|---|---|---|---|---|---|
| Control | Sampled | 99.7 % | 274 | 0.997 / 0.990 | 100 % |
| Control | Deterministic | 55 % | 363 | 0.947 / 0.922 | 92.9 % |
| `sweep6_est_map` | Sampled | 100 % | 252 | 0.996 / 0.986 | 100 % |
| `sweep6_est_map` | Deterministic | 80 % | 294 | 0.976 / 0.947 | 99.7 % |
| **`sweep6_est`** | **Sampled** | **99.8 %** | **235** | **0.998 / 0.991** | **100 %** |
| `sweep6_est` | Deterministic | 93 % | 252 | 0.976 / 0.964 | 98.1 % |
| Rule + sweep 6 + handover, for comparison | | 99.9 % | 231 | 0.998 / 0.990 | 100 % |
| Lawnmower, sweep 3, for comparison | | 100 % | 282 | 0.994 / 0.984 | 100 % |

Training also shows the fixes helping early: at 250,000 decisions, the survey took 250 steps with `sweep6_est`, 312 with `sweep6_est_map` and 357 with the control. `sweep6_est` then flattens at about 240 steps from 500,000 decisions. Its critic is alive, but weak (explained variance 0.29). These are single seeds, and the control differs in three things at once, so the differences are an early sign, not a result.

## Results

1,000 episodes per row, seed 0, unless noted. Trained agents use sampled actions. "Estimate" uses a mean GP std limit of 0.26, known cells at GP std 0.1 or lower, and the check from change 6. "Scripted gas" is change 7, "handover" is change 8. "Trained meta" is `sweep6_est` after 1M decisions on a PC, with one seed, so it is an early check.

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
| Rule + sweep 4 + handover | Visit | 99.6 % | 269 | 0.999 / 0.994 | 97.7 / 96.8 % | 100 % |
| Rule + sweep 6 + handover | Estimate | 100 % | 197 | 0.998 / 0.990 | 94.1 / 95.0 % | 100 % |
| Trained meta + sweep 6 + handover | Estimate | 99.8 % | 200 | 0.997 / 0.989 | 94.7 / 95.2 % | 100 % |
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
| `minimal_values` + handover | Visit | 89 % | 363 | 0.994 / 0.987 | 96.7 / 94.9 % | 99.3 % |
| Rule + space agent + handover | Visit | 94 % | 337 | 0.991 / 0.984 | 94.9 / 94.8 % | 99.1 % |
| Rule + sweep 4 + handover | Visit | 99.1 % | 304 | 0.998 / 0.994 | 96.9 / 96.0 % | 100 % |
| Rule + sweep 6 + handover | Estimate | 99.9 % | 231 | 0.998 / 0.990 | 93.3 / 94.0 % | 100 % |
| Trained meta + sweep 6 + handover | Estimate | 99.8 % | 235 | 0.998 / 0.991 | 93.6 / 94.3 % | 100 % |
| **Rule + sweep 6 + scripted gas** | **Estimate** | **100 %** | **226** | **0.997 / 0.989** | 92.9 / 93.1 % | **100 %** |

**Does the survey adapt to the number of plumes?**

| Survey | AUV steps with 1 / 2 / 3 plumes | Plumes found |
|---|---|---|
| Lawnmower, sweep 2 (visit) | 368 / 373 / 372 | 100 / 99.9 / 100 % |
| Lawnmower, sweep 3 (estimate) | 283 / 281 / 280 | 100 / 100 / 100 % |
| `minimal_values` (visit) | 356 / 389 / 404 | 99.8 / 99.2 / 96.3 % |
| Rule + sweep 6 (estimate) | 272 / 305 / 326 | 100 / 99.7 / 99.1 % |
| Rule + sweep 4 + scripted gas (visit) | 267 / 286 / 296 | 100 / 100 / 100 % |
| Rule + sweep 4 + handover (visit) | 287 / 319 / 330 | 100 / 100 / 100 % |
| Rule + sweep 6 + handover (estimate) | 214 / 244 / 258 | 100 / 100 / 100 % |
| Trained meta + sweep 6 + handover (estimate) | 215 / 250 / 274 | 100 / 100 / 100 % |
| Rule + sweep 6 + scripted gas (estimate) | 209 / 240 / 257 | 100 / 100 / 100 % |

**Does it hold up on plume sizes it did not train on?** 500 episodes per row, 1–3 sources, estimate goal for all but sweep 2 and `minimal_values`:

| Plume size | Survey | Ended by goal | AUV steps | Overlap plume / border | Plumes found |
|---|---|---|---|---|---|
| Small, σ 1–2 | Lawnmower, sweep 2 (visit) | 100 % | 369 | 0.882 / 0.777 | 96.7 % |
| Small, σ 1–2 | Lawnmower, sweep 3 | 98.8 % | 286 | 0.784 / 0.667 | 95.2 % |
| Small, σ 1–2 | `minimal_values` (visit) | 52 % | 426 | 0.767 / 0.636 | 94.8 % |
| Small, σ 1–2 | Rule + sweep 6 + handover | 99.2 % | 276 | 0.845 / 0.741 | 92.1 % |
| Small, σ 1–2 | Trained meta + sweep 6 + handover | 97.6 % | 288 | 0.836 / 0.735 | 92.6 % |
| Large, σ 5–8 | Lawnmower, sweep 3 | 100 % | 286 | 0.998 / 0.992 | 100 % |
| Large, σ 5–8 | `minimal_values` (visit) | 41 % | 446 | 0.983 / 0.966 | 99.5 % |
| Large, σ 5–8 | Rule + sweep 6 + handover | 93.8 % | 339 | 1.000 / 0.997 | 100 % |
| Large, σ 5–8 | Trained meta + sweep 6 + handover | 98.4 % | 328 | 0.999 / 0.997 | 100 % |

## What this shows

- **With handover, a survey with trained gas agents beats the lawnmower on both goals.** With 1–3 sources, the rule uses 304 steps on the visit goal against 370, and 231 on the estimate goal against 282. Its map is at least as good and it finds all plumes. On the visit goal, it visits 96.9 % of the true plume cells, just below the 98 % we asked for.
- **The trained gas agents were held back by their short view more than by their skill.** With handover they come close to the scripted gas option on the estimate goal (231 steps against 226). On the visit goal they get part of the way (304 steps, against 369 without handover and 277 for scripted gas).
- **A trained meta agent passes the estimate target, but has not beaten the rule.** After 1M decisions it uses 235 steps against 231 for the rule and 282 for the lawnmower. There is little room left: the scripted gas option reaches 226. So this simple setup may not show much value from a learned meta agent.
- **A learned meta agent may be most useful where fixed rules break.** On large plumes it ends more episodes than the rule in fewer steps (98.4 % in 328, against 93.8 % in 339). On small plumes the rule is slightly better (99.2 % in 276, against 97.6 % in 288). Both use one seed and need the Fox runs.
- **A fixed lawnmower still wins some cases.** On large plumes, sweep 3 takes 286 steps with all episodes ended, because most of the map is gas. On small plumes, wide tracks miss more plumes (92 % found) than sweep 3 (95 %) and sweep 2 (97 %), even though their maps overlap the true map better than sweep 3.
- **Deterministic choices are much worse than sampled ones.** All three models from the PC check do worse with deterministic actions (55–93 % ended, against more than 99 % with sampled actions). A likely reason is that the same state gives the same choice, so the agent can get stuck in a loop, but we have not checked this. We must report both, and we should prefer an agent that does not need chance to get unstuck.

**What is learned and what is scripted in the best learned survey so far:**

| Part | Status |
|---|---|
| When to sweep, map the plume or map the border | Learned (the meta agent) |
| The moves near gas | Learned (the plume and border agents, about 82 % of their steps) |
| The way to gas that they cannot see | Scripted (change 8) |
| The search for gas | Scripted (the sweep), and `space_est` tests the learned space agent instead |
| Track spacing and the limits of the end goal | Chosen by hand |

**Targets for a learned agent, 1–3 sources:**

| Goal | Lawnmower to beat | The agent needs | Best so far with trained agents |
|---|---|---|---|
| Visit | Sweep 2: 370 steps; scripted gas reaches 277 | Fewer steps, at least 99 % ended, at least 98 / 96 % visited, no more plumes missed | Rule + handover: 304 steps, 99.1 %, 96.9 / 96.0 % visited, all plumes found. No learned meta agent has been trained on this survey yet. |
| Estimate | Sweep 3: 282 steps; scripted gas reaches 226 | Fewer steps, at least 99 % ended, overlap of at least 0.994 / 0.984, all plumes found | Trained meta: 235 steps, 99.8 %, 0.998 / 0.991, all plumes found. Rule: 231 steps. |

Sweep 2 stays the benchmark on the visit goal because sweep 3 leaves gaps between its tracks and reaches that goal in only 6 % of the episodes.

**Limits of these numbers:** the survey results come from one evaluation seed on one PC. The trained meta agents have one training seed and 1M decisions, and the plume size test has 500 episodes. The rule, the sweep and the handover do not learn, and the limits of the estimate goal and the track spacing were chosen by hand. All plumes are simple Gaussian shapes, so we do not know how any of this works on more realistic plumes.

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
3. Run the 18 Fox jobs from change 12, about 2.4 h each.
4. Evaluate every seed on the same 1,000 episodes, sampled and deterministic, and on σ 1–2 and σ 5–8. The test is whether a variant beats the lawnmower and the rule across all 3 seeds. If it cannot beat the rule, that is an honest result to report.
5. Find out why deterministic choices do worse, and train the best variant for longer only if its curve has not flattened.
6. Later: replace the scripted path with a learned one, a learned space agent with wide tracks, a stop action, and the λ and gamma runs. Retrain the plume and border agents only if we need them to travel. They should still train on one plume.

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

# rule with sweep 4 and handover, visit goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 4 --handover --sources 1 3

# rule with sweep 6 and handover, estimate goal
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --baseline rule --spacing 6 --handover --sources 1 3 --goal estimate --goal-mean-std 0.26 --known-std 0.1

# what the plume and border agents see and do inside a survey (the table in change 8)
PYTHONPATH=. .venv/bin/python -m HUGIN_gym.evaluation.log_gas_agents --baseline rule --spacing 4 --sources 1 3 --episodes 300 --output ../gas_agents.csv

# plot the true map against the estimated map after an evaluation
PYTHONPATH=. .venv/bin/python -m HUGIN_gym.evaluation.compare_maps --baseline rule --spacing 6 --handover --sources 1 3 --goal estimate --goal-mean-std 0.26 --known-std 0.1 --output ../compare_maps

# train the meta agent: one variant, seed and length on a PC
PYTHONPATH=. .venv/bin/python "HUGIN_gym/training/train_HRL(ppo).py" --preset sweep6_est --seed 1 --max-steps 1000000

# train the meta agent: all variants and seeds on Fox, from a login node
bash fox_train_HRL_matrix.sh 5000000

# a trained meta agent: it gets its survey from its saved training config
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --model ../trained-agents/<run name>/HRL_PPO_scratch.zip --mode 4

# a trained agent with sampled actions
PYTHONPATH=. .venv/bin/python "HUGIN_gym/evaluation/test_HRL(ppo).py" --model ../trained-agents/HRL_PPO_minimal_values_4371280/HRL_PPO_scratch.zip --mode 4 --sources 1 3
```

Use `--sources 1 1` for one source. `--mode 4` gives sampled actions and `--mode 1` deterministic ones.
