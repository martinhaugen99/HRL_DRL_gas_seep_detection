import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import HUGIN_gym  # This import will automatically register the environment


from HUGIN_gym.agents.options import Option, survey_options
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.agents.term_fn.termination_fn_agent1 import termination_fn1
from HUGIN_gym.agents.term_fn.termination_fn_agent2 import termination_fn2
import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF,WhiteKernel, ConstantKernel as C
from HUGIN_gym.envs.core.visualisation.MapVisualiser import MapVisualiser, visualise_state_space,visualise_state_space_hrl 
from HUGIN_gym.envs.core.visualisation.visualise_heatmap import HeatmapVisualiser, CNN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser
from HUGIN_gym.utils.io import hrl_survey_from_config, hrl_variant_from_config, load_training_config_from_model
from HUGIN_gym.evaluation.rollout import run_episode_hrl, manual_rollout_hrl,run_episodes_hrl_vec
from HUGIN_gym.evaluation.baselines import BASELINES, make_baseline
from HUGIN_gym.evaluation.survey_metrics import map_metrics, summarise_by_plumes

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.network = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             #nn.Flatten()
# )

# =========================================================
# ENV CREATION
# =========================================================

def make_test_env(GP=True, VIS_GP=True, VISUALISE = True,kernel_config=None, MULTIPLE_GAUSSIANS=None, MAX_OPT_LENGTH =None,ACCURACY_GOALS=None,
                  SURVEY_STEPS=None, VARIANT=None, SPACING=None, GOAL="visit", GOAL_MEAN_STD=0.1, KNOWN_STD=0.1, UNKNOWN_STD=0.8, MAX_UNKNOWN_CELLS=10, SCRIPTED_GAS=False, HANDOVER=False, BONUS_MODE="coverage", SIGMA_RANGE=None):
    """
    SURVEY_STEPS: survey budget in AUV steps, which ends the episode (None: no budget)
    VARIANT: the meta agent variant, i.e. the HierarchicalEnv settings of a preset in train_HRL(ppo).py (None: one option length, MAX_OPT_LENGTH)
    SPACING: replace the space sub-agent with a scripted lawnmower option, tracks this many cells apart (None: keep it)
    SCRIPTED_GAS: replace the plume and border sub-agents with GasFillOption, as a benchmark for them
    HANDOVER: give the plume and border sub-agents a scripted way to their target (HandoverOption)
    GOAL, GOAL_MEAN_STD, KNOWN_STD, UNKNOWN_STD, MAX_UNKNOWN_CELLS, BONUS_MODE: the meta agent's end goal and bonus, see the same settings in lower case of GPWrapper
    SIGMA_RANGE: smallest and largest plume width, in cells (None: the range of the HRL training, [2.0, 5.0])
    """

    base_env = gym.make(
        "HUGIN-v0",
        render_mode="human" if VISUALISE or VIS_GP else None,
        max_episode_steps=4
    )
    base_env = base_env.unwrapped
    base_env.train = True
    base_env.GP_ON = GP
    base_env.random_points = True
    base_env.agent_type ="META"
    base_env.HRL = True # if False we can get the base rewards and overwrite them, but still get the 40/-4 filtering
    base_env.multiple_gaussians = [1,1] if MULTIPLE_GAUSSIANS is None else MULTIPLE_GAUSSIANS # default = [1,1] , it defines the range, e.g. [1,5] would cause from 1 up to 5 gaussian sources within the volume at the same time
    base_env.spawn_at_survey_start = True  # start in a corner like a survey, as in the HRL training
    base_env.vary_plume_size = True  # sigma from [2.0, 5.0], as in the HRL training, unless SIGMA_RANGE says otherwise
    if SIGMA_RANGE is not None:
        base_env.gaussian_sigma_range = list(SIGMA_RANGE)

     # ---- GP WRAPPER HERE ----
    if GP:
        if kernel_config is None:
            kernel_config = {
            "RBF": {"type": "RBF", "length_scale": 3.5},
            #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}
            }
        else:
            kernel_config=kernel_config
        gp_env = GPWrapper(
            base_env,
            kernels_config=kernel_config,
            gp_visualise=VIS_GP,
            HRL=True,
            SUB_AGENT_TRAIN_ON_GP=False,
            goal_mode=GOAL,
            goal_mean_std=GOAL_MEAN_STD,
            known_std=KNOWN_STD,
            unknown_std=UNKNOWN_STD,
            max_unknown_cells=MAX_UNKNOWN_CELLS,
            bonus_mode=BONUS_MODE,
        )
        gp_env.accuracy_goals=ACCURACY_GOALS
        gp_vis = GPVisualiser(
            fig_num=33,
            shift=base_env.unwrapped.offset[0], #careful! assuming dim_len X=Y
            kernel_config=kernel_config
        ) if VIS_GP else None
    else:
        gp_vis = None

    # Load LOW-LEVEL policies
    policy1 = PPO.load("../trained-agents/PPO_space_30M/PPO_scratch.zip")
    policy2 = PPO.load("../trained-agents/PPO_compare_100M_4259607/PPO_scratch.zip") #PPO_plume_GP_uncertainty_0p1_and_full_plume # PPO_plume_GP_clipped_RWD_no_distance_to_max
    policy3 = PPO.load("../trained-agents/PPO_border_4280885/PPO_scratch.zip") #PPO_border_GP_uncertainty_0p1_and_full_plume#PPO_border_GP_not_clipped_RWD_no_distance_to_max

    # note that the termination function dont act: return == FALSE
    option1 = Option(policy1, termination_fn1, "spatial exploration", obs_keys=["obs_state","visited_maps_downsampled","space_coverage"])
    option2 = Option(policy2, termination_fn2, "gas plume exploration", obs_keys=["obs_state", "local_GT","plume_coverage","space_coverage","distance_from_max"])#,"visited_maps_downsampled", "space_coverage","GT_c_over_threshold_maps_downsampled"
    option3 = Option(policy3, termination_fn2, "gas border exploration", obs_keys=["obs_state","local_GT_around", "border_coverage","space_coverage","distance_from_max"])# ,"visited_maps_downsampled", "GT_c_around_threshold_maps_downsampled"

    gp_env = gp_env if GP else base_env
    options = [option1, option2, option3]
    if SPACING is not None or HANDOVER or SCRIPTED_GAS:
        if not GP:
            raise ValueError("The scripted options read the GP map, so they need the GP")
        options = survey_options(options, base_env, gp_env, SPACING, HANDOVER, SCRIPTED_GAS)

    env = HierarchicalEnv(
        base_env=gp_env,
        options=options,
        gamma=0.998, # per AUV step, as in the HRL training. Only changes the reward, not the agent's choices
        MAX_OPT_LENGTH=MAX_OPT_LENGTH,
        ACCURACY_GOALS=ACCURACY_GOALS,GP=GP, # accuracy goals work only for no GP
        max_primitive_steps=SURVEY_STEPS,
        **(VARIANT or {}),
    )

    return env,gp_vis

# =========================================================
# TEST AGENT
# =========================================================
DEFAULT_MODEL = Path("../trained-agents/HRL_PPO_untuned_4363368/HRL_PPO_scratch.zip") # run 4, when --model is not given

# The untuned run (run 4) started before the training config saved HRL_VARIANT, so its settings are given here
# (PRESETS["untuned"] in train_HRL(ppo).py). The later runs read theirs from HRL_VARIANT in training_config.json.
UNTUNED_VARIANT = {"option_lengths": [3], "smdp_discount": False, "mean_option_reward": False, "decision_cost": 0.0, "meta_state": "full"}
UNTUNED_SURVEY_STEPS = 480 # survey budget in AUV steps: 160 decisions of 3 steps
OPTION_NAMES = ["space", "plume", "border"] # in the order of the options in make_test_env, as in the TensorBoard tags of the training
OPTION_COLORS = [0x00FF00, 0xFF3399, 0xFFD700] # label and trail color per option in the Meshcat view: green, pink, yellow
STEP_DELAY = 0.1 # seconds between AUV steps in the Meshcat view (0 = full speed)
WATCH_EPISODES = 5 # episodes in the Meshcat view (mode 5)
MAX_OPT_LENGTH = 3 # AUV steps per decision of the variants with one option length
SEED = 0 # the env draws the plumes and start positions from np.random at reset, so every evaluation gets the same episodes

@dataclass
class Survey:
    """The env and the meta agent of an evaluation."""

    env: HierarchicalEnv
    model: object  # a PPO model or a ScriptedMeta: both have predict
    gp_vis: object
    variant: dict
    sources: list[int]
    max_eps_len: int
    settings: dict  # spacing, goal, goal_mean_std, known_std, unknown_std, max_unknown_cells: what this survey uses


def make_survey(model_path: Path, visualise: bool = False, baseline: str | None = None, spacing: int | None = None, sources: list[int] | None = None,
                goal: str | None = None, goal_mean_std: float | None = None, known_std: float | None = None, unknown_std: float = 0.8, max_unknown_cells: int = 10,
                scripted_gas: bool = False, handover: bool = False, sigma_range: list[float] | None = None) -> Survey:
    """
    Build the env and the meta agent of an evaluation.
    model_path: the meta agent model; training_config.json is read from the same folder (not used with a baseline)
    visualise: show the episodes in the Meshcat browser view
    baseline: a meta agent without learning instead of the model, see baselines.py. It uses the untuned variant.
    spacing: replace the space sub-agent with a scripted lawnmower option, tracks this many cells apart
    scripted_gas: replace the plume and border sub-agents with GasFillOption, as a benchmark for them
    handover: give the plume and border sub-agents a scripted way to their target
    sources: the smallest and largest number of gas sources per episode (None: as in the model's training)
    goal, goal_mean_std, known_std, unknown_std, max_unknown_cells: the meta agent's end goal, see GPWrapper (goal is its goal_mode)
    sigma_range: smallest and largest plume width, in cells (None: as in the HRL training)
    A setting left out comes from the survey the model trained in, so a model is evaluated with its own options and goal.
    """
    VISUALISE = visualise
    GP = True
    VIS_GP = False

    # Load config from the same folder
    config = {} if baseline else load_training_config_from_model(model_path)
    kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    MULTIPLE_GAUSSIANS = sources or config.get("HRL_HYPERPARAMETERS", {}).get("MULTIPLE_GAUSSIANS") or [1,1]
    print(MULTIPLE_GAUSSIANS)
    ACCURACY_GOALS = [0.9,0.9,0.9]
    VARIANT, SURVEY_STEPS = hrl_variant_from_config(config, UNTUNED_VARIANT, UNTUNED_SURVEY_STEPS)
    trained_survey = hrl_survey_from_config(config)
    spacing = spacing if spacing is not None else trained_survey.get("spacing")
    handover = handover or trained_survey.get("handover", False)
    goal = goal or trained_survey.get("goal_mode", "visit")
    goal_mean_std = goal_mean_std if goal_mean_std is not None else trained_survey.get("goal_mean_std", 0.1)
    known_std = known_std if known_std is not None else trained_survey.get("known_std", 0.1)
    bonus_mode = trained_survey.get("bonus_mode", "coverage")
    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=kernel_config,MULTIPLE_GAUSSIANS = MULTIPLE_GAUSSIANS,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS,
                                SURVEY_STEPS=SURVEY_STEPS,VARIANT=VARIANT,SPACING=spacing,GOAL=goal,GOAL_MEAN_STD=goal_mean_std,KNOWN_STD=known_std,
                                UNKNOWN_STD=unknown_std,MAX_UNKNOWN_CELLS=max_unknown_cells,SCRIPTED_GAS=scripted_gas,HANDOVER=handover,BONUS_MODE=bonus_mode,SIGMA_RANGE=sigma_range) # add here a kernel_config from the config file !

    # Load HIGH-LEVEL model (the one you trained in train.py)
    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    model = make_baseline(baseline, env, SEED) if baseline else PPO.load(model_path)
    MAX_EPS_LEN = config["ENVIRONMENT"]["max_episode_length"] if config else SURVEY_STEPS // min(VARIANT["option_lengths"]) # e.g. 90
    settings = {"spacing": spacing, "handover": handover, "goal": goal, "goal_mean_std": goal_mean_std, "known_std": known_std,
                "unknown_std": unknown_std, "max_unknown_cells": max_unknown_cells}
    return Survey(env, model, gp_vis, VARIANT, MULTIPLE_GAUSSIANS, MAX_EPS_LEN, settings)


def test_agent(model_path: Path, deterministic: bool = True, episodes: int = 1000, visualise: bool = False, baseline: str | None = None,
               spacing: int | None = None, sources: list[int] | None = None, goal: str | None = None, goal_mean_std: float | None = None, known_std: float | None = None,
               unknown_std: float = 0.8, max_unknown_cells: int = 10, scripted_gas: bool = False, handover: bool = False,
               sigma_range: list[float] | None = None, results_json: Path | None = None):
    """
    Run the trained meta agent or a baseline for a number of episodes and print the mean +- std of the results.
    deterministic: the meta agent takes its most likely action (True) or samples it, as in training (False)
    visualise: slow the Meshcat view down by STEP_DELAY, with a label of the sub-agent in control and a pause after each episode
    results_json: also save the settings and every episode to this file, see save_results
    The other settings are those of make_survey.
    """
    VISUALISE = visualise
    GP = True
    VIS_STATE_SPACE = False
    survey = make_survey(model_path, visualise, baseline, spacing, sources, goal, goal_mean_std, known_std, unknown_std, max_unknown_cells, scripted_gas, handover, sigma_range)
    env, model, gp_vis, VARIANT, MULTIPLE_GAUSSIANS, MAX_EPS_LEN = survey.env, survey.model, survey.gp_vis, survey.variant, survey.sources, survey.max_eps_len
    cnn = CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space

    env.visualise_env = VISUALISE # allowing the simulator to update within an option
    env.step_delay = STEP_DELAY
    env.option_colors = OPTION_COLORS
    env.base_env.META_test = True # for testing. Records the tp,fp,np,fn and other info in info
    env.base_env.max_option_length = MAX_OPT_LENGTH
    env.base_env.option_length = 0
    env.base_env.unwrapped.max_eps_len = -1
    env.test_HRL=True # for counting the turns properly
    if VIS_STATE_SPACE:
        obs,_ = env.reset()
        vis_state_space = visualise_state_space(VIS_STATE_SPACE,obs,cnn,MapVisualiser,HeatmapVisualiser,env.base_env) 
    else:
        vis_state_space=[None,None,None,None]

    percentage_verify =[]
    average_turns = []
    average_steps = []
    average_space_coverage = []
    average_plume_coverage =[]
    average_border_coverage =[]
    actual_average_plume_coverage =[]
    actual_average_border_coverage =[]
    average_options_chosen =[]
    average_reward = []
    meta_actions = [] # every decision of every episode
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    RMSE_c=[]
    extracted_GP_imgs: list[np.ndarray] =[]
    map_rows: list[dict[str, float]] = [] # one per episode, see survey_metrics.py
    episode_rows: list[dict] = [] # one per episode, for results_json
    set_random_seed(SEED)
    for ep in tqdm(range(episodes), desc="episodes", disable=VISUALISE):

        if VISUALISE:
            print(f"\n=== Episode {ep + 1} ===")
        result = run_episode_hrl(env,model,MAX_EPS_LEN,vis_state_space,gp_vis,GP=GP,VISUALISE=VISUALISE,deterministic=deterministic,
                                 verbose=VISUALISE,pause_at_end=VISUALISE)
        average_reward.append(result["reward"])
        meta_actions.extend(result["actions"])
        average_space_coverage.append(result["space_coverage"])
        average_border_coverage.append(result["border_coverage"])
        average_plume_coverage.append(result["plume_coverage"])
        percentage_verify.append(result["terminated"]) # if the agent successfully terminated
        average_turns.append(result["episode_turns"])
        average_steps.append(result["base_env_steps"])
        average_options_chosen.append(result["options_chosen"])
        try:
            
            fp.append(result["info"]["false_positive_above"]) 
            tp.append(result["info"]["true_positive_above"]) 
            fn.append(result["info"]["false_negative_above"]) 
            tn.append(result["info"]["true_negative_above"]) 
            #extracted_GP_imgs.append(env.unwrapped.extract_GP_ERROR_map)
        except:
            pass#print(env.SPACE_test_counter,env.unwrapped.max_eps_len,env.unwrapped.SPACE_test)
        try:
            actual_average_border_coverage.append(100*(result["info"]["actual_current_around_threshold"]))
        except:
            pass
        try:
            actual_average_plume_coverage.append(100*(result["info"]["actual_current_above_threshold"]))
        except:
            pass
        diff = env.base_env.GP_pred.T-env.base_env.unwrapped.conc[:,:,0].T          # numpy array
        mse  = (diff ** 2).mean()
        rmse = np.sqrt(mse)

        RMSE_c.append(rmse)
        map_rows.append({
            **map_metrics(
                env.base_env.GP_pred,
                env.base_env.original_GT_c_over[:,:,0],
                env.base_env.original_GT_c_around[:,:,0],
                env.base_env.unwrapped.state["visited_map"][:,:,0,0],
                env.base_env.c_thresh,
                env.base_env.c_std,
            ),
            "ended_early": float(result["terminated"]),
            "steps": result["base_env_steps"],
        })
        episode_rows.append({
            **map_rows[-1],
            "turns": result["episode_turns"],
            "decisions": result["options_chosen"],
            "reward": result["reward"],
            "rmse": rmse,
            "plume_key": plume_key(env.base_env.unwrapped),
            "actions": result["actions"],
        })
    agent = f"baseline {baseline}" if baseline else f"model {model_path}"
    used = survey.settings
    spacing, goal, goal_mean_std, known_std, unknown_std, max_unknown_cells = used["spacing"], used["goal"], used["goal_mean_std"], used["known_std"], used["unknown_std"], used["max_unknown_cells"]
    print(f"{'DETERMINISTIC' if deterministic else 'SAMPLED'} actions, {episodes} episodes, seed {SEED}, {agent}, spacing {spacing}, handover {used['handover']}, scripted gas {scripted_gas}, sources {MULTIPLE_GAUSSIANS}, plume width {env.base_env.unwrapped.gaussian_sigma_range}, goal {goal}"
          + (f" (mean GP std <= {goal_mean_std}, known cells GP std <= {known_std}, unknown areas with GP std > {unknown_std} of at most {max_unknown_cells} cells)" if goal == "estimate" else ""))
    print(f"REWARD = {np.mean(np.array(average_reward)):.2f} +- {np.std(np.array(average_reward)):.2f}")
    option_names = ["sweep" if spacing else OPTION_NAMES[0], *OPTION_NAMES[1:]]
    action_names = HierarchicalEnv.action_names(option_names, VARIANT["option_lengths"])
    choice_counts = np.bincount(meta_actions, minlength=len(action_names))
    print("CHOICES = " + ", ".join(f"{name} {count / choice_counts.sum():.2f}" for name, count in zip(action_names, choice_counts)))
    print(
        f"RMSE = {np.mean(np.array(RMSE_c)):.5f} "
        f"+- {np.std(np.array(RMSE_c)):.5f}"
    )
    print(f"Options_CHOSEN = {np.mean(np.array(average_options_chosen)):.2f} +- {np.std(np.array(average_options_chosen)):.2f}")
    print(f"STEPS = {np.mean(np.array(average_steps)):.2f} +- {np.std(np.array(average_steps)):.2f}")
    print(f"TURNS= {np.mean(np.array(average_turns)):.2f} +- {np.std(np.array(average_turns)):.2f}")
    print(f"SPACE = {np.mean(np.array(average_space_coverage)):.2f} +- {np.std(np.array(average_space_coverage)):.2f}")
    print(f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} +- {np.std(np.array(average_plume_coverage)):.2f}")
    print(f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} +- {np.std(np.array(average_border_coverage)):.2f}")
    print(f"SUCCESS_RATE = {np.mean(np.array(percentage_verify)):.2f} +- {np.std(np.array(percentage_verify)):.2f}")
    try:
        print(f"ACTUAL BORDER = {np.mean(np.array(actual_average_border_coverage)):.2f} +- {np.std(np.array(actual_average_border_coverage)):.2f}")
        print(f"ACTUAL PLUME = {np.mean(np.array(actual_average_plume_coverage)):.2f} +- {np.std(np.array(actual_average_plume_coverage)):.2f}")
    except:
        pass
    try:
        print(
            f"TP = {np.mean(np.array(tp)):.5f} "
            f"+- {np.std(np.array(tp)):.5f}"
        )
        print(
            f"TN = {np.mean(np.array(tn)):.5f} "
            f"+- {np.std(np.array(tn)):.5f}"
        )
        print(
            f"FP = {np.mean(np.array(fp)):.5f} "
            f"+- {np.std(np.array(fp)):.5f}"
        )
        print(
            f"FN = {np.mean(np.array(fn)):.5f} "
            f"+- {np.std(np.array(fn)):.5f}"
        )
    except:
        pass
    print_map_results(map_rows)
    if results_json is not None:
        settings = {**used, "agent": agent, "baseline": baseline, "deterministic": deterministic, "episodes": episodes, "seed": SEED, "scripted_gas": scripted_gas,
                    "sources": MULTIPLE_GAUSSIANS, "sigma_range": env.base_env.unwrapped.gaussian_sigma_range}
        save_results(results_json, settings, action_names, episode_rows)
    env.close()


def plume_key(base_env) -> str:
    """A short digest of the plumes of this episode. The same keys in the same order show that two evaluations got the same episodes."""
    centers = np.round(np.asarray(base_env.gaussian_centers, dtype=float), 6)
    widths = np.round(np.asarray(base_env.gaussian_sigma, dtype=float), 6)
    return hashlib.md5(centers.tobytes() + widths.tobytes()).hexdigest()[:12]


def save_results(path: Path, settings: dict, action_names: list[str], episodes: list[dict]) -> None:
    """Save an evaluation through a temporary file, so a job that stops midway leaves no half-written results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps({"settings": settings, "action_names": action_names, "episodes": episodes}, default=float))
    temporary.replace(path)


def print_map_results(rows: list[dict[str, float]]) -> None:
    """The map at the end of the episodes, overall and per number of true plumes (sources that overlap count as one)."""
    for name, key in (("PLUME OVERLAP", "plume_overlap"), ("BORDER OVERLAP", "border_overlap"),
                      ("TRUE PLUME VISITED", "true_plume_visited"), ("TRUE BORDER VISITED", "true_border_visited")):
        values = [row[key] for row in rows]
        print(f"{name} = {np.mean(values):.4f} +- {np.std(values):.4f}")
    n_plumes = sum(row["plumes"] for row in rows)
    print(f"PLUMES FOUND = {sum(row['plumes_found'] for row in rows) / n_plumes:.4f} of {n_plumes}")
    print(f"PLUMES VISITED = {sum(row['plumes_visited'] for row in rows) / n_plumes:.4f} of {n_plumes}")
    print("plumes | episodes | ended early | AUV steps | plume overlap | border overlap | plumes found | plumes visited")
    for row in summarise_by_plumes(rows):
        print(f"{row['plumes']:6.0f} | {row['episodes']:8.0f} | {row['ended_early']:11.3f} | {row['steps']:9.1f} | {row['plume_overlap']:13.4f} | {row['border_overlap']:14.4f} | "
              f"{row['plumes_found'] / max(row['plumes'], 1):12.3f} | {row['plumes_visited'] / max(row['plumes'], 1):14.3f}")


def validation(model_path: Path):

    VISUALISE = False
    GP = True
    VIS_GP = False
    EPISODES = 1000

    # Load config from the same folder
    config = load_training_config_from_model(model_path)
    kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    MULTIPLE_GAUSSIANS = config.get("HRL_HYPERPARAMETERS", {}).get("MULTIPLE_GAUSSIANS", None)
    MAX_OPT_LENGTH = 1#config.get("HRL_HYPERPARAMETERS", {}).get("MAX_OPT_LENGTH", None)
   
    # Load HIGH-LEVEL model (the one you trained in train.py)
    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        model = PPO.load(model_path)
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        
        


    MAX_EPS_LEN = config["ENVIRONMENT"]["max_episode_length"] # e.g. 90

    def make_env_fn():
        def _init():
            env, _ = make_test_env(
                GP=GP,
                VIS_GP=VIS_GP,
                VISUALISE=VISUALISE,
                kernel_config=kernel_config,
                MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,
                MAX_OPT_LENGTH=MAX_OPT_LENGTH
            )
            return env
        return _init

    env = SubprocVecEnv([make_env_fn() for _ in range(12)])

    model = PPO.load(model_path)

    results = run_episodes_hrl_vec(
        env,
        model,
        total_episodes=EPISODES,
        max_eps_len=MAX_EPS_LEN
    )

    average_space_coverage=results["space_coverage"]
    average_border_coverage=results["border_coverage"]
    average_plume_coverage=results["plume_coverage"]
    percentage_verify=results["terminated"]# if the agent successfully terminated
    average_turns=results["episode_turns"]
    average_steps=results["steps"]

    print(f"SUCCESS_RATE = {np.mean(np.array(percentage_verify)):.2f} +- {np.std(np.array(percentage_verify)):.2f}")
    print(f"TURNS= {np.mean(np.array(average_turns)):.2f} +- {np.std(np.array(average_turns)):.2f}")
    print(f"STEPS = {np.mean(np.array(average_steps)):.2f} +- {np.std(np.array(average_steps)):.2f}")
    print(f"SPACE = {np.mean(np.array(average_space_coverage)):.2f} +- {np.std(np.array(average_space_coverage)):.2f}")
    print(f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} +- {np.std(np.array(average_plume_coverage)):.2f}")
    print(f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} +- {np.std(np.array(average_border_coverage)):.2f}")
    env.close()

def manual_control():
    """
    Test the environment with manual controls for debugging
    Keys:
    - a: spacial exploration agent (SPACE)
    - s: plume exploration agent (PLUME)
    - d border exploration agent (BORDER)
    """ 
    VISUALISE = True            # Meshcat / real-world simulator
    VIS_STATE_SPACE = False  # State-Space representation
    GP=True            # GP on/off
    VIS_GP = False           # GP visualisation
    EPISODES = 40               
    MAX_EPS_LEN = 420           # maximal episode length
    MULTIPLE_GAUSSIANS = [1,1]
    MAX_OPT_LENGTH = 3
    ACCURACY_GOALS = [0.9,0.9,0.9]
    #config = load_training_config_from_model(model_path_global)
    #kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    kernel_config = None
    if kernel_config is None:
        kernel_config= { #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}# REPLACE with a config term?!
        "RBF": {"type": "RBF", "length_scale": 3.5},
        }
    #np.random.seed(42)
    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=kernel_config,MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS) 
  
    #env.unwrapped.train=True

    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        #model = PPO.load(model_path_global)
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        
        

    
    
    


    obs, _ = env.reset()
    env.base_env.unwrapped.render()
    vis_state_space = visualise_state_space(VIS_STATE_SPACE,obs,cnn,MapVisualiser,HeatmapVisualiser,env.base_env,all_heatmaps=True) # initialising the state space visualisation

    for _ in range(EPISODES):
        obs, reward, done , info = manual_rollout_hrl(env,MAX_EPS_LEN,VISUALISE,vis_state_space,gp_vis,GP)

    env.close()


def make_parser() -> argparse.ArgumentParser:
    """The command line of an evaluation: the meta agent and the survey settings."""
    parser = argparse.ArgumentParser(description="Evaluate a trained HRL meta agent (PPO).")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"meta agent model, e.g. ../trained-agents/HRL_PPO_minimal_4371279/HRL_PPO_scratch.zip (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mode",
        choices=["1", "2", "3", "4", "5"],
        help="1 trained agent, 2 manual control, 3 validation, 4 trained agent with sampled actions, "
        "5 watch the trained agent in Meshcat (default: ask; with --baseline: 1)",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINES,
        help="a meta agent without learning instead of --model: lawnmower (always the sweep, tracks --spacing apart, default 2), "
        "space (always the space sub-agent), rule (a simple rule from the GP map) or random (1/3 per sub-agent)",
    )
    parser.add_argument("--spacing", type=int, help="replace the space sub-agent with a scripted lawnmower, tracks this many cells apart")
    parser.add_argument("--handover", action="store_true", help="give the plume and border sub-agents a scripted way to their target (a model that trained with it always gets it)")
    parser.add_argument("--scripted-gas", action="store_true", help="replace the plume and border sub-agents with a scripted gas option, as a benchmark for them")
    parser.add_argument(
        "--sources",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="smallest and largest number of gas sources per episode. The agent is not told the number (default: as in the model's training)",
    )
    parser.add_argument(
        "--goal",
        choices=GPWrapper.GOAL_MODES,
        help="the end goal: visit the estimated plume and border cells, or estimate them well (default: as in the model's training, otherwise visit)",
    )
    parser.add_argument("--goal-mean-std", type=float, help="with --goal estimate: mean GP std over the map that ends the episode (default: as in the model's training, otherwise 0.1)")
    parser.add_argument("--known-std", type=float, help="with --goal estimate: a cell counts as known at this GP std or lower (default: as in the model's training, otherwise 0.1)")
    parser.add_argument("--unknown-std", type=float, default=0.8, help="with --goal estimate: a cell counts as unknown above this GP std (default: 0.8)")
    parser.add_argument(
        "--max-unknown-cells",
        type=int,
        default=10,
        help="with --goal estimate: the largest connected unknown area, in cells, that still lets the episode end (default: 10)",
    )
    parser.add_argument(
        "--sigma-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="smallest and largest plume width, in cells (default: 2 and 5, as in the HRL training)",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes for modes 1 and 4 (default: 1000)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: the meta agent model and the mode (None: ask for it)."""
    parser = make_parser()
    parser.add_argument("--results-json", type=Path, help="modes 1 and 4: also save the settings and every episode to this file")
    return parser.parse_args(argv)


def survey_settings(args: argparse.Namespace) -> dict:
    """The settings of make_survey from the command line. --baseline lawnmower has tracks 2 cells apart by default."""
    spacing = args.spacing if args.spacing is not None or args.baseline != "lawnmower" else 2
    return {"baseline": args.baseline, "spacing": spacing, "sources": args.sources, "goal": args.goal, "goal_mean_std": args.goal_mean_std,
            "known_std": args.known_std, "unknown_std": args.unknown_std, "max_unknown_cells": args.max_unknown_cells, "scripted_gas": args.scripted_gas,
            "handover": args.handover, "sigma_range": args.sigma_range}


if __name__ == "__main__":
    args = parse_args()
    # Choose whether to run trained agent or manual control
    mode = args.mode or ("1" if args.baseline else input(
        "Enter mode (1 for trained agent, 2 manual control, 3 for validation mode, 4 for trained agent with sampled actions "
        "and 5 to watch the trained agent in Meshcat): "
    ))
    survey = survey_settings(args)

    if mode == "1":
        test_agent(args.model, episodes=args.episodes, results_json=args.results_json, **survey)
    elif mode == "2":
        manual_control()
    elif mode == "3":
        validation(args.model)
    elif mode == "4":
        test_agent(args.model, deterministic=False, episodes=args.episodes, results_json=args.results_json, **survey)
    elif mode == "5":
        actions = input("Enter d for deterministic or s for sampled actions: ")
        if actions in ("d", "s"):
            test_agent(args.model, deterministic=actions == "d", episodes=WATCH_EPISODES, visualise=True, **survey)
        else:
            print("Invalid actions selected")
    else:
        print("Invalid mode selected")
