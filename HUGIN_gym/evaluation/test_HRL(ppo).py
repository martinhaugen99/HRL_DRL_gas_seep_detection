import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import HUGIN_gym  # This import will automatically register the environment


from HUGIN_gym.agents.options import Option
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
from HUGIN_gym.utils.io import hrl_variant_from_config, load_training_config_from_model
from HUGIN_gym.evaluation.rollout import run_episode_hrl, manual_rollout_hrl,run_episodes_hrl_vec

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
                  SURVEY_STEPS=None, VARIANT=None):
    """
    SURVEY_STEPS: survey budget in AUV steps, which ends the episode (None: no budget)
    VARIANT: the meta agent variant, i.e. the HierarchicalEnv settings of a preset in train_HRL(ppo).py (None: one option length, MAX_OPT_LENGTH)
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
    base_env.vary_plume_size = True  # sigma from [2.0, 5.0], as in the HRL training

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
    
    env = HierarchicalEnv(
        base_env=gp_env,
        options=[option1, option2, option3],
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

def test_agent(model_path: Path, deterministic: bool = True, episodes: int = 1000, visualise: bool = False):
    """
    Run the trained meta agent for a number of episodes and print the mean +- std of the results.
    model_path: the meta agent model; training_config.json is read from the same folder
    deterministic: the meta agent takes its most likely action (True) or samples it, as in training (False)
    visualise: show the episodes in the Meshcat browser view, slowed down by STEP_DELAY, with a label of the sub-agent
        in control and a pause after each episode
    """

    VISUALISE = visualise
    GP = True
    VIS_GP = False
    VIS_STATE_SPACE = False
    SEED = 0 # the env draws the plumes and start positions from np.random at reset, so every evaluation gets the same episodes

    # Load config from the same folder
    config = load_training_config_from_model(model_path)
    kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    MULTIPLE_GAUSSIANS = [1,1]#config.get("HRL_HYPERPARAMETERS", {}).get("MULTIPLE_GAUSSIANS", None)
    print(MULTIPLE_GAUSSIANS)
    MAX_OPT_LENGTH = 3#config.get("HRL_HYPERPARAMETERS", {}).get("MAX_OPT_LENGTH", None)
    ACCURACY_GOALS = [0.9,0.9,0.9]
    VARIANT, SURVEY_STEPS = hrl_variant_from_config(config, UNTUNED_VARIANT, UNTUNED_SURVEY_STEPS)
    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=kernel_config,MULTIPLE_GAUSSIANS = MULTIPLE_GAUSSIANS,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS,
                                SURVEY_STEPS=SURVEY_STEPS,VARIANT=VARIANT) # add here a kernel_config from the config file !

    # Load HIGH-LEVEL model (the one you trained in train.py)
    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    model = PPO.load(model_path)
    try:
        model = PPO.load(model_path)
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")

        


    MAX_EPS_LEN = config["ENVIRONMENT"]["max_episode_length"] # e.g. 90

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
    print(f"{'DETERMINISTIC' if deterministic else 'SAMPLED'} actions, {episodes} episodes, seed {SEED}, model {model_path}")
    print(f"REWARD = {np.mean(np.array(average_reward)):.2f} +- {np.std(np.array(average_reward)):.2f}")
    action_names = HierarchicalEnv.action_names(OPTION_NAMES, VARIANT["option_lengths"])
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
    env.close()

    


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: the meta agent model and the mode (None: ask for it)."""
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
        "5 watch the trained agent in Meshcat (default: ask)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    # Choose whether to run trained agent or manual control
    mode = args.mode or input(
        "Enter mode (1 for trained agent, 2 manual control, 3 for validation mode, 4 for trained agent with sampled actions "
        "and 5 to watch the trained agent in Meshcat): "
    )

    if mode == "1":
        test_agent(args.model)
    elif mode == "2":
        manual_control()
    elif mode == "3":
        validation(args.model)
    elif mode == "4":
        test_agent(args.model, deterministic=False)
    elif mode == "5":
        actions = input("Enter d for deterministic or s for sampled actions: ")
        if actions in ("d", "s"):
            test_agent(args.model, deterministic=actions == "d", episodes=WATCH_EPISODES, visualise=True)
        else:
            print("Invalid actions selected")
    else:
        print("Invalid mode selected")
