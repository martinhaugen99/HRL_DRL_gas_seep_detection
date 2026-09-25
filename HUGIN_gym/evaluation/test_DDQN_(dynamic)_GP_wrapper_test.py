import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import torch.nn as nn
import torch as th
from HUGIN_gym.envs.core.visualisation.MapVisualiser import MapVisualiser, CoverageBarVisualiser, visualise_state_space,  visualise_state_space_hrl
from HUGIN_gym.envs.core.visualisation.visualise_heatmap import HeatmapVisualiser, CNN
from HUGIN_gym.utils.io import load_training_config_from_model
import time
import matplotlib.pyplot as plt
from HUGIN_gym.evaluation.rollout import run_episode, manual_rollout
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser



###############################
##### environment creator #####
###############################

def make_test_env(GP=True,AGENT_TYPE="PLUME", VIS_GP=True, VISUALISE = True,kernel_config=None, HRL=False, SUB_AGENT_TRAIN_ON_GP= False):

    base_env = gym.make(
        "HUGIN-v0",
        render_mode="human" if VISUALISE or GP else None,
        max_episode_steps=4
    )
    base_env = base_env.unwrapped
    base_env.agent_type = AGENT_TYPE
    base_env.train = False
    base_env.random_points = True
    base_env.multiple_gaussians = [1,1] # testing 2 instead of 1 source at the same time
    base_env.GP_ON = GP
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
            HRL=HRL,
            SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP
        )

        gp_vis = GPVisualiser(
            fig_num=33,
            shift=base_env.unwrapped.offset[0], #careful! assuming dim_len X=Y
            kernel_config=kernel_config
        ) if VIS_GP else None
    else:
        gp_vis = None


    gp_env = gp_env if GP else base_env
   

    return gp_env,gp_vis

###### test modes


model_path_global = "/Users/martinhaugen/Desktop/master/uio/fall26/trained_agents/DQN_gp_230steps_4218364/DQN_scratch.zip"#/space_diff_NEW_visited

def test_agent():
    VISUALISE = True      # Meshcat / real-world simulator
    VIS_STATE_SPACE = False  # State-Space representation
    GP=True  # GP on/off
    VIS_GP = False             # GP visualisation
    EPISODES = 1000
    SUB_AGENT_TRAIN_ON_GP=True
    
    # Load model
    try:
        model = DQN.load(model_path_global)
        cnn_full = model.policy.q_net.features_extractor.cnn
        # Remove Flatten layer
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        
    # Load config from the same folder
    config = load_training_config_from_model(model_path_global)
    
    percentage_verify =[]
    average_turns = []
    average_steps = []
    average_plume_coverage =[]
    average_border_coverage =[]

    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=None, HRL=False,SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP)
    vis_state_space = visualise_state_space_hrl(VIS_STATE_SPACE,cnn,MapVisualiser,HeatmapVisualiser) # initialising the state space visualisation

    for _ in range(EPISODES):
        result = run_episode(
            env,
            model,
            config,
            vis_state_space=vis_state_space,
            gp_visualise=gp_vis,
            GP=GP,
            VISUALISE=VISUALISE,
            testing_space_agent = False, # for getting border and plume coverrage. Otherwise c_map is off
        )
        
        average_border_coverage.append(result["border_coverage"])
        average_plume_coverage.append(result["plume_coverage"])
        percentage_verify.append(result["space_coverage"])
        average_turns.append(result["episode_turns"])
        average_steps.append(result["steps"])
    print(f"SPACE = {np.mean(np.array(percentage_verify)):.2f} +- {np.std(np.array(percentage_verify)):.2f}")
    print(f"TURNS = {np.mean(np.array(average_turns)):.2f} +- {np.std(np.array(average_turns)):.2f}")
    print(f"STEPS= {np.mean(np.array(average_steps)):.2f} +- {np.std(np.array(average_steps)):.2f}")
    print(f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} +- {np.std(np.array(average_plume_coverage)):.2f}")
    print(f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} +- {np.std(np.array(average_border_coverage)):.2f}")
    env.close()


def test_agent_manual_input():
    pass


def manual_control():
    """
    Test the environment with manual controls for debugging
    Keys:
    - space: Forward (2 cells)
    - w/s: up/down (1 forward, 1up/down)
    - a/d left/right (1 forward, 1left/right)
    """ 
    VISUALISE = True           # Meshcat / real-world simulator
    VIS_STATE_SPACE = True    # State-Space representation
    GP=True      # GP on/off
    VIS_GP = True              # GP visualisation
    EPISODES = 20               
    MAX_EPS_LEN = 100          # maximal episode length
    AGENT_TYPE="PLUME"       # defining reward function and termination condition
    HRL = False
    SUB_AGENT_TRAIN_ON_GP=True
    

    kernel_config= { #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}# REPLACE with a config term?!
        "RBF": {"type": "RBF", "length_scale": 3.5},
        }
    

    env, gp_vis = make_test_env(
    GP=GP,
    AGENT_TYPE=AGENT_TYPE,
    VIS_GP=VIS_GP,
    VISUALISE=VISUALISE,
    kernel_config=kernel_config,
    HRL=HRL,
    SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP
)
    
    env.unwrapped.random_points=True

    
    env.unwrapped.use_c_map=(AGENT_TYPE!="SPACE")
    #env.unwrapped.train=True

    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        model = DQN.load(model_path_global)
        cnn_full = model.policy.q_net.features_extractor.cnn
        # Remove Flatten layer
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        

    
    
    obs, _ = env.reset()
   
    # if GP:
    #     gp_env = GPWrapper(env,kernels_config=kernel_config,gp_visualise=VIS_GP,HRL=HRL,SUB_AGENT_TRAIN_ON_GP =SUB_AGENT_TRAIN_ON_GP)
    #     obs, _ = gp_env.reset()
    #     if VIS_GP:
    #         gp_vis = GPVisualiser(fig_num=3,shift=gp_env.unwrapped.offset[0], kernel_config= kernel_config)
    #     else:
    #         gp_vis = None
    # else:
    #     obs, _ = env.reset()
    #     gp_vis = None
    #     gp_env=env
    #print(obs["space_coverage"],obs["plume_coverage"],obs["border_coverage"])
    #print(type(obs["space_coverage"][0]),type(obs["plume_coverage"][0]),type(obs["border_coverage"][0]))
    env.unwrapped.render()
    vis_state_space = visualise_state_space(VIS_STATE_SPACE,obs,cnn,MapVisualiser,HeatmapVisualiser,env,all_heatmaps=True,CoverageBarVisualiser=CoverageBarVisualiser) # initialising the state space visualisation

    for _ in range(EPISODES):
        obs, reward, done , info = manual_rollout(env,MAX_EPS_LEN,VISUALISE,vis_state_space,gp_vis)

    env.close()


if __name__ == "__main__":
    # Choose whether to run trained agent or manual control
    mode = input(
        "Enter mode (1 for trained agent, 2 for manual control and 3 for predefined manual input): "
    )

    if mode == "1":
        test_agent()
    elif mode == "2":
        manual_control()
    elif mode == "3":
        test_agent_manual_input()
    else:
        print("Invalid mode selected")
