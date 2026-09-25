import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch.nn as nn
import torch as th
from HUGIN_gym.envs.core.visualisation.MapVisualiser import (
    MapVisualiser,
    CoverageBarVisualiser,
    visualise_state_space,
    visualise_state_space_hrl,
)
from HUGIN_gym.envs.core.visualisation.visualise_heatmap import (
    HeatmapVisualiser,
    CNN,CNN3D
)
from HUGIN_gym.utils.io import load_training_config_from_model
import matplotlib.pyplot as plt
from HUGIN_gym.evaluation.rollout import run_episode, manual_rollout
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser
from HUGIN_gym.evaluation.lawnmower_path_generator import build_lawnmower_path_from_initial_pose
from HUGIN_gym.evaluation._3D_lawnmower import build_lawnmower_path_3d_from_initial_pose
import time
###############################
##### environment creator #####
###############################


def make_test_env(
    GP=True,
    AGENT_TYPE="PLUME",
    VIS_GP=True,
    VISUALISE=True,
    kernel_config=None,
    HRL=False,
    SUB_AGENT_TRAIN_ON_GP=False,
    ACCURACY_GOALS = [0.9,1.0,1.0]
):

    base_env = gym.make(
        "HUGIN-v0",
        render_mode="human" if VISUALISE or GP else None,
        max_episode_steps=4,
    )
    base_env = base_env.unwrapped
    base_env.agent_type = AGENT_TYPE
    base_env.train = False
    base_env.random_points = True
    base_env.multiple_gaussians = [1,1]
    #base_env.gaussian_sigma = 7.0
    #base_env.gaussian_amplitude = 1.0
    base_env.GP_ON = GP
    base_env.accuracy_agent_goals = ACCURACY_GOALS

    if GP:
        if kernel_config is None:
            kernel_config = {
                "RBF": {"type": "RBF", "length_scale": 3.5},
            }
        gp_env = GPWrapper(
            base_env,
            kernels_config=kernel_config,
            gp_visualise=VIS_GP,
            HRL=HRL,
            SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
        )
        gp_env.accuracy_goals = ACCURACY_GOALS
        gp_vis = (
            GPVisualiser(
                fig_num=33,
                shift=base_env.unwrapped.offset[0],
                kernel_config=kernel_config,
            )
            if VIS_GP
            else None
        )
    else:
        gp_env = base_env
        gp_vis = None

    return gp_env, gp_vis


###### test modes

# Path to your PPO trained agent directory (where PPO_scratch is saved)
#model_path_global = "../trained-agents/PPO_border_GP_d_to_max_only_terminate_after_plume/PPO_scratch"
model_path_global = "/Users/martinhaugen/Desktop/master/uio/fall26/trained_agents/PPO_gp_230steps_4218364/PPO_scratch.zip"

def test_agent():
    VISUALISE = True  # Meshcat / real-world simulator
    VIS_STATE_SPACE = False
    GP = True          # GP on/off
    VIS_GP = False       # GP visualisation
    EPISODES = 100
    SUB_AGENT_TRAIN_ON_GP = True

    # Load PPO model
    try:
        model = PPO.load(model_path_global)
        # For PPO, the custom feature extractor is at model.policy.features_extractor
        cnn_full = model.policy.features_extractor.cnn
        # Remove Flatten layer (keep conv part only)
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained PPO model")
    except Exception as e:
        print(f"Failed to load PPO model or CNN ({e}), using default CNN")
        model = PPO.load(model_path_global)
        cnn = CNN(layers=3)

    # Load config from the same folder as the model
    config = load_training_config_from_model(model_path_global)

    percentage_verify = []
    average_turns = []
    average_steps = []
    average_plume_coverage = []
    average_border_coverage = []
    actual_average_plume_coverage =[]
    actual_average_border_coverage =[]
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    RMSE_c=[]
    extracted_GP_imgs =[]
    env, gp_vis = make_test_env(
        GP=GP,
        VIS_GP=VIS_GP,
        VISUALISE=VISUALISE,
        kernel_config=None,
        HRL=False,
        SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
        ACCURACY_GOALS=[0.9,1.0,1.0],
    )
    env.unwrapped.SPACE_test = True
    env.action_plan_len=-1
    env.unwrapped.max_eps_len = config["ENVIRONMENT"]["max_episode_length"]
    vis_state_space = visualise_state_space_hrl(
        VIS_STATE_SPACE, cnn, MapVisualiser, HeatmapVisualiser,
    )
    
    for _ in range(EPISODES):
        result = run_episode(
            env,
            model,
            config,
            vis_state_space=vis_state_space,
            gp_visualise=gp_vis,
            GP=GP,
            VISUALISE=VISUALISE,
            testing_space_agent=True,
            CoverageBarVisualiser=CoverageBarVisualiser,
            verbose=False # True
        )

        try:
            
            fp.append(result["info"]["false_positive_above"]) 
            tp.append(result["info"]["true_positive_above"]) 
            fn.append(result["info"]["false_negative_above"]) 
            tn.append(result["info"]["true_negative_above"]) 
            extracted_GP_imgs.append(env.unwrapped.extract_GP_ERROR_map)
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
        diff = env.GP_pred.T-env.unwrapped.conc[:,:,0].T          # numpy array
        mse  = (diff ** 2).mean()
        rmse = np.sqrt(mse)

        RMSE_c.append(rmse)
        percentage_verify.append(100*result["space_coverage"])
        average_turns.append(result["episode_turns"])
        average_steps.append(result["steps"])
        average_plume_coverage.append(100*result["plume_coverage"])
        average_border_coverage.append(100*result["border_coverage"])
        # plt.figure(101)
        # im = plt.imshow(env.unwrapped.extract_GP_ERROR_map,vmin=0, vmax=1)
        # # 2) Create the colorbar from the image (mappable)
        # cbar = plt.colorbar(im,
        #                     ticks=[0, 0.5, 1.0])  # <- tick values you want to show

        # # 3) Add a label (title) to the colorbar
        # cbar.set_label("GP error [arb. units]", rotation=90, fontsize=24)  # or your preferred text
        # cbar.ax.tick_params(labelsize=20)
        # plt.xlabel("X [$m$]", fontsize=24)
        # plt.ylabel("Y [$m$]", fontsize=24)
        # plt.tick_params(axis="both", which="both", labelsize=20)
        # plt.tight_layout()
        # plt.savefig("UNCERTAINTY.pdf", format="pdf")
        # plt.show(block=False)
        # plt.pause(0.001)
        # plt.clf()

    print(
        f"RMSE = {np.mean(np.array(RMSE_c)):.5f} "
        f"+- {np.std(np.array(RMSE_c)):.5f}"
    )
    print(
        f"SPACE = {np.mean(np.array(percentage_verify)):.5f} "
        f"+- {np.std(np.array(percentage_verify)):.5f}"
    )
    print(
        f"TURNS = {np.mean(np.array(average_turns)):.5f} "
        f"+- {np.std(np.array(average_turns)):.5f}"
    )
    print(
        f"STEPS= {np.mean(np.array(average_steps)):.5f} "
        f"+- {np.std(np.array(average_steps)):.5f}"
    )
    print(
        f"PLUME = {np.mean(np.array(average_plume_coverage)):.5f} "
        f"+- {np.std(np.array(average_plume_coverage)):.5f}"
    )
    print(
        f"BORDER = {np.mean(np.array(average_border_coverage)):.5f} "
        f"+- {np.std(np.array(average_border_coverage)):.5f}"
    )
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
    
    # plt.figure(101)
    # im = plt.imshow(np.mean(np.array(extracted_GP_imgs),axis=0),vmin=0, vmax=1)
    # # 2) Create the colorbar from the image (mappable)
    # cbar = plt.colorbar(im,
    #                     ticks=[0, 0.5, 1.0])  # <- tick values you want to show

    # # 3) Add a label (title) to the colorbar
    # cbar.set_label("GP error", rotation=90)  # or your preferred text
    # plt.show()

    env.close()


def test_lawnmower_auto():
    """
    Run episodes using a precomputed lawnmower action sequence.
    The entire action list is planned once per episode from the initial state.
    """
    VISUALISE = True
    GP = True
    VIS_GP = True
    EPISODES = 1000
    SUB_AGENT_TRAIN_ON_GP = True

    # baseline: SPACE agent for coverage
    env, gp_vis = make_test_env(
        GP=GP,
        AGENT_TYPE="PLUME",
        VIS_GP=VIS_GP,
        VISUALISE=VISUALISE,
        kernel_config=None,
        HRL=False,
        SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
    )
    
    env.unwrapped.SPACE_test = True
    #config = load_training_config_from_model(model_path_global)
    env.unwrapped.max_eps_len = 460#config["ENVIRONMENT"]["max_episode_length"]
    #env.unwrapped.accuracy_goals = [0.9,1.0,1.0]
    env.accuracy_goals = [1.0,1.0,1.0]
    percentage_verify = []
    average_turns = []
    average_steps = []
    average_plume_coverage = []
    average_border_coverage = []
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    RMSE_c=[]
    for _ in range(EPISODES):
        obs, info = env.reset()
        # env.unwrapped.state["x"] = 2
        # env.unwrapped.state["y"] = 1
        # env.unwrapped.state["theta"] = np.pi/2
         # set up renderer once per episode
        if VISUALISE:
            try:
                env.unwrapped.renderer.vis.delete()
                env.unwrapped.renderer.reset()
            except:
                pass
            
            env.unwrapped.render()
            
       
        # Plan once based on *initial* state
        max_steps = 3000
        if env.unwrapped._3D:
            actions_plan = build_lawnmower_path_3d_from_initial_pose(env.unwrapped)
        else:
            actions_plan = build_lawnmower_path_from_initial_pose(env.unwrapped)
        env.action_plan_len = len(actions_plan)
        steps = 0
        episode_turns = 0
        terminated = False
        truncated = False

        for action in actions_plan:
            if terminated or truncated:
                break

            if action in (3, 4):
                episode_turns += 1

            obs, reward, terminated, truncated, info = env.step(action)
            # update visualisation
            if VISUALISE:
                env.unwrapped.step_sim()
                time.sleep(0.02)
            steps += 1
            
        # Collect metrics from final obs
        try:
            average_border_coverage.append(100*(info["actual_current_around_threshold"]))
        except:
            pass#print(env.SPACE_test_counter,env.unwrapped.max_eps_len,env.unwrapped.SPACE_test)
        diff = env.GP_pred.T-env.unwrapped.conc[:,:,0].T          # numpy array
        mse  = (diff ** 2).mean()
        rmse = np.sqrt(mse)

        RMSE_c.append(rmse)
        average_plume_coverage.append(100*(info["actual_current_above_threshold"]))
        percentage_verify.append(100*obs["space_coverage"][0])
        average_turns.append(episode_turns)
        average_steps.append(steps)

        fp.append(info["false_positive_above"]) 
        tp.append(info["true_positive_above"]) 
        fn.append(info["false_negative_above"]) 
        tn.append(info["true_negative_above"]) 

            
        # plt.figure(101)
        # im = plt.imshow(env.unwrapped.extract_GP_ERROR_map,vmin=0, vmax=1)
        # # 2) Create the colorbar from the image (mappable)
        # cbar = plt.colorbar(im,
        #                     ticks=[0, 0.5, 1.0])  # <- tick values you want to show

        # # 3) Add a label (title) to the colorbar
        # cbar.set_label("GP error", rotation=90)  # or your preferred text
        # plt.show()

    print(
        f"RMSE = {np.mean(np.array(RMSE_c)):.5f} "
        f"+- {np.std(np.array(RMSE_c)):.5f}"
    )

    print(
        f"SPACE = {np.mean(np.array(percentage_verify)):.5f} "
        f"+- {np.std(np.array(percentage_verify)):.5f}"
    )
    print(
        f"TURNS = {np.mean(np.array(average_turns)):.5f} "
        f"+- {np.std(np.array(average_turns)):.5f}"
    )
    print(
        f"STEPS= {np.mean(np.array(average_steps)):.5f} "
        f"+- {np.std(np.array(average_steps)):.5f}"
    )
    print(
        f"PLUME = {np.mean(np.array(average_plume_coverage)):.5f} "
        f"+- {np.std(np.array(average_plume_coverage)):.5f}"
    )
    print(
        f"BORDER = {np.mean(np.array(average_border_coverage)):.5f} "
        f"+- {np.std(np.array(average_border_coverage)):.5f}"
    )

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
    env.close()




def manual_control():
    """
    Test the environment with manual controls for debugging.
    Keys:
    - space: Forward (2 cells)
    - w/s: up/down (1 forward, 1 up/down)
    - a/d left/right (1 forward, 1 left/right)
    """
    VISUALISE = True
    VIS_STATE_SPACE = False
    GP = True
    VIS_GP = False
    EPISODES = 20
    MAX_EPS_LEN = 8000
    AGENT_TYPE = "PLUME"
    HRL = False
    SUB_AGENT_TRAIN_ON_GP = True

    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
    }

    env, gp_vis = make_test_env(
        GP=GP,
        AGENT_TYPE=AGENT_TYPE,
        VIS_GP=VIS_GP,
        VISUALISE=VISUALISE,
        ACCURACY_GOALS=[0.9,1.0,1.0],
        kernel_config=kernel_config,
        HRL=HRL,
        SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
    )

   
    env.unwrapped.random_points = True
    env.unwrapped.use_c_map = (AGENT_TYPE != "SPACE")

    # CNN reconstruction for downsampling
    try:
        model = PPO.load(model_path_global)
        cnn_full = model.policy.features_extractor.cnn
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained PPO model")
    except Exception as e:
        print(f"Failed to load PPO model or CNN ({e}), using default CNN")
        if env.unwrapped._3D:
            cnn = CNN3D(layers=3)
        else:
            cnn = CNN(layers=3)

    obs, _ = env.reset()
    env.unwrapped.render()

    vis_state_space = visualise_state_space(
        VIS_STATE_SPACE,
        obs,
        cnn,
        MapVisualiser,
        HeatmapVisualiser,
        env,
        all_heatmaps=True,
        CoverageBarVisualiser=CoverageBarVisualiser,
    )

    for _ in range(EPISODES):
        obs, reward, done, info = manual_rollout(
            env, MAX_EPS_LEN, VISUALISE, vis_state_space, gp_vis
        )

    env.close()


if __name__ == "__main__":
    mode = input(
        "Enter mode (1 for trained PPO agent, 2 for manual control and 3 for predefined lawnmower pattern): "
    )

    if mode == "1":
        test_agent()
    elif mode == "2":
        manual_control()
    elif mode == "3":
        test_lawnmower_auto()
    else:
        print("Invalid mode selected")
