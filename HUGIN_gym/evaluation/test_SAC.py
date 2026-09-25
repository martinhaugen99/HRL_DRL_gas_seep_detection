import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import torch.nn as nn
import torch as th
from gymnasium import spaces

from HUGIN_gym.envs.core.visualisation.MapVisualiser import (
    MapVisualiser,
    CoverageBarVisualiser,
    visualise_state_space,
    visualise_state_space_hrl,
)
from HUGIN_gym.envs.core.visualisation.visualise_heatmap import (
    HeatmapVisualiser,
    CNN,
)
from HUGIN_gym.utils.io import load_training_config_from_model
import matplotlib.pyplot as plt
from HUGIN_gym.evaluation.rollout import run_episode, manual_rollout
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser


###############################
##### continuous→discrete wrapper
###############################


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """
    Wrap a discrete-action env into a continuous-action env for SAC.

    Original: Discrete(n_actions)
    Wrapped:  Box(shape=(1,), low=-1, high=1)

    The scalar in [-1,1] is binned into n_actions discrete actions.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(
            env.action_space, spaces.Discrete
        ), "ContinuousToDiscreteActionWrapper requires a Discrete action space."
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, act_continuous):
        """
        Map continuous scalar in [-1,1] to discrete action {0, ..., n_actions-1}.
        """
        x = float(np.clip(act_continuous[0], -1.0, 1.0))
        frac = (x + 1.0) / 2.0  # in [0,1]
        disc = int(frac * self.n_actions)
        disc = min(disc, self.n_actions - 1)
        return disc


###############################
##### environment creator #####
###############################


def make_test_env(
    GP=True,
    AGENT_TYPE="CHANGE",
    VIS_GP=True,
    VISUALISE=True,
    kernel_config=None,
    HRL=False,
    SUB_AGENT_TRAIN_ON_GP=False,
    for_sac=False,
):
    """
    If for_sac=True: wrap in ContinuousToDiscreteActionWrapper so SAC can control it.
    If for_sac=False: return plain (discrete) env, used e.g. for manual control.
    """

    base_env = gym.make(
        "HUGIN-v0",
        render_mode="human" if VISUALISE or GP else None,
        max_episode_steps=4,
    )
    base_env = base_env.unwrapped
    base_env.agent_type = AGENT_TYPE
    base_env.train = False
    base_env.random_points = True
    base_env.multiple_gaussians = [1, 1]
    base_env.GP_ON = GP

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

    if for_sac:
        # SAC expects a continuous action space
        sac_env = ContinuousToDiscreteActionWrapper(gp_env)
        return sac_env, gp_vis
    else:
        # used for manual control (discrete actions)
        return gp_env, gp_vis


###### test modes

# Path to your SAC trained agent file (model.save(...))
model_path_global = "trained-agents/SAC_space_GT_not_clipped_RWD/SAC_scratch.zip"


def test_agent():
    VISUALISE = True       # Meshcat / visualisation
    VIS_STATE_SPACE = False
    GP = True              # GP on/off (match training)
    VIS_GP = False         # GP visualisation
    EPISODES = 1000
    SUB_AGENT_TRAIN_ON_GP = True
    AGENT_TYPE = "PLUME"   # set accordingly

    # Load SAC model
    try:
        model = SAC.load(model_path_global)
        # Custom feature extractor is at model.policy.features_extractor
        cnn_full = model.policy.features_extractor.cnn
        # Remove Flatten layer (keep conv part only)
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained SAC model")
    except Exception as e:
        print(f"Failed to load SAC model or CNN ({e}), using default CNN")
        cnn = CNN(layers=3)

    # Load config from the same folder as the model
    config = load_training_config_from_model(model_path_global)

    percentage_verify = []
    average_turns = []
    average_steps = []
    average_plume_coverage = []
    average_border_coverage = []

    # For SAC: for_sac=True so we wrap in ContinuousToDiscreteActionWrapper
    env, gp_vis = make_test_env(
        GP=GP,
        AGENT_TYPE=AGENT_TYPE,
        VIS_GP=VIS_GP,
        VISUALISE=VISUALISE,
        kernel_config=None,
        HRL=False,
        SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
        for_sac=True,
    )

    vis_state_space = visualise_state_space_hrl(
        VIS_STATE_SPACE, cnn, MapVisualiser, HeatmapVisualiser
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
            testing_space_agent=False,
        )

        average_border_coverage.append(result["border_coverage"])
        average_plume_coverage.append(result["plume_coverage"])
        percentage_verify.append(result["space_coverage"])
        average_turns.append(result["episode_turns"])
        average_steps.append(result["steps"])

    print(
        f"SPACE = {np.mean(np.array(percentage_verify)):.2f} "
        f"+- {np.std(np.array(percentage_verify)):.2f}"
    )
    print(
        f"TURNS = {np.mean(np.array(average_turns)):.2f} "
        f"+- {np.std(np.array(average_turns)):.2f}"
    )
    print(
        f"STEPS= {np.mean(np.array(average_steps)):.2f} "
        f"+- {np.std(np.array(average_steps)):.2f}"
    )
    print(
        f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} "
        f"+- {np.std(np.array(average_plume_coverage)):.2f}"
    )
    print(
        f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} "
        f"+- {np.std(np.array(average_border_coverage)):.2f}"
    )

    env.close()


def test_agent_manual_input():
    pass


def manual_control():
    """
    Test the environment with manual controls for debugging.
    Keys:
    - space: Forward (2 cells)
    - w/s: up/down (1 forward, 1 up/down)
    - a/d left/right (1 forward, 1 left/right)
    """
    VISUALISE = True
    VIS_STATE_SPACE = True
    GP = True
    VIS_GP = True
    EPISODES = 20
    MAX_EPS_LEN = 100
    AGENT_TYPE = "PLUME"
    HRL = False
    SUB_AGENT_TRAIN_ON_GP = True

    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
    }

    # For manual control: for_sac=False -> no continuous wrapper
    env, gp_vis = make_test_env(
        GP=GP,
        AGENT_TYPE=AGENT_TYPE,
        VIS_GP=VIS_GP,
        VISUALISE=VISUALISE,
        kernel_config=kernel_config,
        HRL=HRL,
        SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
        for_sac=False,
    )

    env.unwrapped.random_points = True
    env.unwrapped.use_c_map = (AGENT_TYPE != "SPACE")

    # CNN reconstruction for downsampling (optional, from SAC model if available)
    try:
        model = SAC.load(model_path_global)
        cnn_full = model.policy.features_extractor.cnn
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained SAC model")
    except Exception as e:
        print(f"Failed to load SAC model or CNN ({e}), using default CNN")
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
        "Enter mode (1 for trained SAC agent, 2 for manual control and 3 for predefined manual input): "
    )

    if mode == "1":
        test_agent()
    elif mode == "2":
        manual_control()
    elif mode == "3":
        test_agent_manual_input()
    else:
        print("Invalid mode selected")
