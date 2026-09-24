# train_SAC.py
import argparse
import gymnasium as gym
import torch
from stable_baselines3 import SAC
import pickle
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.utils.io import confirm_overwrite
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.FilterObservationWrapper import FilterObservationWrapper
from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.utils.build_train_config import build_training_config_sac  # or define a SAC-specific variant
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.wrappers.continuous_wrapper import ContinuousToDiscreteActionWrapper  


def main():
    torch.set_num_threads(1)  # the network is tiny: extra torch threads only compete with the env workers for CPU
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="run name, i.e. the output folder in ../trained-agents (default: NAME below)")
    args = parser.parse_args()

    MAX_EPS_LEN = 460
    NUM_ENVS = 12
    GP = False
    ACCURACY_GOALS = [0.9, 1.0, 1.0]
    SUB_AGENT_TRAIN_ON_GP = False
    AGENT_TYPE = "PLUME"

    if AGENT_TYPE == "BORDER":
        from HUGIN_gym.agents.feature_extractor.agent3_border_explore import (
            AgentBoarderExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            "visited_maps_downsampled",
            "GT_c_around_threshold_maps_downsampled",
            "local_GT_around",
            "border_coverage",
            "space_coverage",
            "distance_from_max",
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent3_border.py"

    elif AGENT_TYPE == "SPACE":
        from HUGIN_gym.agents.feature_extractor.agent1_space_explore import (
            AgentSpaceExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            "visited_maps_downsampled",
            "space_coverage",
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent1_exploration.py"

    elif AGENT_TYPE == "PLUME":
        from HUGIN_gym.agents.feature_extractor.agent2_plume_explore import (
            AgentPlumeExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            "visited_maps_downsampled",
            "GT_c_over_threshold_maps_downsampled",
            "local_GT",
            "plume_coverage",
            "space_coverage",
            "distance_from_max",
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent2_plume.py"

    NAME = args.name or "SAC_space_GT_not_clipped_RWD"

    saving_location = f"../trained-agents/{NAME}"
    loading_location = "../trained-agents/..._what_so_ever_..."

    confirm_overwrite(saving_location)

    def episode_length_schedule(local_step):
        return MAX_EPS_LEN

    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
    }

    def make_env():
        def _init():
            env = gym.make("HUGIN-v0")
            env = env.unwrapped
            env.train = True
            env.accuracy_agent_goals = ACCURACY_GOALS
            env.GP_ON = GP
            if AGENT_TYPE == "SPACE":
                env.use_c_map = False
            else:
                env.use_c_map = True
            env.agent_type = AGENT_TYPE

            env = DynamicEpisodeLengthWrapper(env, schedule_fn=episode_length_schedule)

            if GP:
                gp_env = GPWrapper(
                    env,
                    kernels_config=kernel_config,
                    HRL=False,
                    SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
                )
                gp_env.accuracy_goals = ACCURACY_GOALS
                gp_env.reset()
            else:
                gp_env = env

            gp_env = FilterObservationWrapper(
                gp_env, keys_to_keep=keys_you_want_to_keep
            )

            # expose continuous action space for SAC
            sac_env = ContinuousToDiscreteActionWrapper(gp_env)
            return sac_env

        return _init

    dummy_env = make_env()()
    N_STATES = dummy_env.unwrapped.max_return
    dummy_env.close()

    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = VecMonitor(SubprocVecEnv(env_fns))  # records episode reward/length for tensorboard, obs/rewards pass through unchanged
    policy_kwargs = dict(features_extractor_class=Agent)

    load_existing = False

    if load_existing:
        model = SAC.load(
            f"{loading_location}/SAC_scratch",
            env=env,
            tensorboard_log="./tensorboard_logs/",
        )
    else:
        # SAC hyperparameters chosen analogous to your DQN setup
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=5e-5,
            buffer_size=900_000,
            batch_size=256,
            gamma=0.997,
            tau=0.01,
            train_freq=4,
            gradient_steps=4,
            ent_coef="auto",
            target_update_interval=1,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{saving_location}/tensorboard",  # live curves: tensorboard --logdir ../trained-agents
        )

    max_steps = 10_000_000
    checkpoint_steps = 2_000_000 // NUM_ENVS  # save the model every n steps, adjusted for number of parallel envs
    stats_steps = 5_000_000 // NUM_ENVS  # save episode stats every n steps (each file holds all episodes so far)

    stats_callback = EpisodeStatsCallback(
        max_episode_length=MAX_EPS_LEN,
        save_freq=stats_steps,
        NUM_ENVS=NUM_ENVS,
        address=f"{saving_location}/episode_stats",
        N_states=N_STATES,
        GP=GP,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_steps,
        save_path=f"{saving_location}/",
        name_prefix="SAC_checkpoint",
        save_replay_buffer=False,
    )

   
    build_training_config_sac(
    model,
    AGENT_TYPE,
    keys_you_want_to_keep,
    agent_reward_path,
    max_steps,
    MAX_EPS_LEN,
    checkpoint_steps,
    saving_location,
    kernel_config=kernel_config if GP else None,
)

    model.learn(
        total_timesteps=max_steps,
        callback=[stats_callback, checkpoint_callback],
        progress_bar=True,
    )

    stats = stats_callback.get_stats()
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    model.save(f"{saving_location}/SAC_scratch")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    main()
