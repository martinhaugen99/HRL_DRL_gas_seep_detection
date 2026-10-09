# train_DDQN.py
import argparse
import gymnasium as gym
import torch
from stable_baselines3 import DQN
import pickle # saving the stats
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.utils.io import confirm_overwrite
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.FilterObservationWrapper import FilterObservationWrapper
from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.utils.build_train_config import build_training_config
from HUGIN_gym.evaluation.rollout import run_episode, manual_rollout
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser
from HUGIN_gym.training.train_DDQN_config import FULL_CONFIG, GP_2D_CONFIG, SMOKE_CONFIG


def main():
    torch.set_num_threads(1) # the network is tiny: extra torch threads only compete with the env workers for CPU
    parser = argparse.ArgumentParser()
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--smoke", action="store_true")
    preset_group.add_argument("--full", action="store_true")
    parser.add_argument("--name", help="run name, i.e. the output folder in ../trained-agents (default: the preset's name)")
    args = parser.parse_args()
    if args.smoke:
        config = SMOKE_CONFIG
    elif args.full:
        config = FULL_CONFIG
    else:
        config = GP_2D_CONFIG  # default: GP on

    MAX_EPS_LEN = config["max_episode_length"]
    NUM_ENVS = config["num_envs"]
    GP = config["gp"]
    ACCURACY_GOALS = [0.9, 1.0, 0.9]
    SUB_AGENT_TRAIN_ON_GP = True
    AGENT_TYPE = "BORDER"

    if AGENT_TYPE == "BORDER":
        from HUGIN_gym.agents.feature_extractor.agent3_border_explore import AgentBoarderExplore as Agent
        keys_you_want_to_keep = ["obs_state", "visited_maps_downsampled", "GT_c_around_threshold_maps_downsampled","local_GT_around","border_coverage","space_coverage","distance_from_max"]
        # if SUB_AGENT_TRAIN_ON_GP:
        #     keys_you_want_to_keep.append("c_around_threshold_maps")
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent3_border.py"
        SPAWN_CLOSE_TO_SOURCE = True

    elif AGENT_TYPE == "SPACE":
        from HUGIN_gym.agents.feature_extractor.agent1_space_explore import AgentSpaceExplore as Agent
        keys_you_want_to_keep = ["obs_state", "visited_maps_downsampled","space_coverage"]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent1_exploration.py"
        SPAWN_CLOSE_TO_SOURCE = False

    elif AGENT_TYPE == "PLUME":
        from HUGIN_gym.agents.feature_extractor.agent2_plume_explore import AgentPlumeExplore as Agent
        keys_you_want_to_keep = ["obs_state", "visited_maps_downsampled", "GT_c_over_threshold_maps_downsampled","local_GT","plume_coverage","space_coverage","distance_from_max"]
        # if SUB_AGENT_TRAIN_ON_GP:
        #     keys_you_want_to_keep.append("c_over_threshold_maps")
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent2_plume.py"
        SPAWN_CLOSE_TO_SOURCE = True
                
    NAME = args.name or config["name"]
   

    saving_location = f"../trained-agents/{NAME}"
    loading_location = "../trained-agents/..._what_so_ever_..."
    
    confirm_overwrite(saving_location)

    def episode_length_schedule(local_step):
        return MAX_EPS_LEN  # Spaceholder for a more complex function
 
    
    # GP init
    kernel_config= { 
        "RBF": {"type": "RBF", "length_scale": 3.5},} 
        #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}
    def make_env():
        def _init():
            env = gym.make("HUGIN-v0")
            env = env.unwrapped  # REMOVE default TimeLimit wrapper
            env.train = True
            env.accuracy_agent_goals = ACCURACY_GOALS
            env.spawn_close_to_source = SPAWN_CLOSE_TO_SOURCE
            env.GP_ON=GP
            if AGENT_TYPE == "SPACE":
                env.use_c_map = False ## for spatial explorer
            else:
                env.use_c_map = True
            env.agent_type = AGENT_TYPE
            env = DynamicEpisodeLengthWrapper(env, schedule_fn=episode_length_schedule) # spaceholder for use of a Dynamic Episode Length
             # Filtering the Obs space dependant on the agent
            if GP:
                gp_env = GPWrapper(env,kernels_config=kernel_config,HRL=False,SUB_AGENT_TRAIN_ON_GP =SUB_AGENT_TRAIN_ON_GP)
                gp_env.accuracy_goals = ACCURACY_GOALS
                gp_env.reset()
            else:
                gp_env = env
            gp_env = FilterObservationWrapper(gp_env,keys_to_keep=keys_you_want_to_keep)
            return gp_env
        return _init
    
    dummy_env = make_env()()   # call twice: make_env returns _init
    N_STATES = dummy_env.unwrapped.max_return
    dummy_env.close()

    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = VecMonitor(SubprocVecEnv(env_fns)) # records episode reward/length for tensorboard, obs/rewards pass through unchanged
    policy_kwargs = dict(features_extractor_class=Agent)
    
    load_existing = False  # switch to False to train from scratch

    if load_existing:
        model = DQN.load(f"{loading_location}/DQN_scratch",env=env,tensorboard_log="./tensorboard_logs/")
        model.load_replay_buffer(f"{loading_location}/DQN_replay_buffer.pkl")
    else:
        model = DQN(
            "MultiInputPolicy", # MultiInputPolicy for a dict observation space and MlpPolicy for a flat observation space
            env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            gamma=config["gamma"], # ~N=100 => g=0.99, now 41x41 -> N=440 => g= 0.9977
            target_update_interval=config["target_update_interval"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            verbose=0,
            policy_kwargs=policy_kwargs, # CNN as feature extractor!!
            tensorboard_log=f"{saving_location}/tensorboard", # live curves: tensorboard --logdir ../trained-agents
        )

    """WANDB below here: if needed"""
    # INIT (insert here)
    # custom_callback = CustomWandbCallback()  # For your mean_q logging

    max_steps = config["max_steps"]
    checkpoint_steps = config["checkpoint_steps"] // NUM_ENVS # save the model every n steps, adjusted for number of parallel envs
    stats_steps = config["stats_steps"] // NUM_ENVS # save episode stats every n steps (each file holds all episodes so far)
    # --- get max_return safely before SubprocVecEnv ---

    stats_callback = EpisodeStatsCallback(max_episode_length=MAX_EPS_LEN,save_freq=stats_steps, NUM_ENVS=NUM_ENVS,address=f"{saving_location}/episode_stats",N_states=N_STATES,GP=GP) # save every n steps, max episode length for _on_step being called only at the end of every episode
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_steps,save_path=f"{saving_location}/",name_prefix="DQN_checkpoint",save_replay_buffer=False) # True if desired

    # --- Saving a big DICT with all hyperparameters:---
    build_training_config(model,AGENT_TYPE,keys_you_want_to_keep,agent_reward_path,max_steps,MAX_EPS_LEN,checkpoint_steps,saving_location,kernel_config=kernel_config if GP else None)

    
    # -- TRAINING --
    model.learn(total_timesteps=max_steps, callback=[stats_callback, checkpoint_callback], progress_bar=True) #,wandb_callback,custom_callback

    # --SAVING--

    # - stats -
    stats = stats_callback.get_stats()
    #stats["waypoints"] = list(range(0, model.num_timesteps+1, checkpoint_steps))
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # - model -
    model.save(f"{saving_location}/DQN_scratch")
    # - Replay Buffer -
    #model.save_replay_buffer(f"{saving_location}/DQN_replay_buffer.pkl")
   
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Optional but safe for macOS
    main()