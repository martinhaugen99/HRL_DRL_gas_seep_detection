# train_HRL_PPO.py

import argparse
import time
import os
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.agents.options import Option
from HUGIN_gym.agents.smdp_rollout_buffer import SMDPDictRolloutBuffer
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.agents.feature_extractor.agent0_HRL import AgentHRLExplore

from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.callbacks.OptionUsageCallback import OptionUsageCallback
from HUGIN_gym.callbacks.SMDPDiscountCallback import SMDPDiscountCallback
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper

from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.agents.term_fn.termination_fn_agent1 import termination_fn1
from HUGIN_gym.agents.term_fn.termination_fn_agent2 import termination_fn2

from HUGIN_gym.utils.io import append_resume_log, confirm_overwrite, find_latest_checkpoint
from HUGIN_gym.utils.build_train_config import (
    build_training_config_ppo,
    build_subpolicy_metadata,
)

# ----- Run presets: the full run, and a smoke run that tests the script in about a minute -----
RUN_CONFIG = {
    "name": "HRL_PPO",  # default run name, followed by the variant (--preset)
    "num_envs": 12,
    "n_steps": 512,  # per env
    "batch_size": 1024,
    "max_steps": 10_000_000,
    "checkpoint_steps": 2_050_000,  # also how often the episode stats are saved
}

SMOKE_CONFIG = {
    "name": "HRL_PPO_smoke",
    "num_envs": 2,
    "n_steps": 64,
    "batch_size": 64,
    "max_steps": 512,
    "checkpoint_steps": 256,
}

# ----- Meta agent variants, one per run tag (the first argument of fox_train_HRL.sh) -----
# option_lengths: the steps per decision the meta agent chooses from (the action picks sub-agent and length)
# smdp_discount: discount each decision by GAMMA**steps, needed when the lengths differ (SMDPDictRolloutBuffer)
# mean_option_reward: a decision's reward is the mean of its step rewards, with one fixed discount per decision
# decision_cost: subtracted per decision, so longer options pay off slightly
# meta_state: "full" (158 inputs, AgentHRLExplore), "minimal" (6 numbers) or "minimal_values" (6 + 3 sub-agent values)
PRESETS = {
    "untuned": {"option_lengths": [3], "smdp_discount": False, "mean_option_reward": False, "decision_cost": 0.0, "meta_state": "full"},
    "duration": {"option_lengths": [2, 5], "smdp_discount": True, "mean_option_reward": False, "decision_cost": 0.1, "meta_state": "full"},
    "duration_mean": {"option_lengths": [2, 5], "smdp_discount": False, "mean_option_reward": True, "decision_cost": 0.1, "meta_state": "full"},
    "minimal": {"option_lengths": [3], "smdp_discount": False, "mean_option_reward": False, "decision_cost": 0.0, "meta_state": "minimal"},
    "minimal_values": {"option_lengths": [3], "smdp_discount": False, "mean_option_reward": False, "decision_cost": 0.0, "meta_state": "minimal_values"},
    "minimal_values_duration": {"option_lengths": [2, 5], "smdp_discount": True, "mean_option_reward": False, "decision_cost": 0.1, "meta_state": "minimal_values"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a few PPO updates with 2 envs, to test the script")
    parser.add_argument("--preset", choices=PRESETS, default="untuned", help="meta agent variant (default: untuned)")
    parser.add_argument("--name", help="run name, i.e. the output folder in ../trained-agents (default: HRL_PPO[_smoke]_<preset>)")
    parser.add_argument("--resume", action="store_true", help="continue the run in ../trained-agents/<name> from its newest checkpoint")
    args = parser.parse_args()
    config = SMOKE_CONFIG if args.smoke else RUN_CONFIG
    variant = PRESETS[args.preset]

    NAME = args.name or f"{config['name']}_{args.preset}"
    saving_location = f"../trained-agents/{NAME}"

    if args.resume:
        checkpoint_path, checkpoint_step = find_latest_checkpoint(saving_location, "HRL_PPO_checkpoint")
    else:
        confirm_overwrite(saving_location)

    # ----- HRL / env configuration -----
    AGENT_TYPE = "META"
    SURVEY_STEPS = 480  # survey budget in base env steps: 160 decisions of 3 steps in the untuned run
    MAX_EPS_LEN = SURVEY_STEPS // min(variant["option_lengths"])  # in meta decisions, so the survey budget ends the episode
    NUM_ENVS = config["num_envs"]

    GP = True
    MULTIPLE_GAUSSIANS = [1, 1]
    ACCURACY_GOALS = [0.9, 0.9, 0.9]
    MAX_OPT_LENGTH = 3
    GAMMA = 0.998  # per primitive step: discounts the rewards inside an option; the meta agent gets GAMMA**MAX_OPT_LENGTH per decision (GAMMA**steps with smdp_discount)

    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
        # "Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},
    }

    def episode_length_schedule(local_step):
        # Placeholder for more complex logic
        return MAX_EPS_LEN

    # ----- Sub-policies (still DQN) -----
    sub_policies = [
        {
            "id": "option1",
            "name": "spatial exploration",
            "short_name": "space",  # in the TensorBoard tags of OptionUsageCallback
            "dir": "../trained-agents/PPO_space_30M",
            "model_file": "PPO_scratch.zip",
            "termination_fn": termination_fn1,
            "obs_keys": [
                "obs_state",
                "visited_maps_downsampled",
                "space_coverage",
                #"VAR_visited_downsampled",
            ],
        },
        {
            "id": "option2",
            "name": "gas plume exploration",
            "short_name": "plume",
            "dir": "../trained-agents/PPO_compare_100M_4259607",#PPO_plume_GP_clipped_RWD_no_distance_to_max
            "model_file": "PPO_scratch.zip",
            "termination_fn": termination_fn2,
            "obs_keys": [
                "obs_state",
                #"visited_maps_downsampled",
                #"GT_c_over_threshold_maps_downsampled",
                "local_GT",
                "plume_coverage",
                "space_coverage",
                "distance_from_max",
            ],
        },
        {
            "id": "option3",
            "name": "gas border exploration",
            "short_name": "border",
            "dir": "../trained-agents/PPO_border_4280885",#PPO_border_GP_not_clipped_RWD_no_distance_to_max
            "model_file": "PPO_scratch.zip",
            "termination_fn": termination_fn2,
            "obs_keys": [
                "obs_state",
                #"visited_maps_downsampled",
                #"GT_c_around_threshold_maps_downsampled",
                "local_GT_around",
                "border_coverage",
                "space_coverage",
                "distance_from_max",
            ],
        },
    ]

    HRL_sub_policy_data = build_subpolicy_metadata(sub_policies)

    # ----- Env factory -----
    def make_env():
        def _init():
            base_env = gym.make("HUGIN-v0")
            base_env = base_env.unwrapped  # REMOVE default TimeLimit wrapper

            # Base env config
            base_env.use_c_map = True
            base_env.GP_ON = GP
            base_env.train = True
            base_env.HRL = True
            base_env.agent_type = AGENT_TYPE
            base_env.multiple_gaussians = MULTIPLE_GAUSSIANS
            base_env.spawn_at_survey_start = True  # start in a corner like a survey, not near the plume
            base_env.vary_plume_size = True  # sigma from [2.0, 5.0], as in the plume agent's training

            # Load sub-policies as Options
            options = []
            for sp in sub_policies:
                model_path = os.path.join(sp["dir"], sp["model_file"])
                policy = PPO.load(model_path)

                option = Option(
                    policy,
                    sp["termination_fn"],
                    sp["name"],
                    obs_keys=sp["obs_keys"],
                )
                options.append(option)

            # Optional GP wrapper around base env
            if GP:
                gp_env = GPWrapper(
                    base_env,
                    kernels_config=kernel_config,
                    gp_visualise=False,
                    HRL=True,
                    SUB_AGENT_TRAIN_ON_GP=False,  # HRL-level training only
                )
            else:
                gp_env = base_env

            # Hierarchical environment with options
            env = HierarchicalEnv(
                base_env=gp_env,
                options=options,
                gamma=GAMMA,
                MAX_OPT_LENGTH=MAX_OPT_LENGTH,
                ACCURACY_GOALS=ACCURACY_GOALS,
                GP=GP,
                max_primitive_steps=SURVEY_STEPS,
                **variant,
            )

            # Control episode length
            env = DynamicEpisodeLengthWrapper(
                env,
                schedule_fn=episode_length_schedule,
            )

            return env

        return _init

    # ----- Extract some metadata (states, keys) -----
    # Create one temp env to extract shape safely
    temp_env = make_env()()
    keys_you_want_to_keep = list(temp_env.observation_space.keys())  # what the meta agent observes
    N_STATES = temp_env.env.base_env.unwrapped.max_return
    temp_env.close()

    # ----- Vectorized env -----
    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = VecMonitor(SubprocVecEnv(env_fns))  # episode reward and length in TensorBoard (rollout/ep_rew_mean, ep_len_mean)

    # ----- PPO meta-controller -----
    if variant["meta_state"] == "full":
        policy_kwargs = dict(
            features_extractor_class=AgentHRLExplore,
        )
    else:
        policy_kwargs = {}  # a few numbers: SB3's default extractor concatenates them
    # one discount per decision when the options run different numbers of steps
    buffer_kwargs = {"rollout_buffer_class": SMDPDictRolloutBuffer} if variant["smdp_discount"] else {}

    if args.resume:
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=f"{saving_location}/tensorboard",
            **buffer_kwargs,
        )
    else:
        model = PPO(
            "MultiInputPolicy",  # dict observation space
            env,
            learning_rate=3e-4,
            n_steps=config["n_steps"],  # per env
            batch_size=config["batch_size"],  # effective batch = n_steps * NUM_ENVS
            n_epochs=4,
            # exact per decision when every option that does not end the episode runs MAX_OPT_LENGTH steps (HierarchicalEnv
            # checks this). With smdp_discount the buffer uses GAMMA**steps instead, and this only bootstraps truncated episodes.
            gamma=GAMMA**MAX_OPT_LENGTH,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{saving_location}/tensorboard",  # live curves: tensorboard --logdir ../trained-agents
            **buffer_kwargs,
        )

    # ----- Training setup & callbacks -----
    max_steps = config["max_steps"]
    checkpoint_steps = config["checkpoint_steps"] // NUM_ENVS  # in steps per env

    stats_callback = EpisodeStatsCallback(
        max_episode_length=MAX_EPS_LEN,
        save_freq=checkpoint_steps,
        NUM_ENVS=NUM_ENVS,
        address=f"{saving_location}/episode_stats",
        N_states=N_STATES,
        GP=GP,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_steps,
        save_path=f"{saving_location}/",
        name_prefix="HRL_PPO_checkpoint",
        save_replay_buffer=False,  # PPO has no replay buffer
    )

    action_names = HierarchicalEnv.action_names([sp["short_name"] for sp in sub_policies], variant["option_lengths"])
    callbacks = [stats_callback, checkpoint_callback, OptionUsageCallback(action_names)]
    if variant["smdp_discount"]:
        callbacks.append(SMDPDiscountCallback())

    if args.resume:  # keep the config (and start time) the run was started with
        append_resume_log(
            saving_location,
            f"resumed from {checkpoint_path.name}, {stats_callback.load_snapshot(up_to_step=checkpoint_step)}",
        )
    else:
        # ----- Save training config (PPO version) -----
        build_training_config_ppo(
            model,
            AGENT_TYPE,
            keys_you_want_to_keep,
            agent_reward_path=None,  # HRL-level reward is likely internal to HierarchicalEnv
            max_steps=max_steps,
            max_eps_len=MAX_EPS_LEN,
            checkpoint_steps=checkpoint_steps,
            saving_location=saving_location,
            kernel_config=kernel_config if GP else None,
            MAX_OPT_LENGTH=MAX_OPT_LENGTH,
            MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,
            sub_policy_metadata=HRL_sub_policy_data,
            hrl_variant={"preset": args.preset, "survey_steps": SURVEY_STEPS, **variant},
        )

    # ----- Train meta-controller -----
    # a resumed run continues its step count and tensorboard curves up to max_steps
    model.learn(
        total_timesteps=max_steps - model.num_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )

    # ----- Save stats and model -----
    stats = stats_callback.get_stats()
    # Optional: add waypoints if you still want them
    stats["waypoints"] = list(range(0, model.num_timesteps + 1, checkpoint_steps))
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    model.save(f"{saving_location}/HRL_PPO_scratch")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
