# utils/build_train_config.py
import datetime
import os
from HUGIN_gym.utils.load_reward_function import load_reward_function_source
from HUGIN_gym.utils.io import save_training_config,save_training_config_json

def build_training_config(model, agent_type, keys_to_keep, agent_reward_path,
                          max_steps, max_eps_len, checkpoint_steps, saving_location, kernel_config=None, MAX_OPT_LENGTH=None,MULTIPLE_GAUSSIANS=None,sub_policy_metadata=None ):
    """
    Build a dictionary with all training configuration and hyperparameters.
    """
    training_config = {
        "RUN_INFO": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AGENT_TYPE": agent_type,
            "FEATURE_EXTRACTOR": model.policy.features_extractor_class.__name__,
            "filtered_obs_keys": keys_to_keep,
            "TOTAL_TIMESTEPS": max_steps,
            "NUM_ENVS": model.get_env().num_envs if hasattr(model.get_env(), "num_envs") else 1,
        },

        "DQN_HYPERPARAMETERS": {
            "learning_rate": model.learning_rate,
            "buffer_size": model.buffer_size,
            "learning_starts": model.learning_starts,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "target_update_interval": model.target_update_interval,
            "train_freq": model.train_freq,
            "gradient_steps": model.gradient_steps,
            "exploration_fraction": model.exploration_fraction,
            "exploration_final_eps": model.exploration_final_eps,
        },

        "ENVIRONMENT": {
            "max_episode_length": max_eps_len,
            "checkpoint_steps": checkpoint_steps,
        },

        "POLICY": {
            "architecture": str(model.policy),
        },

        "REWARD_FUNCTION_SOURCE": load_reward_function_source(agent_reward_path)
    }

    if any(v is not None for v in [kernel_config, MAX_OPT_LENGTH, MULTIPLE_GAUSSIANS]):
        training_config["HRL_HYPERPARAMETERS"] = {}

        if kernel_config is not None:
            training_config["HRL_HYPERPARAMETERS"]["kernel_config"] = kernel_config

        if MAX_OPT_LENGTH is not None:
            training_config["HRL_HYPERPARAMETERS"]["MAX_OPT_LENGTH"] = MAX_OPT_LENGTH

        if MULTIPLE_GAUSSIANS is not None:
            training_config["HRL_HYPERPARAMETERS"]["MULTIPLE_GAUSSIANS"] = MULTIPLE_GAUSSIANS

    if sub_policy_metadata is not None:
        training_config["SUB_POLICIES"] = sub_policy_metadata
    os.makedirs(saving_location, exist_ok=True)
    config_path = os.path.join(saving_location, "training_config.txt")
    save_training_config(config_path, training_config)
    save_training_config_json(f"{saving_location}/training_config.json", training_config)

    pass

import json

def load_subpolicy_config(directory):
    """
    Try to load a config file from a trained agent directory.
    Supports JSON and TXT.
    """
    possible_files = [
        "training_config.json",
        #"training_config.txt"
    ]

    for fname in possible_files:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            try:
                if fname.endswith(".json"):
                    with open(path, "r") as f:
                        return json.load(f)
                else:
                    with open(path, "r") as f:
                        return f.read()
            except Exception as e:
                return f"Failed to load config: {e}"

    return "No config file found"


def build_training_config_ppo(
    model,
    agent_type,
    keys_to_keep,
    agent_reward_path,
    max_steps,
    max_eps_len,
    checkpoint_steps,
    saving_location,
    kernel_config=None,
    MAX_OPT_LENGTH=None,
    MULTIPLE_GAUSSIANS=None,
    sub_policy_metadata=None,
    hrl_variant=None,
):
    """
    Build a dictionary with all training configuration and hyperparameters
    for a PPO model (Stable-Baselines3).

    Usage: call this instead of build_training_config when training PPO.
    hrl_variant: settings of the HRL meta agent variant (preset), stored under HRL_VARIANT.
    """
    # Handle possible schedule functions: store repr instead of callable
    def _as_serializable(value):
        if callable(value):
            return repr(value)
        return value

    training_config = {
        "RUN_INFO": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AGENT_TYPE": agent_type,
            "FEATURE_EXTRACTOR": model.policy.features_extractor_class.__name__,
            "filtered_obs_keys": keys_to_keep,
            "TOTAL_TIMESTEPS": max_steps,
            "NUM_ENVS": model.get_env().num_envs
            if hasattr(model.get_env(), "num_envs")
            else 1,
        },
        "PPO_HYPERPARAMETERS": {
            "learning_rate": _as_serializable(model.learning_rate),
            "n_steps": getattr(model, "n_steps", None),
            "batch_size": getattr(model, "batch_size", None),
            "n_epochs": getattr(model, "n_epochs", None),
            "gamma": getattr(model, "gamma", None),
            "gae_lambda": getattr(model, "gae_lambda", None),
            "clip_range": _as_serializable(getattr(model, "clip_range", None)),
            "ent_coef": getattr(model, "ent_coef", None),
            "vf_coef": getattr(model, "vf_coef", None),
            "max_grad_norm": getattr(model, "max_grad_norm", None),
        },
        "ENVIRONMENT": {
            "max_episode_length": max_eps_len,
            "checkpoint_steps": checkpoint_steps,
        },
        "POLICY": {
            "architecture": str(model.policy),
        },
        "REWARD_FUNCTION_SOURCE": load_reward_function_source(agent_reward_path),
    }

    if any(v is not None for v in [kernel_config, MAX_OPT_LENGTH, MULTIPLE_GAUSSIANS]):
        training_config["HRL_HYPERPARAMETERS"] = {}

        if kernel_config is not None:
            training_config["HRL_HYPERPARAMETERS"]["kernel_config"] = kernel_config

        if MAX_OPT_LENGTH is not None:
            training_config["HRL_HYPERPARAMETERS"]["MAX_OPT_LENGTH"] = MAX_OPT_LENGTH

        if MULTIPLE_GAUSSIANS is not None:
            training_config["HRL_HYPERPARAMETERS"]["MULTIPLE_GAUSSIANS"] = MULTIPLE_GAUSSIANS

    if sub_policy_metadata is not None:
        training_config["SUB_POLICIES"] = sub_policy_metadata

    if hrl_variant is not None:
        training_config["HRL_VARIANT"] = hrl_variant

    os.makedirs(saving_location, exist_ok=True)
    config_path = os.path.join(saving_location, "training_config.txt")
    save_training_config(config_path, training_config)
    save_training_config_json(
        f"{saving_location}/training_config.json", training_config
    )




def build_training_config_sac(
    model,
    agent_type,
    keys_to_keep,
    agent_reward_path,
    max_steps,
    max_eps_len,
    checkpoint_steps,
    saving_location,
    kernel_config=None,
    MAX_OPT_LENGTH=None,
    MULTIPLE_GAUSSIANS=None,
    sub_policy_metadata=None,
):
    """
    Build a dictionary with all training configuration and hyperparameters
    for a SAC model (Stable-Baselines3).

    Usage: call this instead of build_training_config / build_training_config_ppo
    when training SAC.
    """

    def _as_serializable(value):
        # Turn schedules / callables into strings
        if callable(value):
            return repr(value)
        return value

    training_config = {
        "RUN_INFO": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AGENT_TYPE": agent_type,
            "FEATURE_EXTRACTOR": model.policy.features_extractor_class.__name__,
            "filtered_obs_keys": keys_to_keep,
            "TOTAL_TIMESTEPS": max_steps,
            "NUM_ENVS": model.get_env().num_envs
            if hasattr(model.get_env(), "num_envs")
            else 1,
        },
        "SAC_HYPERPARAMETERS": {
            "learning_rate": _as_serializable(getattr(model, "learning_rate", None)),
            "buffer_size": getattr(model, "buffer_size", None),
            "batch_size": getattr(model, "batch_size", None),
            "gamma": getattr(model, "gamma", None),
            "tau": getattr(model, "tau", None),
            "train_freq": getattr(model, "train_freq", None),
            "gradient_steps": getattr(model, "gradient_steps", None),
            "ent_coef": _as_serializable(getattr(model, "ent_coef", None)),
            "target_update_interval": getattr(
                model, "target_update_interval", None
            ),  # SB3 SAC has this
        },
        "ENVIRONMENT": {
            "max_episode_length": max_eps_len,
            "checkpoint_steps": checkpoint_steps,
        },
        "POLICY": {
            "architecture": str(model.policy),
        },
        "REWARD_FUNCTION_SOURCE": load_reward_function_source(agent_reward_path),
    }

    if any(v is not None for v in [kernel_config, MAX_OPT_LENGTH, MULTIPLE_GAUSSIANS]):
        training_config["HRL_HYPERPARAMETERS"] = {}

        if kernel_config is not None:
            training_config["HRL_HYPERPARAMETERS"]["kernel_config"] = kernel_config

        if MAX_OPT_LENGTH is not None:
            training_config["HRL_HYPERPARAMETERS"]["MAX_OPT_LENGTH"] = MAX_OPT_LENGTH

        if MULTIPLE_GAUSSIANS is not None:
            training_config["HRL_HYPERPARAMETERS"]["MULTIPLE_GAUSSIANS"] = MULTIPLE_GAUSSIANS

    if sub_policy_metadata is not None:
        training_config["SUB_POLICIES"] = sub_policy_metadata

    os.makedirs(saving_location, exist_ok=True)
    config_path = os.path.join(saving_location, "training_config.txt")
    save_training_config(config_path, training_config)
    save_training_config_json(
        f"{saving_location}/training_config.json", training_config
    )


# ---- existing helpers for HRL configs ----

def load_subpolicy_config(directory):
    """
    Try to load a config file from a trained agent directory.
    Supports JSON (and TXT if you re-enable it).
    """
    possible_files = [
        "training_config.json",
        # "training_config.txt"
    ]

    for fname in possible_files:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            try:
                if fname.endswith(".json"):
                    with open(path, "r") as f:
                        return json.load(f)
                else:
                    with open(path, "r") as f:
                        return f.read()
            except Exception as e:
                return f"Failed to load config: {e}"

    return "No config file found"

def build_subpolicy_metadata(sub_policies):
    metadata = {}

    for sp in sub_policies:
        directory = sp["dir"]
        model_path = os.path.join(directory, sp["model_file"])

        metadata[sp["id"]] = {
            "name": sp["name"],
            "directory": directory,
            "model_path": model_path,
            "obs_keys": sp["obs_keys"],
            "config": load_subpolicy_config(directory)
        }

    return metadata