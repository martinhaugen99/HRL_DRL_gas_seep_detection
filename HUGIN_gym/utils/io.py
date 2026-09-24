import os
import sys
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFrequencyUnit

def confirm_overwrite(directory: str):
    if os.path.exists(directory) and os.listdir(directory):
            print(f"\n⚠️  Warning: The directory '{directory}' already exists and is not empty.")
            if not sys.stdin.isatty(): # batch job (e.g. Slurm): nobody can answer, so never overwrite
                print("❌ No terminal to confirm overwriting. Aborting; choose a new run name.")
                sys.exit(1)
            while True:
                response = input("Do you want to continue and potentially overwrite files? [y/n]: ").strip().lower()
                if response == "y":
                    print("✅ Continuing training...\n")
                    return
                elif response == "n":
                    print("❌ Aborting to prevent overwrite.")
                    sys.exit(0)
                else:
                    print("Please enter 'y' or 'n'.")

def save_training_config(filepath, config_dict):
    with open(filepath, "w") as f:
        for key, value in config_dict.items():
            if isinstance(value, dict):
                f.write("\n--------------------------------------------------\n")
                f.write(f"{key}\n")
                f.write("--------------------------------------------------\n\n")
                for sub_key, sub_value in value.items():
                    f.write(f"{sub_key}: {sub_value}\n")
            elif key == "SEPARATOR":
                f.write("\n--------------------------------------------------\n")
            elif key == "REWARD_FUNCTION_SOURCE":
                f.write("\n--------------------------------------------------\n")
                f.write(f"{key}\n")
                f.write("--------------------------------------------------\n\n")
                f.write(f"{value}\n")
            else:
                f.write(f"{key}: {value}\n")


def make_json_serializable(obj):
    """Recursively convert objects to something JSON can handle."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, TrainFrequencyUnit):
        return str(obj)
    elif isinstance(obj, BaseCallback):
        return obj.__class__.__name__
    else:
        try:
            json.dumps(obj)  # test if serializable
            return obj
        except TypeError:
            return str(obj)

def save_training_config_json(filepath, config_dict):
    serializable_config = make_json_serializable(config_dict)
    with open(filepath, "w") as f:
        json.dump(serializable_config, f, indent=4)


def load_training_config_from_model(model_path):
    """
    Loads training config JSON from the same directory as the model.
    """
    config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config