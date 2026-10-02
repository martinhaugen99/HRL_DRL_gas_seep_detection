FULL_CONFIG = {
    "max_episode_length": 8000,
    "num_envs": 12,
    "gp": True,
    "name": "3D_PPO_border_GP_sub_task_completion_41x41x41_7k_GP",
    "n_steps": 2048,
    "batch_size": 1024,
    "n_epochs": 4,
    "max_steps": 100_000_000,
    "checkpoint_steps": 10_500_020,
}

PAPER_CONFIG = {
    **FULL_CONFIG,
    "max_episode_length": 460,
    "gp": False,
    "name": "paper_2D_PPO_space_10M",
    "n_steps": 512,
    "max_steps": 10_000_000,
    "checkpoint_steps": 1_000_000,
}

GP_2D_CONFIG = {
    **PAPER_CONFIG,
    "gp": True,
    "name": "2D_PPO_border_GP_10M",
}

SMOKE_CONFIG = {
    "max_episode_length": 256,
    "num_envs": 1,
    "gp": False,
    "name": "smoke_3D_PPO_space_no_GP",
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 2,
    "max_steps": 2_048,
    "checkpoint_steps": 2_048,
}
