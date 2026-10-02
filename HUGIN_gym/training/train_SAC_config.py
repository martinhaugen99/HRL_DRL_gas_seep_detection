FULL_CONFIG = {
    "max_episode_length": 460,
    "num_envs": 12,
    "gp": False,
    "name": "SAC_space_GT_not_clipped_RWD",
    "learning_rate": 5e-5,
    "buffer_size": 900_000,
    "batch_size": 256,
    "gamma": 0.997,
    "tau": 0.01,
    "train_freq": 4,
    "gradient_steps": 4,
    "max_steps": 10_000_000,
    "checkpoint_steps": 2_000_000,
    "stats_steps": 5_000_000,
}

GP_2D_CONFIG = {
    **FULL_CONFIG,
    "gp": True,
    "name": "2D_SAC_border_GP_10M",
}

SMOKE_CONFIG = {
    **FULL_CONFIG,
    "max_episode_length": 256,
    "num_envs": 1,
    "gp": False,
    "name": "smoke_SAC_no_GP",
    "buffer_size": 10_000,
    "batch_size": 64,
    "max_steps": 2_048,
    "checkpoint_steps": 2_048,
    "stats_steps": 2_048,
}
