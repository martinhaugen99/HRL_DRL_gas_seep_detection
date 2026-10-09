FULL_CONFIG = {
    "max_episode_length": 460,
    "num_envs": 24,
    "gp": True,
    "name": "NEW_plume_GP_clipped_RWD_min_steps_in_plume",
    "learning_rate": 1e-4,
    "buffer_size": 800_000,
    "learning_starts": 400_000,
    "batch_size": 128,
    "gamma": 0.9945,
    "target_update_interval": 50_000,
    "train_freq": 4,
    "gradient_steps": 4,
    "exploration_fraction": 0.8,
    "exploration_final_eps": 0.05,
    "max_steps": 60_000_000,
    "checkpoint_steps": 2_000_000,
    "stats_steps": 5_000_000,
}

GP_2D_CONFIG = {
    **FULL_CONFIG,
    "name": "DDQN_border_GP",
}

SMOKE_CONFIG = {
    **FULL_CONFIG,
    "max_episode_length": 256,
    "num_envs": 1,
    "gp": False,
    "name": "smoke_DDQN_no_GP",
    "buffer_size": 10_000,
    "learning_starts": 500,  # must be below max_steps, or the smoke run never trains
    "batch_size": 64,
    "target_update_interval": 1_000,
    "max_steps": 2_048,
    "checkpoint_steps": 2_048,
    "stats_steps": 2_048,
}
