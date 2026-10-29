import torch as th


class Option:
    def __init__(self, policy, termination_fn, name="option", obs_keys=None,obs_transform=None):
        self.policy = policy
        self.termination_fn = termination_fn
        self.name = name
        self.obs_keys = obs_keys
        self.obs_transform = obs_transform

    def _filter_obs(self, obs):
        if self.obs_keys is None:
            return obs
        return {k: obs[k] for k in self.obs_keys}

    def act(self, state):
        filtered_state = self._filter_obs(state)
        if self.obs_transform is not None:
            filtered_state = self.obs_transform(filtered_state)
        action, _ = self.policy.predict(filtered_state, deterministic=True)
        return action

    def value(self, state) -> float:
        """The sub-agent's own estimate of its return from this state (its PPO critic)."""
        filtered_state = self._filter_obs(state)
        if self.obs_transform is not None:
            filtered_state = self.obs_transform(filtered_state)
        obs_tensor, _ = self.policy.policy.obs_to_tensor(filtered_state)
        with th.no_grad():
            return float(self.policy.policy.predict_values(obs_tensor).item())

    def should_terminate(self, state, next_state, threshold1=None):
        state_f = self._filter_obs(state)
        next_state_f = self._filter_obs(next_state)
        return self.termination_fn(state_f, next_state_f, threshold1=threshold1)
    
    

 