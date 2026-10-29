import gymnasium as gym
import numpy as np
import time

class HierarchicalEnv(gym.Env):
    META_STATES = ("full", "minimal", "minimal_values")

    def __init__(self, base_env, options, gamma=0.99,MAX_OPT_LENGTH=5,ACCURACY_GOALS = None,GP=True,
                 option_lengths=None, max_primitive_steps=None, smdp_discount=False, mean_option_reward=False,
                 decision_cost=0.0, meta_state="full"):
        """
        option_lengths: the numbers of steps a sub-agent can run per meta decision (default [MAX_OPT_LENGTH]).
            The meta agent picks the sub-agent and the length together: action = option index * len(option_lengths) + length index.
        max_primitive_steps: survey budget in base env steps. The episode is truncated when it is used up.
        smdp_discount: options may run different numbers of steps. Each decision reports its discount
            gamma**option_length in info["discount"], which SMDPDiscountCallback passes to SMDPDictRolloutBuffer.
        mean_option_reward: reward a decision with the mean of its step rewards instead of their sum.
            The goal bonuses and the penalties still count in full.
        decision_cost: subtracted once per decision, so fewer and longer options pay off slightly.
        meta_state: what the meta agent observes. "full": the base env's observation. "minimal": time used, the three
            coverages, plume found and the GP uncertainty. "minimal_values": minimal plus the sub-agents' value estimates.
        """
        super().__init__()
        self.base_env = base_env # gp_env lol
        #self.base_env.unwrapped.HRL = True
        self.options = options
        self.gamma = gamma
        self.option_lengths = list(option_lengths) if option_lengths else [MAX_OPT_LENGTH]
        self.max_primitive_steps = max_primitive_steps
        self.smdp_discount = smdp_discount
        self.mean_option_reward = mean_option_reward
        self.decision_cost = decision_cost
        self.meta_state = meta_state
        if smdp_discount and mean_option_reward:
            raise ValueError("Choose smdp_discount or mean_option_reward: the mean reward uses one fixed discount per decision.")
        if len(self.option_lengths) > 1 and not (smdp_discount or mean_option_reward):
            raise ValueError("Options of different lengths need smdp_discount or mean_option_reward.")
        if meta_state not in self.META_STATES:
            raise ValueError(f"meta_state must be one of {self.META_STATES}, not {meta_state!r}")
        if meta_state != "full" and (max_primitive_steps is None or not GP):
            raise ValueError("The minimal meta states need max_primitive_steps (time used) and the GP (plume found, uncertainty).")
        self.accuracy_goals = ACCURACY_GOALS if ACCURACY_GOALS else self.base_env.accuracy_goals
        self.base_env.accuracy_goals = self.accuracy_goals# NOTE, for the expert training we used 0.9, but for HRL we set the level a bit lower

        # Action space: choose option and its length
        self.action_space = gym.spaces.Discrete(len(options) * len(self.option_lengths))

        # Observation space: the base env's, or the minimal meta state
        self.observation_space = self._meta_observation_space(base_env.observation_space)
        self.current_obs = None
        self.primitive_steps = 0

        self.visualise_env = False
        self.no_progress_steps=0
        self.once_per_episode=True # for pure GT run, GT HRL and GT subagents
        self.GP=GP
        self.turn_counter = 0
        self.test_HRL = False
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self.current_obs = obs
        self.no_progress_steps=0
        self.once_per_episode=True # for pure GT run, GT HRL and GT subagents
        self.turn_counter = 0
        self.primitive_steps = 0
        return self._meta_observation(obs), info

    @staticmethod
    def action_names(option_names, option_lengths):
        """Names of the meta actions in action order, e.g. space_2, space_5, plume_2, ..."""
        return [f"{name}_{length}" for name in option_names for length in option_lengths]

    def _decode_action(self, meta_action):
        """Option index and number of steps of a meta action."""
        option_idx, length_idx = divmod(int(meta_action), len(self.option_lengths))
        return option_idx, self.option_lengths[length_idx]

    def _meta_observation_space(self, base_space):
        if self.meta_state == "full":
            return base_space
        unit = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        keys = ("time_used", "space_coverage", "plume_coverage", "border_coverage", "plume_found", "gp_uncertainty")
        spaces = {key: unit for key in keys}
        if self.meta_state == "minimal_values":
            spaces["subagent_values"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.options),), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def _meta_observation(self, obs):
        """What the meta agent sees of the full observation obs (the sub-agents always get the full one)."""
        if self.meta_state == "full":
            return obs
        base = self.base_env.unwrapped
        meta_obs = {
            "time_used": np.array([self.primitive_steps / self.max_primitive_steps], dtype=np.float32),
            "space_coverage": obs["space_coverage"],
            "plume_coverage": obs["plume_coverage"],
            "border_coverage": obs["border_coverage"],
            # before the GP finds a plume, the plume belief is only the corner cells, so the coverages show 1.0
            "plume_found": np.array([float(base.maxN_over_thresh > self.base_env.N_corners)], dtype=np.float32),
            "gp_uncertainty": np.array([min(base.MEAN_uncertainty, 1.0)], dtype=np.float32),
        }
        if self.meta_state == "minimal_values":
            values = np.array([option.value(obs) for option in self.options], dtype=np.float32)
            # symlog: the sub-agents' returns have different scales
            meta_obs["subagent_values"] = np.sign(values) * np.log1p(np.abs(values))
        return meta_obs

    def step(self, meta_action):
        option_idx, target_length = self._decode_action(meta_action)
        option = self.options[option_idx]

        shaping_reward = 0 # the per-step rewards of the meta agent, discounted inside the option
        event_reward = 0 # goal bonuses (40) and border penalties (-4) from the base env
        discount = 1
        done = False
        trunc = False
        #trajectory_info = []

        obs = self.current_obs
        start_step = self.primitive_steps

        #trajectory_info.append(state)
        option_length = 0
        self.base_env.option_length = option_length

        while True:

            action = option.act(obs)
            action = int(action)
            option_length +=1
            self.primitive_steps += 1
            self.base_env.option_length = option_length
            next_obs, reward_debug , done, trunc, info = self.base_env.step(action) # we are not interested in the reward from the base env
            #trajectory_info.append(info)
            if self.test_HRL and self.base_env.env.agent_turns:
                self.turn_counter+=1
            if self.GP==False:
                meta_reward,meta_done = self._compute_meta_subgoal_reward(next_obs)
                if meta_reward != 0.0:
                    done = meta_done
                    reward_debug = meta_reward

            if self.visualise_env:
                self.base_env.unwrapped.step_sim() # step_sim only uses the self.state # NOTE: only works IF GP ACTIVATED
                #time.sleep(0.07) #to make vis better
            
            current_visited_sum = np.sum(self.base_env.unwrapped.state["visited_map"])
            current_c_over_sum =(np.sum(self.base_env.unwrapped.state["c_over_threshold_map"]))
            current_c_around_sum =(np.sum(self.base_env.unwrapped.state["c_around_threshold_map"]))


            if  (self.base_env.unwrapped.old_visited_sum==current_visited_sum) and (current_c_over_sum ==self.base_env.unwrapped.old_c_over_sum) and (current_c_around_sum==self.base_env.unwrapped.old_c_around_sum):
                self.no_progress_steps+=1
            else:
                self.no_progress_steps=0
            
            # -- Reward Design --
            #       SPACE
            reward = 0
            if option_idx == 0: 
                if (current_c_over_sum-self.base_env.unwrapped.old_c_over_sum)>0.0 or (current_c_around_sum-self.base_env.unwrapped.old_c_around_sum)>0.0 or (next_obs["plume_coverage"][0])<self.accuracy_goals[1] or (next_obs["border_coverage"][0])<self.accuracy_goals[1]:
                    reward = 0
                else:
                    #print(f"New_SUM{new_sum}")
                    reward = (current_visited_sum-self.base_env.unwrapped.old_visited_sum)/6
                    #reward = np.clip((current_visited_sum-self.base_env.old_visited_sum)/4,0,1) #+ (self.base_env.unwrapped.max_return-new_sum)/self.base_env.unwrapped.max_return # encouraging exploration
            #       PLUME
            elif option_idx == 1: 
                if ((current_c_around_sum-self.base_env.unwrapped.old_c_around_sum)>0.0):
                    
                    reward = (current_c_around_sum-self.base_env.unwrapped.old_c_around_sum) / 6  + (current_visited_sum-self.base_env.unwrapped.old_visited_sum) / 6 #+ (self.base_env.unwrapped.GP_maxN_over_thresh-new_sum)/self.base_env.unwrapped.GP_maxN_over_thresh # encouraging exploration
                    #reward*=2/3
                else:
                    #print(f"New_SUM{new_sum}")
                    #reward = np.clip((np.sum(current_visited_sum-self.base_env.old_visited_sum))/4,0,1)/2
            #       BORDER
                    #print("no new >c")
                    reward = (current_visited_sum-self.base_env.unwrapped.old_visited_sum)/6/2
            else: 
                if (current_c_around_sum-self.base_env.unwrapped.old_c_around_sum)>0.0:
                    
                    reward = (current_c_around_sum-self.base_env.unwrapped.old_c_around_sum) / 6  + (current_visited_sum-self.base_env.unwrapped.old_visited_sum) / 6 #+ (self.base_env.unwrapped.GP_maxN_around_thresh-new_sum)/self.base_env.unwrapped.GP_maxN_around_thresh
                    #reward*=2/3
                else:
                    #print("no new c~")
                    
                    #print(f"New_SUM{new_sum}")
                    #reward = np.clip((current_visited_sum-self.base_env.old_visited_sum)/4,0,1)/2
                    reward = (current_visited_sum-self.base_env.unwrapped.old_visited_sum)/6/2
            #print(reward)
            reward -= 1.0 # TEMP DISCOUNT :encouraging faster option changes
            
            if reward_debug==40.0 or reward_debug==-4.0:
                event_reward += discount * reward_debug # replaces the step's reward
            else:
                shaping_reward += discount * reward
            discount *= self.gamma

            if self.max_primitive_steps is not None and self.primitive_steps >= self.max_primitive_steps:
                trunc = True # survey budget used up

            if done or trunc or self.no_progress_steps>2 or option_length >=target_length: #! >1
                if not (done or trunc) and option_length < target_length and not (self.smdp_discount or self.mean_option_reward):
                    # the meta agent discounts each decision by gamma**MAX_OPT_LENGTH, which is only exact for full-length options
                    raise RuntimeError(
                        f"Option ended after {option_length} of {target_length} steps while the episode goes on. "
                        "The meta agent's discount gamma**MAX_OPT_LENGTH needs every such option to run MAX_OPT_LENGTH "
                        "steps (options end after 3 steps without progress, so use MAX_OPT_LENGTH <= 3), "
                        "or smdp_discount=True for a discount of gamma**option_length per decision."
                    )
                if self.mean_option_reward:
                    total_reward = shaping_reward / option_length + event_reward
                else:
                    total_reward = shaping_reward + event_reward
                if self.no_progress_steps>2:
                    total_reward-=2.0 # penalty for no progress!
                total_reward -= self.decision_cost

                #option.should_terminate(obs,next_obs, threshold1 = self.base_env.c_threshold)
                # if self.visualise_env:
                #     self.base_env.unwrapped.renderer.reset()
                obs = next_obs
                self.current_obs = obs
                self.no_progress_steps=0
                break

            obs = next_obs

        #print(f"Option_length={option_length}")
        #total_reward = total_reward/option_length
        # -- logging the option behaviour -- (NOTE that the info is also still coming from the base env. FIX IT!)
        
        info["option"] = option_idx
        info["option_length"] = option_length
        info["discount"] = self.gamma**option_length # of this decision, for SMDPDiscountCallback
        info["survey_fraction"] = start_step / self.max_primitive_steps if self.max_primitive_steps else 0.0 # when the decision was made
        info["survey_steps"] = self.primitive_steps
        #info["trajectory"]=trajectory_info

        return self._meta_observation(next_obs), total_reward, done, trunc, info

    def _compute_meta_subgoal_reward(self, obs):
        if self.once_per_episode == False and (obs["plume_coverage"][0]<self.accuracy_goals[1] or obs["border_coverage"][0]< self.accuracy_goals[2]):
            self.once_per_episode = True # allowing to be rewarded for 2 or more plumes
        #print(f"ONCE per EPS:{self.once_per_episode}, PLUME cov{obs['plume_coverage'][0]}, BORDER cov{obs["border_coverage"][0]}, MAXN OVER {self.base_env.unwrapped.maxN_over_thresh}, MAXN AROUND {self.base_env.unwrapped.maxN_around_thresh}")
        if self.once_per_episode and obs["plume_coverage"][0]>=self.accuracy_goals[1] and obs["border_coverage"][0]>= self.accuracy_goals[2] and self.base_env.unwrapped.maxN_over_thresh>0.0 and self.base_env.unwrapped.maxN_around_thresh>0.0:
            self.once_per_episode = False
            reward = 40.0 # bonus for reaching the subgoal, to 
            terminated = False # we want to explore at least 80% of the rest
            #print("✅")
        elif self.base_env.env.unwrapped.MEAN_uncertainty<=(1-self.accuracy_goals[0]) and obs["plume_coverage"][0]>=self.accuracy_goals[1] and obs["border_coverage"][0]>= self.accuracy_goals[2] :#obs["space_coverage"][0]>=self.accuracy_goals[0]:
            reward = 40.0
            terminated = True
            #print("✅✅ covered 80% of the area")
        else:
            # placeholders for the HRL env reward construction
            reward = 0.0 
            terminated = False
        return reward, terminated