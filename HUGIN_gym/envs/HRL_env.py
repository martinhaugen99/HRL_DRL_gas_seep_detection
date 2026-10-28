import gymnasium as gym
import numpy as np
import time

class HierarchicalEnv(gym.Env):
    def __init__(self, base_env, options, gamma=0.99,MAX_OPT_LENGTH=5,ACCURACY_GOALS = None,GP=True):
        super().__init__()
        self.base_env = base_env # gp_env lol
        #self.base_env.unwrapped.HRL = True
        self.options = options
        self.gamma = gamma
        self.max_opt_length = MAX_OPT_LENGTH
        self.accuracy_goals = ACCURACY_GOALS if ACCURACY_GOALS else self.base_env.accuracy_goals 
        self.base_env.accuracy_goals = self.accuracy_goals# NOTE, for the expert training we used 0.9, but for HRL we set the level a bit lower

        # Action space: choose option
        self.action_space = gym.spaces.Discrete(len(options))

        # Observation space: SAME as base env
        self.observation_space = base_env.observation_space
        self.current_obs = None

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
        return obs, info

    def step(self, option_idx):
        option = self.options[option_idx]

        total_reward = 0
        discount = 1
        done = False
        trunc = False
        #trajectory_info = []
        
        obs = self.current_obs
        
        #trajectory_info.append(state)
        option_length = 0
        self.base_env.option_length = option_length

        while True:
            
            action = option.act(obs)
            action = int(action)
            option_length +=1
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
            
            if not done and reward_debug!=-4.0 and reward_debug!=40.0:
                total_reward += discount * reward
            elif reward_debug==40.0:
                total_reward +=discount * reward_debug
            elif reward_debug==-4.0:
                total_reward +=discount * reward_debug
            else:
                total_reward += discount * reward
            discount *= self.gamma

            if done or trunc or self.no_progress_steps>2 or option_length >=self.max_opt_length: #! >1
                if not (done or trunc) and option_length < self.max_opt_length:
                    # the meta agent discounts each decision by gamma**MAX_OPT_LENGTH, which is only exact for full-length options
                    raise RuntimeError(
                        f"Option ended after {option_length} of {self.max_opt_length} steps while the episode goes on. "
                        "The meta agent's discount gamma**MAX_OPT_LENGTH needs every such option to run MAX_OPT_LENGTH "
                        "steps (options end after 3 steps without progress, so use MAX_OPT_LENGTH <= 3), "
                        "or a discount of gamma**option_length per decision."
                    )
                if self.no_progress_steps>2:
                    total_reward-=2.0 # penalty for no progress!

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
        #info["trajectory"]=trajectory_info

        return next_obs, total_reward, done, trunc, info

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