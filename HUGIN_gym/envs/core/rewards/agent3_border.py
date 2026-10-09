import numpy as np

class Reward:
    def __init__(self):
        pass
        
    def get_reward(self,  agent_turns, c_around_threshold_normed,diff_around_new, time_penalty = -0.7,SUBAGENT_TRAIN_ON_GP = False,diff_new_cells=None,found_source=None,percentage_visited=None,_3D=False, agent_rises=False):
        

        reward = 0
        if SUBAGENT_TRAIN_ON_GP and (not found_source):
                # reward += np.clip(diff_new_cells/4,0,1)
                reward += diff_new_cells/6*0.7
        #reward += np.clip(diff_around_new/4,0,1)
        reward += diff_around_new/6
        # if fill_around_threshold:
        #     #print("FILL")
        #     #reward += 0.2
        #     if not current_position_IN_visited_locations:
        #         reward=0.9+0.3*(1-c_over_threshold_normed)
        #         reward = diff_around_new
        # if current_around_threshold:
        #     #print("CURRENT")
        #     #reward += 0.2
        #     if not current_position_IN_visited_locations:
        #         reward=0.9+0.3*(1-c_over_threshold_normed)
        #         reward = diff_around_new
        # if distance_to_closest_unvisited_cell<=1+1e-3:
        #     reward += 0.2
        # else:
        #     reward += 0.2/distance_to_closest_unvisited_cell
        """
        if agent_turns:
            if percentage_visited:
                if c_around_threshold_normed==1.0:
                    reward-=0.3 * (1-percentage_visited)
                else:
                    reward -= 0.3 * (1-c_around_threshold_normed)*(1-percentage_visited) # turning is bad unless it helps the progress
            else:
                if c_around_threshold_normed==1.0:
                    reward-=0.3
                else:
                    reward -= 0.3 * (1-c_around_threshold_normed
        """
        reward += time_penalty
        # no velocity penalties as the HUGIN has to constantly move 2m/s forward

        #[reward is between -1 and 1]
        return reward


   




