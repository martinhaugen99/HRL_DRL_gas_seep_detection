import time
import torch as th
import gymnasium as gym

def run_episode(env, 
                model, 
                config, 
                vis_state_space=[None,None, None], # 1: MapVisualiser, 2: heatmap, 3:obs, 4:cnn 
                gp_visualise = None,
                GP=None,
                VISUALISE=None,
                verbose = 0,
                testing_space_agent = False,
                CoverageBarVisualiser=False,):
    keys_to_keep = config["RUN_INFO"]["filtered_obs_keys"]
    agent_type = config["RUN_INFO"]["AGENT_TYPE"]
    
    
    max_eps_len = config["ENVIRONMENT"]["max_episode_length"]

    if GP:
        env.base_envHUGIN = env.unwrapped # just make the base_env accessible due to the GP wrapper which adds an additional layer agent->HierarchicalEnv->GPWrapper->HUGINEnv
    else:
        env.base_envHUGIN = env

    # ---- PRE-RESET CLEANUP ----
    if VISUALISE:
        try:
            env.unwrapped.renderer.vis.delete()
        except:
            pass
        try:
            cov_bar=CoverageBarVisualiser(obs)
        except:
            pass

    obs, _ = env.reset()
    
    if VISUALISE:
        env.render()

    # ---- ENV CONFIG ----
    env.unwrapped.agent_type = agent_type
    if agent_type == "SPACE":
        env.use_c_map = False ## for spatial explorer
    else:
        env.use_c_map = True
    if testing_space_agent:
        env.use_c_map = True
    env.unwrapped.random_points = True
    
    episode_reward = 0
    step_count = 0
    episode_turns = 0
    if gp_visualise is not None:
        GT_grid = env.unwrapped.GT_c_over_threshold_maps[:,:,0] # z=0
    if vis_state_space[2]:
        #initialise MapVisualiser:
        Mapvisualiser=vis_state_space[0](obs,env)
        compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
        vis_state_space[1].update(compressed_map,step_count)
    # =========================
    #  MAIN LOOP 
    # =========================
    while True:
        # ---- FILTER OBS (SAFE VERSION) ----
        obs_input = {k: obs[k] for k in keys_to_keep}

        # ---- ACTION ----
        action, _ = model.predict(obs_input, deterministic=True)

        # ---- STEP ----
        #obs, reward, terminated, truncated, info = env.step(int(action))
        if isinstance(env.action_space, gym.spaces.Box):
            step_action = action  # SAC: ContinuousToDiscreteActionWrapper maps it to a discrete action
        else:
            step_action = int(action)
        obs, reward, terminated, truncated, info = env.step(step_action)
        #if terminated== 1:
           # time.sleep(1.5)
            #print(reward)
        #print(f"REWARD:{reward:.2f}-----STEP:{step_count}\n")
        #print(f"SPACE={obs['space_coverage'][0]*100:.2f}%, PLUME={obs['plume_coverage'][0]*100:.2f}%, BORDER={obs['border_coverage'][0]*100:.2f}%")
        episode_reward += reward
        if env.base_envHUGIN.agent_turns:
            episode_turns+=1
        
        # ---- GP ----
        try:
            pass #print(np.mean(env.unwrapped.MEAN_uncertainty))
        except:
            pass
        if gp_visualise is not None:
            try:
                gp_visualise.update(GT_grid,{"RBF":env.unwrapped.extract_GP_maps,"ERROR": env.unwrapped.extract_GP_ERROR_map},env.real_grid,)#, }
            except:
                pass
        # ---- VISUAL ----
        if VISUALISE:
            env.unwrapped.step_sim()
            #time.sleep(0.01)

        step_count += 1
        # ---- UPDATE VIS - State-Space

        if vis_state_space[2]:
            Mapvisualiser.update(obs,step_count)
            compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
            
            vis_state_space[1].update(compressed_map,step_count)
        
        # ---- TERMINATION ----
        if terminated or truncated or step_count == max_eps_len :
            if env.unwrapped.train == False and verbose == 1:
                if terminated:
                    print("✅ reached the goal")
                else:
                    print("❌ did not reach the goal within MAX_EPS_LEN")
            # ---- METRIC EXTRACTION ----
            if agent_type == "BORDER":
                metric = info.get("current_around_threshold", 0)
            elif agent_type == "PLUME":
                metric = info.get("current_above_threshold", 0)
            elif agent_type == "SPACE":
                metric = info.get("visited_states_count", 0)
            else:
                metric = None


            if VISUALISE:
                env.unwrapped.renderer.reset()

            return {
                "plume_coverage": obs["plume_coverage"][0],
                "border_coverage": obs["border_coverage"][0],
                "reward": episode_reward,
                "episode_turns": episode_turns,
                "steps": step_count,
                "space_coverage": obs["space_coverage"][0],
                "info": info,
            }

import torch as th
import numpy as np
import matplotlib.pyplot as plt
def manual_rollout(
    env, MAX_EPS_LEN,
    visualise=False,
    vis_state_space=[None,None, None, None], # 0: MapVisualiser, 1: heatmap, 2:obs, 3:cnn , 4: extra heatmap, 5: extra heatmap, 6:Coverage Bars
    gp_visualise = None,
):
    """
    Manual rollout similar to run_episode, but actions are provided by keyboard input.
    """
   

    episode_reward = 0
    step_count = 0

    done = False
    if gp_visualise:
        if env.unwrapped.GP_ON:
            #print(env.unwrapped.conc.shape)
            GT_grid = env.unwrapped.conc[:,:,0].T #env.unwrapped.extract_GT_maps[0].T
        else:
            GT_grid = env.unwrapped.GT_c_over_threshold_maps[:,:,0].T
    if vis_state_space[3]:
        compressed_map = th.from_numpy(vis_state_space[2]["visited_maps_downsampled"].reshape(4,4))
        vis_state_space[1].update(compressed_map,step_count)
        if  len(vis_state_space)>4:
            vis_state_space[4].update(th.from_numpy(env.unwrapped.downsample_map(env.unwrapped.state["c_over_threshold_map"][:,:,:,0]))[:,:,0],step_count)
            vis_state_space[5].update(th.from_numpy(env.unwrapped.downsample_map(env.unwrapped.state["c_around_threshold_map"][:,:,:,0]))[:,:,0],step_count)
            if len(vis_state_space)>6:
                vis_state_space[6].update(vis_state_space[2])
    
    if gp_visualise is not None:
                try:
                    gp_visualise.update(GT_grid,{"RBF": np.zeros_like(GT_grid),"ERROR":np.zeros_like(GT_grid)},np.zeros_like(GT_grid)) #, "Matern":np.zeros_like(GT_grid)
                except Exception as e:
                    print(f"Please define your GT grid {e}!")

    
    if visualise:
            env.unwrapped.step_sim()
            #time.sleep(0.2)
    while not done and step_count < MAX_EPS_LEN:
        # --- Get manual action ---
        key = input("Enter control (wasdqerf, x to exit): ").lower()
        if key == "x":
            print("Exiting manual rollout...")
            break
        elif key == " ":
            action = 0
        elif key == "w":
            action = 1
        elif key == "s":
            action = 2
        elif key == "a":
            action = 3
        elif key == "d":
            action = 4
        else:
            print("Invalid key")
            continue

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        # diff = env.unwrapped.extract_GP_maps.T-GT_grid           # numpy array
        # mse  = (diff ** 2).mean()
        # rmse = np.sqrt(mse)
        #print(f"REWARD:{reward:.2f}-----STEP:{step_count}\n")
        #print(f"SPACE={obs['space_coverage'][0]*100:.2f}%, PLUME={obs['plume_coverage'][0]*100:.2f}%, BORDER={obs['border_coverage'][0]*100:.2f}%")
        #print(f"RMSE{rmse:.5f}")
        try:
            print(f"UNCERTAINTY mean: {(env.unwrapped.MEAN_uncertainty):.2f}")
        except:
            pass

        #print(obs["plume_coverage"][0])
        print(f"x={obs["obs_state"][0]:.2f}, y={obs["obs_state"][1]:.2f},z={obs["obs_state"][2]:.2f},theta={env.unwrapped.state["theta"]:.2f}")
        #print(f"OBS sum {np.sum(obs["visited_maps_downsampled"])}")
        print(f"AFTER STEP: SPACE{obs["space_coverage"]}, PLUME{obs["plume_coverage"]}, BORDER{obs["border_coverage"]}")
        #print(f"REWARD:{reward}")
        # print(type(obs["space_coverage"][0]),type(obs["plume_coverage"][0]),type(obs["border_coverage"][0]))
        # ---- UPDATE VIS - State-Space
      
        if vis_state_space[3]:
            vis_state_space[0].update(obs,step_count)
            compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
            
            vis_state_space[1].update(compressed_map,step_count)
            if  len(vis_state_space)>4:
                vis_state_space[4].update(th.from_numpy(env.unwrapped.downsample_map(env.unwrapped.state["c_over_threshold_map"][:,:,:,0]))[:,:,0],step_count)
                vis_state_space[5].update(th.from_numpy(env.unwrapped.downsample_map(env.unwrapped.state["c_around_threshold_map"][:,:,:,0]))[:,:,0],step_count)
                
                if len(vis_state_space)>6:
                    
                    vis_state_space[6].update(obs)
        # ---- GP VIS ----

        if gp_visualise is not None:
            try:
                # plt.figure(55)
                # plt.imshow(
                #     env.unwrapped.extract_GP_ERROR_map,
                #     cmap="magma",
                #     extent=[-20, 20, -20, 20],
                #     origin="lower",
                # )
                # plt.xlabel("Y")
                # plt.colorbar()
                gp_visualise.update(GT_grid,{"RBF":env.unwrapped.extract_GP_maps.T-GT_grid, "ERROR": env.unwrapped.extract_GP_ERROR_map},env.real_grid,)
            except Exception as e:
                print(f"Please define your GT grid {e}!")
        # ---- VISUAL ----
        if visualise:
            env.unwrapped.step_sim()
            time.sleep(0.2)
    
    # ---- RESET ----
    env.unwrapped.renderer.vis.delete()
    env.unwrapped.renderer.reset()
    obs, _ = env.reset()
    env.unwrapped.render()
    print("Episode ended, resetting...")

    return obs, reward, done, info



###########
#   HRL
###########

def run_episode_hrl(
    env,
    model,
    max_eps_len,
    vis_state_space=[None,None, None,None], # 1: MapVisualiser, 2: heatmap, 3:obs, 4:cnn 
    gp_visualise=None,
    GP=True,
    verbose = False,
    VISUALISE=False
):
    """
    HRL rollout:
    - model selects OPTIONS
    - env executes them internally
    """
    if GP:
        env.base_envHUGIN = env.base_env.unwrapped # just make the base_env accessible due to the GP wrapper which adds an additional layer agent->HierarchicalEnv->GPWrapper->HUGINEnv
    else:
        env.base_envHUGIN = env.base_env
    # ---- PRE-RESET CLEANUP ----
    if vis_state_space[0] or VISUALISE:
        try:
            env.base_envHUGIN.renderer.vis.delete()
        except:
            pass

    obs, _ = env.reset()

    if vis_state_space[0] or gp_visualise or VISUALISE:
        env.base_envHUGIN.render()

    episode_reward = 0
    step_count = 0
    episode_turns = 0
    base_env_step_count = 0
    env.base_env.option_length = 0
    # ---- optional GT snapshot ----
    if gp_visualise is not None:
        try:
            GT_grid = env.base_envHUGIN.GT_c_over_threshold_maps.T
        except:
            GT_grid = None

    if vis_state_space[3]:
        #initialise MapVisualiser:
        #Mapvisualiser=vis_state_space[0](obs)
        compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
        vis_state_space[1].update(compressed_map,step_count)

    # =========================
    # MAIN LOOP
    # =========================
    while True:

        # ---- NO FILTERING (HRL safer) ----
        action, _ = model.predict(obs, deterministic=True)

       # print(f"Chosen Option: {action}")

        obs, reward, terminated, truncated, info = env.step(int(action))
        env.base_env.option_length +=1
        # try:
        #     print(f"actual plume coverage {info['actual_current_above_threshold']*100:.2f}%, actual border coverage {info['actual_current_around_threshold']*100:.2f}%")
        # except:
        #     pass
        # if reward>0:
        #     print(f"REWARD {reward}")
        episode_reward += reward
        step_count += 1
        base_env_step_count += info["option_length"]
        # if env.base_envHUGIN.agent_turns: # WRONG, because the HRL can turn multiple tiems within one option
        #     episode_turns+=1
        #print(env.turn_counter)
        # print(f"REWARD:{reward:.2f}")
        # print(f"SPACE={obs['space_coverage'][0]*100:.2f}%, PLUME={obs['plume_coverage'][0]*100:.2f}%, BORDER={obs['border_coverage'][0]*100:.2f}%")
         # ---- UPDATE VIS - State-Space

        if vis_state_space[3]:
            vis_state_space[0].update(obs,step_count)
            compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
            
            vis_state_space[1].update(compressed_map,step_count)
        
        # ---- GP VIS ----
        if gp_visualise is not None:
            try:
                gp_visualise.update(GT_grid,{"RBF":env.base_envHUGIN.GT_c_over_threshold_maps},env.base_env.real_grid,)

            except Exception as e:
                print(f"GP visualisation failed: {e}")

        # ---- VISUAL ----
        if vis_state_space[0] or VISUALISE:
            env.base_envHUGIN.step_sim()
            #time.sleep(0.1)

        # ---- TERMINATION ----
        if terminated or truncated or step_count >= max_eps_len:

            if not env.base_envHUGIN.unwrapped.train and verbose:
                if terminated:
                    print("✅ reached goal")
                else:
                    print("❌ max steps reached")

            if vis_state_space[0] or VISUALISE:
                env.base_envHUGIN.renderer.reset()
            
            if verbose:
                print(f"Episode finished in {step_count} steps")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Final coverage - SPACE: {obs['space_coverage'][0]*100:.2f}%, PLUME: {obs['plume_coverage'][0]*100:.2f}%, BORDER: {obs['border_coverage'][0]*100:.2f}%")
                print("Episode ended, resetting...")
            
            return {
                "space_coverage": obs["space_coverage"][0],
                "plume_coverage": obs["plume_coverage"][0],
                "border_coverage": obs["border_coverage"][0],
                "reward": episode_reward,
                "episode_turns":  env.turn_counter,
                "base_env_steps": base_env_step_count,
                "options_chosen": step_count,
                "terminated": terminated,
                "info": info,
            }

        

def manual_rollout_hrl(
    env,
    MAX_EPS_LEN,
    visualise=False,
    vis_state_space=[None,None, None, None], # 1: MapVisualiser, 2: heatmap, 3:obs, 4:cnn
    gp_visualise = None,
    GP=True,
):
    """
    Manual rollout similar to run_episode, but actions are provided by keyboard input.
    """
    if GP:
        env.base_envHUGIN = env.base_env.unwrapped # just make the base_env accessible due to the GP wrapper which adds an additional layer agent->HierarchicalEnv->GPWrapper->HUGINEnv
    else:
        env.base_envHUGIN = env.base_env
    
    episode_reward = 0
    step_count = 0

    done = False
    GT_grid = env.base_envHUGIN.GT_c_over_threshold_maps[0].T
    if gp_visualise:
        if env.base_env.unwrapped.GP_ON:
            #print(env.unwrapped.conc.shape)
            GT_grid = env.base_env.unwrapped.conc[:,:,0].T #env.unwrapped.extract_GT_maps[0].T
        else:
            GT_grid = env.unwrapped.GT_c_over_threshold_maps[:,:,0].T
    if vis_state_space[3]:
        compressed_map = th.from_numpy(vis_state_space[2]["visited_maps_downsampled"].reshape(4,4))
        vis_state_space[1].update(compressed_map,step_count)
        if  len(vis_state_space)>4:
            vis_state_space[4].update(th.from_numpy(env.base_env.unwrapped.downsample_map(env.base_env.unwrapped.state["c_over_threshold_map"][:,:,:,0]))[:,:,0],step_count)
            vis_state_space[5].update(th.from_numpy(env.base_env.unwrapped.downsample_map(env.base_env.unwrapped.state["c_around_threshold_map"][:,:,:,0]))[:,:,0],step_count)
            if len(vis_state_space)>6:
                vis_state_space[6].update(vis_state_space[2])
    
    if gp_visualise is not None:
                try:
                    gp_visualise.update(GT_grid,{"RBF": np.zeros_like(GT_grid),"ERROR":np.zeros_like(GT_grid)},np.zeros_like(GT_grid)) #, "Matern":np.zeros_like(GT_grid)
                except Exception as e:
                    print(f"Please define your GT grid {e}!")

    while not done and step_count < MAX_EPS_LEN:
        # --- Get manual action ---
        key = input("Enter control (wasdqerf, x to exit): ").lower()
        if key == "x":
            print("Exiting manual rollout...")
            break
        elif key == "a":
            action = 0
        elif key == "s":
            action = 1
        elif key == "d":
            action = 2
        else:
            print("Invalid key")
            continue

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.base_env.unwrapped.MEAN_uncertainty)
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        print("------------------")
        print(f"step{step_count}")
        print(f"REWARD:{reward:.2f}")
        print(f"SPACE={obs['space_coverage'][0]*100:.2f}%, PLUME={obs['plume_coverage'][0]*100:.2f}%, BORDER={obs['border_coverage'][0]*100:.2f}%")
        

        # ---- UPDATE VIS - State-Space

        if vis_state_space[3]:
            vis_state_space[0].update(obs,step_count)
            compressed_map = (th.from_numpy(obs["visited_maps_downsampled"].reshape(4,4)))
            
            vis_state_space[1].update(compressed_map,step_count)
            if  len(vis_state_space)>4:
                vis_state_space[4].update(th.from_numpy(env.base_env.unwrapped.downsample_map(env.base_env.unwrapped.state["c_over_threshold_map"][:,:,:,0]))[:,:,0],step_count)
                vis_state_space[5].update(th.from_numpy(env.base_env.unwrapped.downsample_map(env.base_env.unwrapped.state["c_around_threshold_map"][:,:,:,0]))[:,:,0],step_count)
                
                if len(vis_state_space)>6:
                    
                    vis_state_space[6].update(obs)
        # ---- GP VIS ----

        if gp_visualise is not None:
            try:
                # plt.figure(55)
                # plt.imshow(
                #     env.unwrapped.extract_GP_ERROR_map,
                #     cmap="magma",
                #     extent=[-20, 20, -20, 20],
                #     origin="lower",
                # )
                # plt.xlabel("Y")
                # plt.colorbar()
                gp_visualise.update(GT_grid,{"RBF":env.base_env.unwrapped.extract_GP_maps.T-GT_grid, "ERROR": env.base_env.unwrapped.extract_GP_ERROR_map},env.real_grid,)
            except Exception as e:
                print(f"Please define your GT grid {e}!")
        
        # ---- VISUAL ----
        if visualise:
            env.base_envHUGIN.step_sim()
            time.sleep(0.2)
    
    # ---- RESET ----
    env.base_envHUGIN.renderer.vis.delete()
    env.base_envHUGIN.renderer.reset()
    obs, _ = env.reset()
    env.base_envHUGIN.render()
    print("Episode ended, resetting...")

    return obs, reward, done, info


def run_episodes_hrl_vec(
    env,               # SubprocVecEnv
    model,
    total_episodes,
    max_eps_len,
):
    import numpy as np

    num_envs = env.num_envs

    obs = env.reset()

    # ---- per-env trackers ----
    episode_rewards = np.zeros(num_envs)
    step_counts = np.zeros(num_envs)
    episode_turns = np.zeros(num_envs)

    completed_episodes = 0
    results = []

    while completed_episodes < total_episodes:

        actions, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(actions)

        episode_rewards += rewards
        step_counts += 1

        for i in range(num_envs):

            
            if env.base_env.agent_turns:
                episode_turns[i] += 1

            if dones[i] or step_counts[i] >= max_eps_len:

                # ---- collect result ----
                results.append({
                    "space_coverage": obs[i]["space_coverage"][0],
                    "plume_coverage": obs[i]["plume_coverage"][0],
                    "border_coverage": obs[i]["border_coverage"][0],
                    "reward": episode_rewards[i],
                    "episode_turns": episode_turns[i],
                    "steps": step_counts[i],
                    "terminated": dones[i],
                    "info": infos[i],
                })

                completed_episodes += 1

                # ---- reset that env ----
                episode_rewards[i] = 0
                step_counts[i] = 0
                episode_turns[i] = 0


                if completed_episodes >= total_episodes:
                    break

    return results