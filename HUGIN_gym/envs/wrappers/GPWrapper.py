import numpy as np
import gymnasium as gym
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
from HUGIN_gym.envs.core.rewards.agent2_plume import Reward as Reward2_plume
#from HUGIN_gym.overlay.overlay_server import latest_values

class GPWrapper(gym.Wrapper):
    def __init__(self, env, kernels_config = None, gp_optimise = False, gp_visualise = False, max_points = 2000,HRL=False, SUB_AGENT_TRAIN_ON_GP =False):
        super().__init__(env)
        self.HRL = HRL
        self.env.unwrapped.HRL = HRL # for the base_envs
        self.gp_optimise = gp_optimise
        self.gp_visualise = gp_visualise
        self.accuracy_goals = [0.9,0.9,0.9]
        self.option_length = None
        self.action_plan_len = None
        self.META_test = False
       # Default kernels if none provided
        if kernels_config is None:
            kernels_config = {
                "RBF": {"type": "RBF", "length_scale": 3.5},
            }
        self.kernels_config = kernels_config
        if len(kernels_config.keys())>1:
            print("[WARNING] only one kernel allowed!")
            exit()
      
        self.GP_key = next(iter(self.kernels_config))
        
        self.max_points = max_points
        if env.unwrapped.map_size_x != env.unwrapped.map_size_y:
            print("[WARNING] X and Y dimension should be the same!!")
            exit()

        self.c_thresh = env.unwrapped.c_threshold # NOTE, these are assumed to be constant during training. If intended to modify, pls move into reset
        self.c_std = env.unwrapped.c_around_width
        
        self.kernels = {}
       # self.env.unwrapped.GP_maxN_over_thresh = 0
        #self.env.unwrapped.GP_maxN_around_thresh = 0

        self._build_kernels()
        self.once_per_episode = True
        self.subpolicy_GP= SUB_AGENT_TRAIN_ON_GP
        #self.env.unwrapped.subpol_GP = SUB_AGENT_TRAIN_ON_GP # not needed as GP_ON exists already in HUGIN_env

        if self.HRL or self.subpolicy_GP:
            self.actual_GT_N_c_over = np.sum(self.env.unwrapped.GT_c_over_threshold_maps)
            self.actual_GT_N_c_around = np.sum(self.env.unwrapped.GT_c_around_threshold_maps)
        
        self._3D=env.unwrapped._3D
    
        self.grid_size = self.env.unwrapped.map_size_x
        self.shift = self.env.unwrapped.offset[0]
        coords = np.arange(-self.shift, self.shift + 1)

        if self._3D:
            self.grid_shape = (self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y, self.env.unwrapped.map_size_z)
            X, Y, Z = np.meshgrid(coords, coords, coords) # NOTE, meshgrid convention = X varies along columns, Y along rows (X<->Y flip)
            self.grid_positions = np.column_stack([Z.ravel(),Y.ravel(), X.ravel()])
        else:
            self.grid_shape = (self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y)
            X, Y = np.meshgrid(coords, coords) # NOTE, meshgrid convention = X varies along columns, Y along rows (X<->Y flip)
            self.grid_positions = np.column_stack([Y.ravel(), X.ravel()])
        
        
        
        self.gp_map_zeros = np.zeros(self.grid_shape, dtype=np.float32) # 2D thingy
        self.GP_pred=self.gp_map_zeros

        if self.gp_visualise: # CAREFUL! only for visualisation, NEVER for trainig!
            self.env.unwrapped.extract_GT_maps= self.env.unwrapped.GT_c_over_threshold_maps
        ### INIT GP UPDATE, so that state space doesnt start with GT at step 0 and switches then to GP at step1...

        ##### COrner filler:

        X = self.env.unwrapped.map_size_x
        Y = self.env.unwrapped.map_size_y
        Z = self.env.unwrapped.map_size_z  # 1 in 2D

        corner_mask = np.zeros((X, Y, Z), dtype=bool)

        if Z > 1:
            # 3D: 8 corners
            coords = [
                (0,      0,      0),
                (0,      0,      Z-1),
                (0,      Y-1,    0),
                (0,      Y-1,    Z-1),
                (X-1,    0,      0),
                (X-1,    0,      Z-1),
                (X-1,    Y-1,    0),
                (X-1,    Y-1,    Z-1),
            ]
        else:
            # 2D: 4 corners (z = 0)
            coords = [
                (0,      0,      0),
                (X-1,    0,      0),
                (0,      Y-1,    0),
                (X-1,    Y-1,    0),
            ]

        for ix, iy, iz in coords:
            corner_mask[ix, iy, iz] = True

        self.corner_mask = corner_mask  # shape (X, Y, Z)
        self.N_corners = np.sum(corner_mask)
        self.original_GT_c_over = self.env.unwrapped.GT_c_over_threshold_maps.copy()
        self.original_GT_c_around = self.env.unwrapped.GT_c_around_threshold_maps.copy()
         # ---- 6. Threshold Maps --> Overwrite Global ----
        if self.env.unwrapped.agent_type in ("PLUME","META"):
            over_2d = self.gp_map_zeros # a priori wrong everywhere
            if self._3D:
                over_3d = over_2d.astype(bool) 
            else:
                over_3d = over_2d[..., None].astype(bool) 
            # --- force corners to True in the belief plume map ---
            over_3d[self.corner_mask] = True
            self._overwrite_GT_with_GP(over_3d,env_key= "GT_c_over_threshold_maps")
            #print(f"SUM OVER:{np.sum(over)}")
            if self.subpolicy_GP==True:
                maxN_over_thresh = np.sum(self.env.unwrapped.GT_c_over_threshold_maps).astype(np.float32) 
                #print(f"MAX N OVER{maxN_over_thresh}")
                self._overwrite_GT_with_GP(maxN_over_thresh,env_key = "maxN_over_thresh")
            
        if self.env.unwrapped.agent_type in ("BORDER","META"):
            around_2d = self.gp_map_zeros              # (X, Y)
            if self._3D:
                around_3d = around_2d.astype(bool)
            else:
                around_3d = around_2d[..., None].astype(bool)  # (X, Y, 1)
            around_3d[self.corner_mask] = True
            self._overwrite_GT_with_GP(around_3d, env_key="GT_c_around_threshold_maps")
            if self.subpolicy_GP==True:
                maxN_around_thresh = np.sum(self.env.unwrapped.GT_c_around_threshold_maps).astype(np.float32) 
                self._overwrite_GT_with_GP(maxN_around_thresh,env_key = "maxN_around_thresh")

        if self.env.unwrapped.agent_type in ("META"):
            maxN_over_thresh = np.sum(self.env.unwrapped.GT_c_over_threshold_maps).astype(np.float32) 
            self._overwrite_GT_with_GP(maxN_over_thresh,env_key = "maxN_over_thresh")
            maxN_around_thresh = np.sum(self.env.unwrapped.GT_c_around_threshold_maps).astype(np.float32) 
            self._overwrite_GT_with_GP(maxN_around_thresh,env_key = "maxN_around_thresh")
            
        
        self._get_local_patch = env.unwrapped.get_local_patch
        self.distance_from_max = 1.0 # initialised as 1.0 ton be as far as possible away
        self.min_steps_in_a_plume = 5
        self.plume_counter_not_yet_init = 0
        self.SPACE_test_counter = 0

        if self._3D ==True:
            #rn *5 ,this is simply blocking GP, cause we changed our investigation
            N_space = int(self.env.unwrapped.map_size_x * self.env.unwrapped.map_size_y * self.env.unwrapped.map_size_z / 12 *0.7) 
        else:
            N_space = int(self.env.unwrapped.map_size_x * self.env.unwrapped.map_size_y / 6 *5) #
        self.N_predict = 1 if self.env.unwrapped.agent_type!="SPACE" else N_space # change the 200 if you go to 3D or scale the space!
        #print(self._3D,self.N_predict)
    # -------------------------
    # Build GP kernels dynamically
    # -------------------------
    def _build_kernels(self):
        for name, kcfg in self.kernels_config.items():

            if kcfg["type"] == "RBF":
                kernel = C(1.0) * RBF(length_scale=kcfg.get("length_scale", 3.5))

            elif kcfg["type"] == "Matern":
                kernel = C(1.0) * Matern(
                    length_scale=kcfg.get("length_scale", 3.5),
                    nu=kcfg.get("nu", 1.5),
                )

            else:
                raise ValueError(f"Unknown kernel type: {kcfg['type']}")

            # Add noise kernel
            kernel += WhiteKernel(
                noise_level=1e-4,
                noise_level_bounds=(1e-8, 1e1),
            )

            self.kernels[name] = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=False, # True?!? for varying concentration field that could be better!
                optimizer=None if not self.gp_optimise else "fmin_l_bfgs_b",
            )

    # -------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.GP_pred=self.gp_map_zeros
        self.distance_from_max = 1.0  
        self.SPACE_test_counter = 0
        self.X = []
        self.y = []
        
        if self.HRL or self.subpolicy_GP:
            self.actual_GT_N_c_over = np.sum(self.env.unwrapped.GT_c_over_threshold_maps)
            self.actual_GT_N_c_around = np.sum(self.env.unwrapped.GT_c_around_threshold_maps)
            self.original_GT_c_over = self.env.unwrapped.GT_c_over_threshold_maps.copy()
            self.original_GT_c_around = self.env.unwrapped.GT_c_around_threshold_maps.copy()

        self.real_grid = np.zeros((self.grid_size, self.grid_size))
        self.once_per_episode = True
        self.plume_counter_not_yet_init=0

        
        if self.gp_visualise: # CAREFUL! only for visualisation, NEVER for trainig!
            self.env.unwrapped.extract_GT_maps= self.env.unwrapped.GT_c_over_threshold_maps
            
         # ---- 6. Threshold Maps --> Overwrite Global ----
        if self.env.unwrapped.agent_type in ("PLUME","META"):
            over_2d = self.gp_map_zeros # a priori wrong everywhere
            if self._3D:
                over_3d = over_2d.astype(bool) 
            else:
                over_3d = over_2d[..., None].astype(bool) 
            # --- force corners to True in the belief plume map ---
            over_3d[self.corner_mask] = True
            self._overwrite_GT_with_GP(over_3d,env_key= "GT_c_over_threshold_maps")
            #print(f"SUM OVER:{np.sum(over)}")
            if self.subpolicy_GP==True:
                maxN_over_thresh = np.sum(self.env.unwrapped.GT_c_over_threshold_maps).astype(np.float32) 
                #print(f"MAX N OVER{maxN_over_thresh}")
                self._overwrite_GT_with_GP(maxN_over_thresh,env_key = "maxN_over_thresh")
                self._overwrite_GT_with_GP(np.array([1.0]),obs_key = "plume_coverage" ,obs=obs)
            
        if self.env.unwrapped.agent_type in ("BORDER","META"):
            around_2d = self.gp_map_zeros              # (X, Y)
            if self._3D:
                around_3d = around_2d.astype(bool)
            else:
                around_3d = around_2d[..., None].astype(bool)  # (X, Y, 1)
            around_3d[self.corner_mask] = True
            self._overwrite_GT_with_GP(around_3d, env_key="GT_c_around_threshold_maps")
            if self.subpolicy_GP==True:
                maxN_around_thresh = np.sum(self.env.unwrapped.GT_c_around_threshold_maps).astype(np.float32) 
                self._overwrite_GT_with_GP(maxN_around_thresh,env_key = "maxN_around_thresh")
                self._overwrite_GT_with_GP(np.array([1.0]),obs_key = "border_coverage" ,obs=obs)

        if self.env.unwrapped.agent_type in ("META"):
            maxN_over_thresh = np.sum(self.env.unwrapped.GT_c_over_threshold_maps).astype(np.float32)
            self._overwrite_GT_with_GP(maxN_over_thresh,env_key = "maxN_over_thresh")
            maxN_around_thresh = np.sum(self.env.unwrapped.GT_c_around_threshold_maps).astype(np.float32)
            self._overwrite_GT_with_GP(maxN_around_thresh,env_key = "maxN_around_thresh")
            self._overwrite_reset_obs_with_belief(obs)
        return obs, info

    def _overwrite_reset_obs_with_belief(self, obs):
        """
        META: the base env built the first observation from the true plume. Rebuild its plume keys from the
        empty belief set above (GT_c_*_threshold_maps hold the belief now: only the corner cells), and store the
        belief patches and maps in the base env, which reuses them on boundary steps before the first GP update.
        """
        base = self.env.unwrapped
        ix, iy, iz = base._pos_to_idx(base.state["x"], base.state["y"], base.state["z"])
        visited_3d = base.state["visited_map"][:, :, :, 0].astype(bool)

        over_belief_3d = np.logical_and(base.GT_c_over_threshold_maps, visited_3d)
        base.state["c_over_threshold_map"][:, :, :, 0] = over_belief_3d # as after every GP update in step()
        local_over_GT = self._get_local_patch(base.GT_c_over_threshold_maps, ix, iy, iz, radius=base.patch_radius)
        over_downsampled = base.downsample_map(base.GT_c_over_threshold_maps).flatten().astype(np.float32)
        self._overwrite_GT_with_GP(local_over_GT, env_key="local_GT")
        self._overwrite_GT_with_GP(over_downsampled, env_key="_gt_over_downsampled")
        self._overwrite_GT_with_GP(local_over_GT.flatten(), obs_key="local_GT", obs=obs)
        self._overwrite_GT_with_GP(over_downsampled, obs_key="GT_c_over_threshold_maps_downsampled", obs=obs)
        plume_coverage = np.array(
            [np.sum(over_belief_3d) / base.maxN_over_thresh if base.maxN_over_thresh > 0 else 1.0],
            dtype=np.float32
        )
        self._overwrite_GT_with_GP(plume_coverage, obs_key="plume_coverage", obs=obs)

        around_belief_3d = np.logical_and(base.GT_c_around_threshold_maps, visited_3d)
        base.state["c_around_threshold_map"][:, :, :, 0] = around_belief_3d
        local_around_GT = self._get_local_patch(base.GT_c_around_threshold_maps, ix, iy, iz, radius=base.patch_radius)
        around_downsampled = base.downsample_map(base.GT_c_around_threshold_maps).flatten().astype(np.float32)
        self._overwrite_GT_with_GP(local_around_GT, env_key="local_GT_around")
        self._overwrite_GT_with_GP(around_downsampled, env_key="_gt_around_downsampled")
        self._overwrite_GT_with_GP(local_around_GT.flatten(), obs_key="local_GT_around", obs=obs)
        self._overwrite_GT_with_GP(around_downsampled, obs_key="GT_c_around_threshold_maps_downsampled", obs=obs)
        border_coverage = np.array(
            [np.sum(around_belief_3d) / base.maxN_around_thresh if base.maxN_around_thresh > 0 else 1.0],
            dtype=np.float32
        )
        self._overwrite_GT_with_GP(border_coverage, obs_key="border_coverage", obs=obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.SPACE_test:
            self.SPACE_test_counter +=1
            if (self.SPACE_test_counter==self.env.unwrapped.max_eps_len or self.SPACE_test_counter==self.action_plan_len) and (self.env.unwrapped.agent_type in ("SPACE")):
                visited_3d = self.env.unwrapped.state["visited_map"][:, :, :, 0].astype(bool)
                reward, terminated = self._compute_SPACE_reward(obs)

                if self.env.unwrapped.SPACE_test and (terminated or truncated or self.SPACE_test_counter==self.env.unwrapped.max_eps_len or self.SPACE_test_counter==self.action_plan_len):
                    
                    info = self._confusion_matrix(info)
                    
                
                    info["actual_current_above_threshold"] = np.sum(self.original_GT_c_over & visited_3d)/self.actual_GT_N_c_over
                    info["actual_current_around_threshold"] = np.sum(self.original_GT_c_around & visited_3d)/self.actual_GT_N_c_around
        
        if reward == -4.0:
            pass
        else:
            if self._3D:
                # -- 1. Load X,Y,Z
                pos = [obs["obs_state"][0],obs["obs_state"][1] ,obs["obs_state"][2]]
                c = obs["obs_state"][7]
                #print(f"pos{pos}")
                # -- 2. Store the data --
                x,y,z = pos[0]*self.shift, pos[1]*self.shift,pos[2]*self.shift
                ix,iy,iz = self._pos_to_idx(x,y,z)
                self.update([x,y,z],c,ix,iy,iz) # 3D!
            else:
                # -- 1. Load X,Y,Z
                pos = [obs["obs_state"][0],obs["obs_state"][1]] #,obs["obs_state"][2]
                c = obs["obs_state"][7]
                #print(f"pos{pos}")
                # -- 2. Store the data --
                x,y = pos[0]*self.shift, pos[1]*self.shift
                ix,iy,iz = self._pos_to_idx(x,y)
                self.update([x,y],c,ix,iy,iz) # 3D!

            # -- 3. Limit dataset --
            if len(self.X) > self.max_points:
                self.X = self.X[-self.max_points:]
                self.y = self.y[-self.max_points:]

            # -- 4. Fit GP --
            if True:#c>=self.env.unwrapped.c_around_width:
                outputs = self.predict()
                GP_pred = outputs[self.GP_key] # assuming that the first (hopefully only) kernel is the one intended for the GP when running the trainig
                self.GP_pred = GP_pred
                
            if self.gp_visualise: # CAREFUL! only for visualisation, NEVER for trainig!
                self.env.unwrapped.extract_GP_maps=self.GP_pred

            visited_3d = self.env.unwrapped.state["visited_map"][:, :, :, 0].astype(bool)
            
            if self.env.unwrapped.agent_type in ("SPACE"):
                reward, terminated = self._compute_SPACE_reward(obs)
            # if self.env.unwrapped.SPACE_test and (terminated or truncated or self.SPACE_test_counter==self.env.unwrapped.max_eps_len) :
                
            #     info = self._confusion_matrix(info)
                
                
            #     info["actual_current_above_threshold"] = np.sum(self.original_GT_c_over & visited_3d)/self.actual_GT_N_c_over
            #     info["actual_current_around_threshold"] = np.sum(self.original_GT_c_around & visited_3d)/self.actual_GT_N_c_around
            # ---- 6. Threshold Maps --> Overwrite Global ----
            if self.env.unwrapped.agent_type in ("PLUME","META"):
                
                gp_over_2d = (self.GP_pred > self.c_thresh)     #(X,Y)
                if np.sum(gp_over_2d)==0:
                    gp_over_2d = (self.GP_pred > self.c_thresh/2)
                # Convert to 3D (X, Y, Z=1)
                if self._3D:
                    gp_over_3d = gp_over_2d
                else:
                    gp_over_3d = gp_over_2d[..., None]  # (X, Y, 1)
                # --- force corners to True in the belief plume map ---
                gp_over_3d[self.corner_mask] = True
                
                # Downsample expects (H, W, D) = (X, Y, Z)
                downsampled = self.env.unwrapped.downsample_map(gp_over_3d)
                self._overwrite_GT_with_GP(downsampled.flatten().astype(np.float32),
                                        obs_key="GT_c_over_threshold_maps_downsampled",
                                        obs=obs)

                self._overwrite_GT_with_GP(gp_over_3d.astype(bool),
                                        env_key="GT_c_over_threshold_maps")
                over_belief_3d = np.logical_and(gp_over_3d, visited_3d) 
                self.env.unwrapped.state["c_over_threshold_map"][:, :, :, 0] = over_belief_3d.astype(bool)
                local_over_GT = self._get_local_patch(gp_over_3d, ix, iy, iz,
                                                radius=self.env.unwrapped.patch_radius)
                self._overwrite_GT_with_GP(local_over_GT.flatten(), obs_key="local_GT", obs=obs)
                if self.env.unwrapped.agent_type == "META": # boundary steps reuse these: keep them on the belief, not the true plume
                    self._overwrite_GT_with_GP(local_over_GT, env_key="local_GT")
                    self._overwrite_GT_with_GP(obs["GT_c_over_threshold_maps_downsampled"], env_key="_gt_over_downsampled")

                if self.gp_visualise:
                    local_over = self._get_local_patch(over_belief_3d, ix, iy, iz,
                                                radius=self.env.unwrapped.patch_radius)
                    self._overwrite_GT_with_GP(local_over.flatten(), obs_key="obs_state", obs=obs,trim={"lower":8+self.env.unwrapped.patch_size,"upper":8+2*self.env.unwrapped.patch_size})
                #print(f"SUM OVER:{np.sum(over)}")
                
                maxN_over_thresh = np.sum(self.env.unwrapped.GT_c_over_threshold_maps).astype(np.float32) #- self.N_corners , , leaving it initially as N_corners will help stability!
                
                #print(f"MAX N OVER{maxN_over_thresh}")
                self._overwrite_GT_with_GP(maxN_over_thresh,env_key = "maxN_over_thresh")
                
                c_over_thresh_belief = np.sum(over_belief_3d) #- self.N_corners , leaving it initially as N_corners will help stability!

                plume_coverage = np.array(
                    [c_over_thresh_belief / maxN_over_thresh if maxN_over_thresh > 0 else 1.0],
                    dtype=np.float32
                )
                
                self._overwrite_GT_with_GP(plume_coverage,obs_key = "plume_coverage" ,obs=obs)
                info["assumed_current_above_threshold"] = plume_coverage[0]
                info["actual_current_above_threshold"] = np.sum(self.original_GT_c_over & visited_3d)/self.actual_GT_N_c_over
                if self.subpolicy_GP:
                    reward, terminated = self._compute_PLUME_reward(obs)
                
            if self.env.unwrapped.agent_type in ("BORDER","META"):
                gp_around_2d = ((self.GP_pred > self.c_thresh - self.c_std) &
             (self.GP_pred <= self.c_thresh + self.c_std))  # (X, Y)
                
                if self._3D:
                    gp_around_3d = gp_around_2d
                else:
                    gp_around_3d = gp_around_2d[..., None]  # (X, Y, 1)
                gp_around_3d[self.corner_mask] = True
                
                downsampled = self.env.unwrapped.downsample_map(gp_around_3d)
                self._overwrite_GT_with_GP(downsampled.flatten().astype(np.float32),
                                        obs_key="GT_c_around_threshold_maps_downsampled",
                                        obs=obs)

                self._overwrite_GT_with_GP(gp_around_3d.astype(bool),
                                        env_key="GT_c_around_threshold_maps")

                around_belief_3d = np.logical_and(gp_around_3d, visited_3d) 
                self.env.unwrapped.state["c_around_threshold_map"][:, :, :, 0] = around_belief_3d.astype(bool)
                local_around_GT = self._get_local_patch(gp_around_3d, ix, iy, iz,
                                                radius=self.env.unwrapped.patch_radius)
                self._overwrite_GT_with_GP(local_around_GT.flatten(),
                                        obs_key="local_GT_around",
                                        obs=obs)
                if self.env.unwrapped.agent_type == "META": # boundary steps reuse these: keep them on the belief, not the true plume
                    self._overwrite_GT_with_GP(local_around_GT, env_key="local_GT_around")
                    self._overwrite_GT_with_GP(obs["GT_c_around_threshold_maps_downsampled"], env_key="_gt_around_downsampled")
                if self.gp_visualise:
                    local_around = self._get_local_patch(around_belief_3d, ix, iy, iz,
                                                    radius=self.env.unwrapped.patch_radius)
                    self._overwrite_GT_with_GP(local_around.flatten(), obs_key="obs_state", obs=obs,trim={"lower":8+2*self.env.unwrapped.patch_size,"upper":8+3*self.env.unwrapped.patch_size})
                maxN_around_thresh = np.sum(self.env.unwrapped.GT_c_around_threshold_maps).astype(np.float32) 
                self._overwrite_GT_with_GP(maxN_around_thresh,env_key = "maxN_around_thresh")
                
                
                c_around_thresh_belief = np.sum(around_belief_3d)
                border_coverage = np.array(
                    [c_around_thresh_belief / maxN_around_thresh if maxN_around_thresh > 0 else 1.0],
                    dtype=np.float32
                )
                
                self._overwrite_GT_with_GP(border_coverage,obs_key = "border_coverage" ,obs=obs)
                info["assumed_current_around_threshold"] = border_coverage[0]
                info["actual_current_around_threshold"] = np.sum(self.original_GT_c_around & visited_3d)/self.actual_GT_N_c_around
                if self.subpolicy_GP:
                    reward, terminated = self._compute_BORDER_reward(obs)

            if self.env.unwrapped.agent_type in ("META"):
                # --- recompute reward + termination ---
                reward, terminated = self._compute_meta_subgoal_reward(obs)
                if self.META_test and (terminated or truncated or  self.option_length>=self.max_option_length-1 ):
                
                    info = self._confusion_matrix(info)
            
            if self.env.unwrapped.SPACE_test and (terminated or truncated or self.SPACE_test_counter==self.env.unwrapped.max_eps_len) :
                
                info = self._confusion_matrix(info)
                
                
                info["actual_current_above_threshold"] = np.sum(self.original_GT_c_over & visited_3d)/self.actual_GT_N_c_over
                info["actual_current_around_threshold"] = np.sum(self.original_GT_c_around & visited_3d)/self.actual_GT_N_c_around
        return obs, reward, terminated, truncated, info
    def update(self, pos, c,ix,iy,iz, step=None):
        self.X.append(pos) # 3D!! 
        self.y.append(c)

        if self.gp_visualise:
            # carefull with indices when introducing 3D, Z will be 0, X 1, and Y 2
            if 0 <= ix < self.env.unwrapped.map_size_x and 0 <= iy < self.env.unwrapped.map_size_y:
                self.real_grid[ix, iy] = c

    def predict(self):
        
        
        if len(self.X) < self.N_predict:  # set to 1 as we filter prediction with concentration measurements
            self.env.unwrapped.MEAN_uncertainty = 1.0 # fictive value
            return {
                name: self.gp_map_zeros # assuming a reset before
                for name in self.kernels
            }
        
        X = np.array(self.X)
        y = np.array(self.y)

        outputs = {}

        for name, kernel in self.kernels.items():
            kernel.fit(X, y)
            if self.env.unwrapped.train and self.env.unwrapped.agent_type in ("SPACE","META"): #,"PLUME","BORDER"
                mean, std = kernel.predict(self.grid_positions, return_std=True)
                self.env.unwrapped.MEAN_uncertainty = np.mean(std)
            elif self.env.unwrapped.train:
                mean = kernel.predict(self.grid_positions, return_std=False)
            else:
                mean, std = kernel.predict(self.grid_positions, return_std=True)
                if self._3D:
                    outputs[name] = mean.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y,self.env.unwrapped.map_size_z)
                    self.env.unwrapped.MEAN_uncertainty = np.mean(std)

                    self.env.unwrapped.extract_GP_ERROR_map = std.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y,self.env.unwrapped.map_size_z ) # careful!! No reset yet
                else:
                    outputs[name] = mean.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y)
                    self.env.unwrapped.MEAN_uncertainty = np.mean(std)

                    self.env.unwrapped.extract_GP_ERROR_map = std.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y) # careful!! No reset yet
                return outputs
            if self._3D:
                outputs[name] = mean.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y,self.env.unwrapped.map_size_z)
            else:
                outputs[name] = mean.reshape(self.env.unwrapped.map_size_x, self.env.unwrapped.map_size_y)

        return outputs


    def get_real_grid(self):
        return self.real_grid

        # -------------------------
    def _overwrite_GT_with_GP(self,  corresponding_GP_est, obs_key=None,obs=None,env_key = None, trim=None):
        if (obs_key is not None):
            if (obs_key not in obs.keys()):
                print("[WARNING - key mismatch! Fix your choice of keys... Provide obs, if not already done so.]")
            if (trim is not None):
                obs[obs_key][trim["lower"]:trim["upper"]] = corresponding_GP_est
            else:
                obs[obs_key] = corresponding_GP_est
        elif (env_key is not None):
            if not hasattr(self.env.unwrapped, env_key):
                print(f"[WARNING - env key '{env_key}' does not exist! Creating it.]")
            setattr(self.env.unwrapped, env_key, corresponding_GP_est)


    
    def _pos_to_idx(self, x, y, z=0.0):
        ix = int(round(x / self.env.unwrapped.cell_size)) + self.env.unwrapped.offset[0]
        iy = int(round(y / self.env.unwrapped.cell_size)) + self.env.unwrapped.offset[1]
        iz = int(round(z / self.env.unwrapped.cell_size)) + self.env.unwrapped.offset[2]

        ix = np.clip(ix, 0, self.env.unwrapped.map_size_x - 1)
        iy = np.clip(iy, 0, self.env.unwrapped.map_size_y - 1)
        iz = np.clip(iz, 0, self.env.unwrapped.map_size_z - 1)

        return ix, iy, iz
    
    def _compute_meta_subgoal_reward(self, obs):
        if self.once_per_episode == False and (obs["plume_coverage"][0]<self.accuracy_goals[1] or obs["border_coverage"][0]< self.accuracy_goals[2]):
            self.once_per_episode = True # allowing to be rewarded for 2 or more plumes
        if self.once_per_episode and obs["plume_coverage"][0]>=self.accuracy_goals[1] and obs["border_coverage"][0]>= self.accuracy_goals[2] and self.env.unwrapped.maxN_over_thresh>self.N_corners and self.env.unwrapped.maxN_around_thresh>self.N_corners:
            self.once_per_episode = False
            reward = 40.0 # bonus for reaching the subgoal, to 
            terminated = False # we want to explore at least 80% of the rest
            #print("✅")
        elif self.env.unwrapped.MEAN_uncertainty<=(1-self.accuracy_goals[0]) and obs["plume_coverage"][0]>=self.accuracy_goals[1] and obs["border_coverage"][0]>=self.accuracy_goals[2]:#obs["space_coverage"][0]>=self.accuracy_goals[0]:
            reward = 40.0
            terminated = True
            #print("✅✅ YOU SHOULD TERMINATE!!!!!")
        else:
            # placeholders for the HRL env reward construction
            reward = 0.0 
            terminated = False

        return reward, terminated
    
    def _compute_SPACE_reward(self, obs):
        terminated = False
        VAR_downsampled = None#obs["VAR_visited_downsampled"][0]
        new_cells = np.sum(self.env.unwrapped.state["visited_map"][:,:,:,0]) - self.env.unwrapped.old_visited_sum
        reward = self.env.unwrapped.reward_fn_1explore.get_reward(
            self.env.unwrapped.agent_turns, obs["space_coverage"][0], new_cells, 
            _3D=self.env.unwrapped._3D, agent_rises=self.env.unwrapped.agent_rises, VAR_downsampled=VAR_downsampled
        )
        if self.env.unwrapped.MEAN_uncertainty<=(1-self.accuracy_goals[0]):#obs["space_coverage"][0] >= self.accuracy_goals[0]:#
            reward = 300.0
            terminated = True
        return reward, terminated
    
    def _compute_PLUME_reward(self, obs):
        terminated = False
        diff_new_cells = np.sum(self.env.unwrapped.state["visited_map"][:,:,:,0]) - self.env.unwrapped.old_visited_sum
        found_source =  (obs["plume_coverage"][0] < self.accuracy_goals[1])
        #print(f"FOUND:{found_source}")
        reward = self.env.unwrapped.reward_fn_2plume.get_reward( self.env.unwrapped.agent_turns,obs["plume_coverage"][0],np.sum(self.env.unwrapped.state["c_over_threshold_map"])-self.env.unwrapped.old_c_over_sum,SUBAGENT_TRAIN_ON_GP=self.env.unwrapped.GP_ON,diff_new_cells=diff_new_cells,found_source=found_source,percentage_visited=obs["space_coverage"][0])
        #print(f"C_was_already_below:{self.env.unwrapped.c_plume_coverage_was_already_below}")
        # -- TERMINATION conditions
        if obs["plume_coverage"][0] >= self.accuracy_goals[1] and self.env.unwrapped.c_plume_coverage_was_already_below: 
            reward = 40.0 # bonus for finding the source, as the agent might get stuck in local areas with higher concentration but not the source
            self.env.unwrapped.c_plume_coverage_was_already_below = False
            terminated = True
                    
        elif (obs["plume_coverage"][0] < self.accuracy_goals[1]) :
            self.plume_counter_not_yet_init+=1
            if self.plume_counter_not_yet_init>=self.min_steps_in_a_plume:
                self.env.unwrapped.c_plume_coverage_was_already_below = True

        # if self.env.unwrapped.MEAN_uncertainty<=(1-self.accuracy_goals[0]) and obs["plume_coverage"][0]==1:#obs["space_coverage"][0] >= self.accuracy_goals[0]: 
        #     reward = 5.0
        #     terminated = True

        return reward, terminated
    
    def _compute_BORDER_reward(self, obs):
        terminated = False
        diff_new_cells = np.sum(self.env.unwrapped.state["visited_map"][:,:,:,0]) - self.env.unwrapped.old_visited_sum
        found_source =  obs["border_coverage"][0] < self.accuracy_goals[2]
        reward = self.env.unwrapped.reward_fn_3border.get_reward( self.env.unwrapped.agent_turns,obs["border_coverage"][0],np.sum(self.env.unwrapped.state["c_around_threshold_map"])-self.env.unwrapped.old_c_around_sum,SUBAGENT_TRAIN_ON_GP=self.env.unwrapped.GP_ON,diff_new_cells=diff_new_cells,found_source=found_source,percentage_visited=obs["space_coverage"][0])
        # -- TERMINATION conditions
        if obs["border_coverage"][0] >= self.accuracy_goals[2] and self.env.unwrapped.c_border_coverage_was_already_below: 
            reward = 40.0 # bonus for finding the source, as the agent might get stuck in local areas with higher concentration but not the source
            self.env.unwrapped.c_border_coverage_was_already_below = False
            terminated = True
        elif obs["border_coverage"][0] < self.accuracy_goals[2] :
            self.plume_counter_not_yet_init+=1
            if self.plume_counter_not_yet_init>=self.min_steps_in_a_plume:
                self.env.unwrapped.c_border_coverage_was_already_below = True

        # if self.env.unwrapped.MEAN_uncertainty<=(1-self.accuracy_goals[0]) and obs["border_coverage"][0]==1:#obs["space_coverage"][0] >= self.accuracy_goals[0]: 
        #     reward = 5.0
        #     terminated = True
        return reward, terminated
    

    def _confusion_matrix(self,info): 
        max_cells = self.grid_size*self.grid_size
        gp_over_2d = (self.GP_pred > self.c_thresh) 
        gp_over_3d = gp_over_2d[..., None]  # (X, Y, 1)
        # --- force corners to True in the belief plume map ---
        gp_over_3d[self.corner_mask] = True
        true_positive = np.logical_and(self.original_GT_c_over,gp_over_3d )
        true_negative = np.logical_and(~self.original_GT_c_over, ~gp_over_3d)
        false_positive  = np.logical_and(~self.original_GT_c_over ,  gp_over_3d)
        false_negative  =  np.logical_and(self.original_GT_c_over, ~gp_over_3d)

        info["false_positive_above"] = np.sum(false_positive) / max_cells
        info["true_positive_above"] = np.sum(true_positive)/ max_cells
        info["true_negative_above"] = np.sum(true_negative) /max_cells
        info["false_negative_above"] = np.sum(false_negative) /max_cells

        gp_around_2d = ((self.GP_pred > self.c_thresh - self.c_std) &
             (self.GP_pred <= self.c_thresh + self.c_std))  # (X, Y)
                

        gp_around_3d = gp_around_2d[..., None]  # (X, Y, 1)
        gp_around_3d[self.corner_mask] = True
        true_positive = np.logical_and(self.original_GT_c_around, gp_around_3d) 
        true_negative = np.logical_and(~self.original_GT_c_around, ~gp_around_3d )
        false_positive  = np.logical_and(~self.original_GT_c_around,  gp_around_3d)
        false_negative  =  np.logical_and(self.original_GT_c_around, ~gp_around_3d)

        info["false_positive_around"] = np.sum(false_positive) / max_cells
        info["true_positive_around"] = np.sum(true_positive)/ max_cells
        info["true_negative_around"] = np.sum(true_negative) /max_cells
        info["false_negative_around"] = np.sum(false_negative) /max_cells

        return info
     
        