from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy

from sklearn import dummy
from HUGIN_gym.envs.core.dynamics import Dynamics
from HUGIN_gym.envs.core.rewards.agent2_plume import Reward as Reward2_plume
from HUGIN_gym.envs.core.rewards.agent1_exploration import Reward as Reward1_explore
from HUGIN_gym.envs.core.rewards.agent3_border import Reward as Reward3_border
from HUGIN_gym.envs.core.visualisation.renderer import HUGINRenderer

#from HUGIN_gym.overlay.overlay_server import start_server

class HUGIN(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        with resources.path("HUGIN_gym.assets", "AUV2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "human":
            self.renderer = HUGINRenderer()
            meshcat_url = self.renderer.vis.url()  # dynamically get the port
            #start_server(meshcat_url)  # pass it to Flask overlay
        self.train = False
        self.use_c_map = True
        
        self.target_range = [-1, 1]
        self.dynamics = Dynamics()
        
        # choose 3^n as dimensions, due to symmetric downsampling
        self.domain_limit = {
            "x": 20,#22
            "y": 20,#22
            "z": 0, #1
        }
        self.eps = 0.5 # uncertainty acceptance at boarders. NOTE, due to the coordinate transform due to different angle states for the actions, we get drifting floats. For isntance, -4,00000002... .
        if self.render_mode == "human":
            self.renderer.size=self.domain_limit["x"]*2
            if self.domain_limit["z"]>0.0:
                self.renderer.depth=self.domain_limit["z"]*2
                self._3D=True
            else:
                self._3D=False
        else:
            if self.domain_limit["z"]>0.0:
                self._3D=True
            else:
                self._3D=False
        # -- bit map for visited places --
    
        self.cell_size = 1.0  # meters per cell (choose based on required resolution)
        # map sizes (integers)
        self.map_size_x = int(2*int(np.ceil(self.domain_limit["x"] / self.cell_size))) + 1
        self.map_size_y = int(2*int(np.ceil(self.domain_limit["y"] / self.cell_size))) + 1
        self.map_size_z = int(2*int(np.ceil(self.domain_limit["z"] / self.cell_size))) + 1

        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "concentration": 0.0,
            "theta": 0,
            "visited_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool),
            "c_over_threshold_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool), 
            "c_around_threshold_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool), 
        }

        self._zero_c_map = np.zeros((self.map_size_z,self.map_size_y,self.map_size_x), dtype = bool) # transposed and [:,:,:,0]
        
        self.max_return=self.map_size_x*self.map_size_y*self.map_size_z # for normalising return later on

        # -- guassian concentration field parameters -- # NOTE: center sampled from uniform prefers closer to center of domain

        # Sample a random hotspot location
        self.multiple_gaussians = [1,1] # default = [1,1] , it defines the range, e.g. [1,5] would cause from 1 up to 5 gaussian sources within the volume at the same time
        self.N_gaussians = np.random.randint(self.multiple_gaussians[0], self.multiple_gaussians[1] + 1)
        z_3D_up_to_2 = np.clip(self.domain_limit["z"],0,2)
        self.gaussian_centers = np.random.uniform(
        low=-np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]-z_3D_up_to_2]),
        high=np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]-z_3D_up_to_2]),
        size=(self.N_gaussians, 3))
        # if self.spawn_close_to_source:
        # def _sample_position_close_to_source(self,max_tries=1000, margin_x=2, margin_y = 2, margin_z=2):
        #     for _ in range(max_tries):
        #         x =np.random.randint(-6,6)
        #         y_max = int(np.sqrt(6**2-np.abs(x)**2))
        #         y = np.random.randint(-y_max,y_max)

        #         if self._3D:
        #             z_max = int(np.sqrt(6**2-np.abs(x)**2-np.abs(y)**2))
        #             z = np.random.randint(-z_max,z_max)
        #         else : 
        #             z = 0
        #         if x**2+y**2+z**2<4.0**2
        #         if (
        #             abs(x) <= self.domain_limit["x"] - margin_x and
        #             abs(y) <= self.domain_limit["y"] - margin_y and
        #             abs(z) <= self.domain_limit["z"] - margin_z
        #         ):
        #             return np.array([x, y, z], dtype=float)
        #     6.0**2 > (x**2+y**2+z**2) >= 4.0**
        #self.state["x"],self.state["y"],self.state["z"]=self._sample_position_close_to_source()

        # (x0, y0, z0)
        self.gaussian_sigma = 3.5   # spread of the Gaussian blob
        self.gaussian_amplitude = 1.0  # peak concentration
        self.c_threshold = 0.3 # threshold for over threshold map, can be adapted
        #variable plume
        self.vary_plume_size = False      # new random plume size every episode (switched on by the PLUME training/test scripts)
        self.gaussian_sigma_range = [2.0, 5.0]  # sigma range used when vary_plume_size is on: plumes 5-15 cells wide
        self.spawn_near_plume = False     # start in a box around the plume instead of anywhere on the map (switched on by the PLUME training/test scripts)
        self.spawn_margin = 3             # cells added around the plume's bounding box

        self.c_around_width = 0.15
        self.get_concentration = self.concentration_function
        self.reward_fn_1explore = Reward1_explore() # Note, concentration GP model must be in one of the reward functions
        self.reward_fn_2plume = Reward2_plume()
        self.reward_fn_3border = Reward3_border()

        """ -- action space (discrete) -- """
        self.action_space = spaces.Discrete(5)  
     
        self.move_vectors = {
            0: np.array([2.0,        0.0,        0.0,        0.0       ], dtype=np.float32),
            1: np.array([1.0,  0.0,        1.0,  0.0       ], dtype=np.float32),
            2: np.array([1.0,  0.0,       -1.0,  0.0       ], dtype=np.float32),
            3: np.array([1.0,  1.0,  0.0,        np.pi/2   ], dtype=np.float32),
            4: np.array([1.0, -1.0,  0.0,       -np.pi/2   ], dtype=np.float32),
        }

        
        "replace soon patch -> full map"
        self.patch_radius = 3 # radius of local visited map patch (2 -> 5x5)
        parameters= 3+1+1+(self.patch_radius*2+1)**2#(self.map_size_x*self.map_size_y*self.map_size_z) # x,y,z + concentration + theta + visited patch + current position in visited map

        self._zero_c_patch = np.zeros_like(self._get_local_visited_patch(1, 1,0,self.patch_radius))
        
        if self._3D:
            self.patch_size=(2*self.patch_radius+1)**3
        else:
            self.patch_size=(2*self.patch_radius+1)**2
        
        

        ##!###!!!##!!!##!!!##!!!!
        CHANGE_THIS_PLEASE_TO_KERNEL_SIZE_AND_STRIDE_FROM_GP_WRAPPER = True
        ##!###!!!##!!!##!!!##!!!!
        
        # Use per-dimension kernel and stride
        if self.map_size_z <= 1:
            # effectively 2D: do not downsample along depth
            self.kernel_size = (3, 3, 1)   # (kD, kH, kW)
            self.stride      = (2, 2, 1)   # (sD, sH, sW)
        else:
            # full 3D downsampling
            self.kernel_size = (3, 3, 3)
            self.stride      = (2, 2, 2)

        dummy = np.zeros((self.map_size_x, self.map_size_y, self.map_size_z), dtype=np.float32)
        downsampled = self.downsample_map(dummy)
        #print(f"downsampled shape: {downsampled.shape}") # note that we remove the edges close to the border was we dont do padding
        obs_map_shape = np.array([*downsampled.flatten()]).shape
       
        # -- discrete observation space -- DQN #
        self.observation_space = spaces.Dict({#155
            "obs_state": spaces.Box(low=0, high=1, shape=(8+3*self.patch_size,), dtype=np.float32), # x,y,z, [_ _ _ _], c = 8 params (pos., orientation, conc.) +343 local patch visited +343 local over thresh patch + 343 local patch around thresh
            "visited_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32), # Z,X,Y # here a gigantic version of visited_patch
            #"c_over_threshold_maps_counter": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            #"c_around_threshold_maps_counter": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "GT_c_over_threshold_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "GT_c_around_threshold_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "local_GT": spaces.Box(low=0, high=1, shape=(self.patch_size,), dtype=np.float32),
            "local_GT_around": spaces.Box(low=0, high=1, shape=(self.patch_size,), dtype=np.float32),
            "space_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "plume_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "border_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "distance_from_max":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            #"VAR_visited_downsampled":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
    
       
       # -- transfering state position into uint8 map (0,255) --
        
        self.offset = np.array([self.map_size_x // 2, self.map_size_y // 2, self.map_size_z // 2])

        ix = int(round(self.state["x"] / self.cell_size)) + self.offset[0]
        iy = int(round(self.state["y"] / self.cell_size)) + self.offset[1]
        iz = int(round(self.state["z"] / self.cell_size)) + self.offset[2]

        current_position = np.array([ix, iy, iz],dtype=np.uint8)
        
        self.state["visited_map"][current_position[0],current_position[1],current_position[2],:] = 1 # updating visited map



        self.dt = 1  # Time step 0.1->1 second # doesnt impact anything in discrete step env
        self.render_mode = render_mode
        self.random_points = False # for simulation testing
        self.agent_turns = False # to handle reward function, incentify not to turn too much
        self.agent_rises = False
        

        self.x = np.arange(-self.map_size_x//2, self.map_size_x//2)[:, None, None]
        self.y = np.arange(-self.map_size_y//2, self.map_size_y//2)[None, :, None]
        self.z = np.arange(-self.map_size_z//2, self.map_size_z//2)[None, None, :]
        
        self.conc = self.get_concentration(self.x, self.y, self.z)
        
        # debugging for finding out how many steps may be needed
        #print(f"fields above threshold at start: {np.sum(self.conc>=self.c_threshold)}, around threshold and above: {np.sum( (self.conc >= self.c_threshold - self.c_around_width))}")

        mask_above = (self.conc >= self.c_threshold) 

        self.GT_c_over_threshold_maps = np.where(mask_above, 1, 0).astype(bool)
        
        mask_around = (self.conc <= self.c_threshold + self.c_around_width) & (self.conc >= self.c_threshold - self.c_around_width)

        self.GT_c_around_threshold_maps = np.where(mask_around, 1, 0).astype(bool)

        self.local_GT_around=self._get_local_GT_around_patch(ix,iy,iz,self.patch_radius)
        self.local_GT=self._get_local_GT_patch(ix,iy,iz,self.patch_radius)
        if self._3D:
            self.maxN_over_thresh=(np.sum(self.GT_c_over_threshold_maps)+8).astype(np.float32) # careful! only in z plane
        else:
            self.maxN_over_thresh=(np.sum(self.GT_c_over_threshold_maps)+4).astype(np.float32)
        if self._3D:
            self.maxN_around_thresh=(np.sum(self.GT_c_around_threshold_maps)+8).astype(np.float32) # careful! only in z plane
        else:
            self.maxN_around_thresh=(np.sum(self.GT_c_around_threshold_maps)+4).astype(np.float32) # careful! only in z plane
        
        
        self.GP_ON=False # default, no GP active
        self.c_border_coverage_was_already_below = False
        self.c_plume_coverage_was_already_below = False
        
        self.accuracy_agent_goals = [0.9,0.9,0.9]
        self.HRL = False
        self.distance_from_max = 1.0 # initial maximal distance, will be managed after step 1
          # Pre-allocate observation buffers (add these)
        self._obs_state_buffer = np.zeros(8+3*self.patch_size, dtype=np.float32)
        self._visited_downsampled = None  # Cache this
        self._gt_over_downsampled = None  # Cache this (static after reset)
        self._gt_around_downsampled = None  # Cache this (static after reset)
        
        self._visited_map_version = 0
        self._set_corner_cells_visited()
        # For GP training/testing of the space agent
        self.SPACE_test = False
        self.MEAN_uncertainty = 1.0
        
    # -- Define the concentration function -- #   
    def concentration_function(self, x, y, z):
            centers = np.array(self.gaussian_centers)  # (N, 3)
            #sigma = self.gaussian_sigma
            sigma = np.reshape(self.gaussian_sigma, (-1, 1, 1, 1))  # one sigma per source; a plain number still works
            A = self.gaussian_amplitude

            # reshape centers for broadcasting
            cx = centers[:, 0][:, None, None, None]  # (N,1,1,1)
            cy = centers[:, 1][:, None, None, None]
            cz = centers[:, 2][:, None, None, None]

            # your x,y,z are already (Nx,1,1), (1,Ny,1), (1,1,Nz)
            dx = x - cx   # → (N, Nx, Ny, Nz)
            dy = y - cy
            dz = z - cz

            r2 = dx**2 + dy**2 + dz**2

            c = A * np.exp(-r2 / (2 * sigma**2))

            return np.sum(c, axis=0).astype(np.float32)
    

    def _snap_theta(self, theta):
        theta = theta % (2*np.pi)

        dirs = np.array([
            0.0,
            0.5*np.pi,
            1.0*np.pi,
            1.5*np.pi
        ])

        diff = (dirs - theta + np.pi) % (2*np.pi) - np.pi
        idx = np.argmin(np.abs(diff))
        return dirs[idx]

    def downsample_map(self, x ):
        x = x.astype(np.float32)

        x = self.avg_pool3d_numpy(x, self.kernel_size, self.stride)
        x = self.avg_pool3d_numpy(x, self.kernel_size, self.stride)
        x = self.avg_pool3d_numpy(x, self.kernel_size, self.stride)
    
        return x
    def avg_pool3d_numpy(self, x, kernel_size, stride):
        """
        x: (D, H, W)
        kernel_size, stride: either int or (kD, kH, kW) / (sD, sH, sW)
        """
        # Normalize kernel_size and stride to 3-tuples
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kH, kW, kD = kernel_size

        if isinstance(stride, int):
            sD = sH = sW = stride
        else:
            sH, sW, sD = stride

        H, W, D = x.shape

        # If depth is <= 1, always use kD=1, sD=1 so we don't shrink the depth axis
        if D <= 1:
            kD = 1
            sD = 1

        
        out_h = (H - kH) // sH + 1
        out_w = (W - kW) // sW + 1
        out_d = (D - kD) // sD + 1

        # Guard against invalid sizes
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid pooling output size: "
                f"(out_d={out_d}, out_h={out_h}, out_w={out_w}) "
                f"for input shape {(H, W, D)}, "
                f"kernel={( kH, kW, kD)}, stride={( sH, sW,sD)}"
            )

        shape = (
            out_h,
            out_w,
            out_d,
            kH,
            kW,
            kD,
        )

        strides = (
            sH * x.strides[0],
            sW * x.strides[1],
            sD * x.strides[2],
            x.strides[0],
            x.strides[1],
            x.strides[2],
        )

        windows = np.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides
        )

        return windows.mean(axis=(3, 4, 5))

    
  
    
    def _get_local_GT_patch(self, ix, iy,iz, radius=2):
        return self.get_local_patch(
            self.GT_c_over_threshold_maps,
            ix, iy,iz ,radius
        )
    def _get_local_GT_around_patch(self, ix, iy, iz, radius=2):
        return self.get_local_patch(
            self.GT_c_around_threshold_maps,
            ix, iy, iz ,radius
        )
    def _get_local_visited_patch(self, ix, iy, iz, radius=2):
        visited = self.state["visited_map"][:, :, :, 0]
        return self.get_local_patch(visited, ix, iy,iz ,radius)

    def _get_local_over_c_patch(self, ix, iy, iz, radius=2):
        return self.get_local_patch(
            self.state["c_over_threshold_map"][:, :, :, 0],
            ix, iy,iz, radius
        )
    def _get_local_around_c_patch(self, ix, iy, iz, radius=2):
        return self.get_local_patch(
            self.state["c_around_threshold_map"][:, :, :, 0],
            ix, iy, iz,radius
        )
    def get_local_patch(self, map_3d, ix, iy, iz, radius=2, normalize=True):
        """
        Generic local patch extractor.

        Args:
            map_3d: 3D numpy array with shape (X, Y, Z)
            ix, iy, iz: center index in this array
            radius: patch radius
            normalize: map values from [-1,1] to [0,1] if True

        Returns:
            For 3D: patch of shape (2r+1, 2r+1, 2r+1)
            For 2D (_3D == False): patch of shape (2r+1, 2r+1, 1)
        """

        map_3d = map_3d.astype(np.float32)

        if self._3D:
            # Pad alle tre dimensjoner
            padded = np.pad(
                map_3d,
                pad_width=radius,
                mode="constant",
                constant_values=-1.0
            )

            ix_p = ix + radius
            iy_p = iy + radius
            iz_p = iz + radius

            patch = padded[
                ix_p - radius: ix_p + radius + 1,
                iy_p - radius: iy_p + radius + 1,
                iz_p - radius: iz_p + radius + 1,
            ]
        else:
            # 2D: kartet har Z=1 (X, Y, 1).
            # Pad bare X og Y, ikke Z.
            padded = np.pad(
                map_3d,
                pad_width=((radius, radius), (radius, radius), (0, 0)),
                mode="constant",
                constant_values=-1.0
            )

            ix_p = ix + radius
            iy_p = iy + radius
            # iz er 0 i praksis, og vi padder ikke Z, så iz_p = iz
            iz_p = iz

            # Ta bare ett lag i Z-retning, men behold Z-dimensjonen (size 1)
            patch = padded[
                ix_p - radius: ix_p + radius + 1,
                iy_p - radius: iy_p + radius + 1,
                iz_p: iz_p + 1
            ]
        
    

        if normalize:
            patch = (patch + 1.0) / 2.0

        return patch

    
    def _pos_to_idx(self, x, y, z=0.0):
        """
        returns values from 0 to max-1, e.g. 0-40
        """
        ix = int(round(x / self.cell_size)) + self.offset[0]
        iy = int(round(y / self.cell_size)) + self.offset[1]
        iz = int(round(z / self.cell_size)) + self.offset[2]

        ix = np.clip(ix, 0, self.map_size_x - 1)
        iy = np.clip(iy, 0, self.map_size_y - 1)
        iz = np.clip(iz, 0, self.map_size_z - 1)

        return ix, iy, iz

    def _spawn_near_plume(self):
        """
        Start inside the bounding box of the above-threshold cells, grown by spawn_margin cells,
        so the plume agent begins near a plume like after a hand-over from the exploration agent.
        Uses the true plume map (also in GP mode); the agent never observes it.
        """
        xs, ys, _ = np.nonzero(self.GT_c_over_threshold_maps)
        if len(xs) == 0:
            return # no cell above threshold: keep the random start
        lim_x = int(self.domain_limit["x"] - 1) # same start limits as the random start in reset()
        lim_y = int(self.domain_limit["y"] - 1)
        # map indices -> positions (cell_size = 1)
        x_lo = max(xs.min() - self.offset[0] - self.spawn_margin, -lim_x)
        x_hi = min(xs.max() - self.offset[0] + self.spawn_margin, lim_x)
        y_lo = max(ys.min() - self.offset[1] - self.spawn_margin, -lim_y)
        y_hi = min(ys.max() - self.offset[1] + self.spawn_margin, lim_y)
        self.state["x"] = np.random.randint(x_lo, x_hi + 1)
        self.state["y"] = np.random.randint(y_lo, y_hi + 1)

   
    def reset(self, *, seed=None, options=None,goal_distance=None):
        super().reset(seed=seed)

        self._visited_map_version = 0
        self._cached_visit_version = -1

        self.c_border_coverage_was_already_below = False
        self.c_plume_coverage_was_already_below = False
        self.agent_rises = False

        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "concentration": 0.0,
            "theta": 0,
            "visited_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool),
            "c_over_threshold_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool),
            "c_around_threshold_map": np.zeros((self.map_size_x,self.map_size_y,self.map_size_z,1), dtype = bool),
        }
       
        
        
        if self.train==True or self.random_points==True:
            """
            set random start concentration within domain limits
            set random initial theta
            """
            lim_y=self.domain_limit["y"]-1.0
            lim_x=self.domain_limit["x"]-1.0
            #randomise start position of the robot
            self.state["x"] = np.random.randint(-np.array(lim_x),np.array(lim_x)+1) # random start position on the grid, within limits, +1 because of how np.random.randint works
            self.state["y"] = np.random.randint(-np.array(lim_y),np.array(lim_y)+1) 
            self.state["z"] = 0
            # z fixed at 0 for 2D plane
            #... later add random z start within limits

            # random new gaussian center
            self.N_gaussians = np.random.randint(self.multiple_gaussians[0], self.multiple_gaussians[1] + 1)
            z_3D_up_to_2 = np.clip(self.domain_limit["z"],0,2)
            self.gaussian_centers = np.random.uniform(
            low=-np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]-z_3D_up_to_2]),
            high=np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]-z_3D_up_to_2]),
            size=(self.N_gaussians, 3))
            # random plume size
            if self.vary_plume_size:
                self.gaussian_sigma = np.random.uniform(*self.gaussian_sigma_range, size=self.N_gaussians)
            # random start orientation
            #self.state["theta"] = np.random.uniform(-np.pi, np.pi)

            # simple random start orientation from set angles
            self.state["theta"] = np.random.choice([
                0.0,
                0.5*np.pi,
                1.0*np.pi,
                1.5*np.pi
            ])

        self.conc = self.get_concentration(self.x, self.y, self.z)
        self._set_corner_cells_visited()
        mask_above = (self.conc >= self.c_threshold) 

        self.GT_c_over_threshold_maps = np.where(mask_above, 1, 0).astype(bool)
        
        mask_around = (self.conc <= self.c_threshold + self.c_around_width) & (self.conc >= self.c_threshold - self.c_around_width)

        self.GT_c_around_threshold_maps = np.where(mask_around, 1, 0).astype(bool)
        
        if self.spawn_near_plume and (self.train or self.random_points):
            self._spawn_near_plume() # needs the plume maps above, and must run before anything below reads the start position
        
        if self._3D:
            self.maxN_over_thresh=(np.sum(self.GT_c_over_threshold_maps)+8).astype(np.float32) # careful! only in z plane
        else:
            self.maxN_over_thresh=(np.sum(self.GT_c_over_threshold_maps)+4).astype(np.float32)
        if self._3D:
            self.maxN_around_thresh=(np.sum(self.GT_c_around_threshold_maps)+8).astype(np.float32) # careful! only in z plane
        else:
            self.maxN_around_thresh=(np.sum(self.GT_c_around_threshold_maps)+4).astype(np.float32) # careful! only in z plane
        

        # Cache downsampled GT maps (they don't change during episode)
        self._gt_over_downsampled = np.array([
            *self.downsample_map(self.GT_c_over_threshold_maps).flatten()
        ]).astype(np.float32)
        self._gt_around_downsampled = np.array([
            *self.downsample_map(self.GT_c_around_threshold_maps).flatten()
        ]).astype(np.float32)
        
        # Reset caches
        self._visited_downsampled = None
        if hasattr(self, '_cached_free'):
            del self._cached_free
        # -- visited map update --
        ix, iy, iz = self._pos_to_idx(
            self.state["x"],
            self.state["y"],
            self.state["z"]
        )
        self.state["concentration"] = self.conc[ix, iy, iz]
        if not self.GP_ON:
            cx, cy, cz = np.unravel_index(np.argmax(self.conc), self.conc.shape)
            self.distance_from_max = np.sqrt((ix -cx)**2 + (iy-cy)**2 + (iz-cz)**2) / np.sqrt(self.domain_limit["x"]**2 +self.domain_limit["y"]**2 +self.domain_limit["z"]**2)
        else:
            self.distance_from_max = 1.0 # GP will handle this
        self.disturbance_dist = self.dynamics.reset() # doesn't do anything rn, cause disturbances are commented out
        
        
        self.local_GT=self._get_local_GT_patch(ix,iy,iz,self.patch_radius)
        self.local_GT_around=self._get_local_GT_around_patch(ix,iy,iz,self.patch_radius)
        # mark visited BEFORE observation
        self.state["visited_map"][ix, iy, iz, :] = 1

        visited_patch = self._get_local_visited_patch(ix, iy,iz,self.patch_radius)
        if self.use_c_map:
            over_c_patch = self._get_local_over_c_patch(ix, iy,iz,self.patch_radius)
            around_c_patch = self._get_local_around_c_patch(ix, iy,iz,self.patch_radius)
        else:
            over_c_patch = self._zero_c_patch
            around_c_patch = self._zero_c_patch

        x_norm = (self.state["x"] ) / (self.domain_limit["x"]) # normalise to [-1,1]
        y_norm = (self.state["y"] ) / (self.domain_limit["y"]) # normalise to [-1,1]
        if self._3D:
            z_norm = (self.state["z"] ) / (self.domain_limit["z"]) # as the space is reduced to 2D plane# (self.state["z"] ) / (self.domain_limit["z"]) # normalise to [-1,1]
        else:
            z_norm = 0.0

        

        

        #theta_norm = (self.state["theta"]/np.pi) /(2) #normalise to [0,1], with 1 being 1.5pi
        concentration = self.state["concentration"]  # already in [0,1]


        free = (self.state["visited_map"][:,:,:,0]).astype(np.float32) # the unvisited fields
        
        if concentration >= self.c_threshold:
            self.state["c_over_threshold_map"][ix, iy, iz, 0] = 1 # threshold for over threshold map, can be adapted
        if concentration<=self.c_threshold +self.c_around_width and concentration>=self.c_threshold-self.c_around_width:
                self.state["c_around_threshold_map"][ix, iy, iz, 0] = 1
        if self.use_c_map:
            c_over_threshold = self.state["c_over_threshold_map"][:,:,:,0] # fatching map
            
            c_around_threshold = self.state["c_around_threshold_map"][:,:,:,0] # fatching map
           
            
        else:
            c_over_threshold = self._zero_c_map
            c_around_threshold = self._zero_c_map
            
        if np.isclose(self.state["theta"],0.0,atol=1e-4) or np.isclose(self.state["theta"],2*np.pi,atol=1e-4):
            theta_encoded = [1,0,0,0]
        elif np.isclose(self.state["theta"],0.5*np.pi,atol=1e-4):
            theta_encoded = [0,1,0,0]
        elif np.isclose(self.state["theta"],1.0*np.pi,atol=1e-4):
            theta_encoded = [0,0,1,0]
        else:
            theta_encoded = [0,0,0,1]

        obs = {
            "obs_state": np.array([
                x_norm,
                y_norm,
                z_norm,
                theta_encoded[0],
                theta_encoded[1],
                theta_encoded[2],
                theta_encoded[3],
                concentration,
                *visited_patch.flatten().astype(np.float32),
                *over_c_patch.flatten().astype(np.float32),
                *around_c_patch.flatten().astype(np.float32),
            ], dtype=np.float32),
            "visited_maps_downsampled": np.array([*self.downsample_map(free).flatten()]).astype(np.float32),# shape= (batch,X;Y;Z) #!! in this way the CNN does not get an information about where not to move!
            #"c_over_threshold_maps": c_over_threshold.astype(np.float32) ,
            #"c_around_threshold_maps": c_around_threshold.astype(np.float32) ,
            "GT_c_over_threshold_maps_downsampled": self._gt_over_downsampled,
            "GT_c_around_threshold_maps_downsampled": self._gt_around_downsampled,  
            "local_GT": self.local_GT.flatten(),
            "local_GT_around": self.local_GT_around.flatten(),
            "space_coverage": np.array([np.sum(self.state["visited_map"][:,:,:,0])/self.max_return], dtype=np.float32),
            "plume_coverage": np.array([np.sum(c_over_threshold)/self.maxN_over_thresh if self.maxN_over_thresh>0.0 else 1.0], dtype=np.float32),
            "border_coverage": np.array([np.sum(c_around_threshold)/self.maxN_around_thresh if self.maxN_around_thresh>0.0 else 1.0], dtype=np.float32),
            "distance_from_max": np.array([self.distance_from_max],dtype=np.float32),
            #"VAR_visited_downsampled": np.array([0.0],dtype=np.float32),
        }

        self.MEAN_uncertainty = 1.0
        return obs, {}

    def step(self, action):
        # Cache old state
        old_x, old_y, old_z = self.state["x"], self.state["y"], self.state["z"]
        old_theta = self.state["theta"]
        old_conc = self.state["concentration"]
        old_ix, old_iy, old_iz = self._pos_to_idx(old_x, old_y, old_z)
        
        # Store metrics before update
        self.old_visited_sum = np.sum(self.state["visited_map"])
        self.old_c_around_sum = np.sum(self.state["c_around_threshold_map"])
        self.old_c_over_sum = np.sum(self.state["c_over_threshold_map"])
        
        # Update dynamics
        self.agent_turns = action in (3, 4)
        self.agent_rises = action in (1,2)
        self.dynamics.step(self.state, self.move_vectors[action], self.conc)
        self.state["theta"] = self._snap_theta(self.state["theta"])
        
        # Check boundaries (vectorized check)
        x, y, z, theta = self.state["x"], self.state["y"], self.state["z"], self.state["theta"]
        if self._check_boundary_violation(x, y, z, theta):
            return self._handle_boundary_violation(
                old_x, old_y, old_z, old_theta, old_conc, 
                old_ix, old_iy, old_iz, self.old_visited_sum
            )
        
        # Get new position
        ix, iy, iz = self._pos_to_idx(x, y, z)
        if not self.GP_ON:
            cx, cy, cz = np.unravel_index(np.argmax(self.conc), self.conc.shape)
            
            # Update distance to max (vectorized)
            self.distance_from_max = np.sqrt(
                (ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2
            ) / np.sqrt(
                self.domain_limit["x"]**2 + 
                self.domain_limit["y"]**2 + 
                self.domain_limit["z"]**2
            )
        else:
            self.distance_from_max = 1.0 # GP will handle this
        
        self.state["concentration"] = self.conc[ix, iy, iz]
        conc = self.state["concentration"]
        
        # === VECTORIZED 3D EXPANSION ===
        # Mark visited and concentration maps in one vectorized operation
        self._update_visited_and_concentration_maps(
            old_ix, old_iy, old_iz, ix, iy, iz, conc
        )
        #print(f"VISITED:{np.sum(self.state["visited_map"][:,:,:,0])}, C_OVER:{ np.sum(self.state["c_over_threshold_map"][:,:,:,0])},C_ARUND:{ np.sum(self.state["c_around_threshold_map"][:,:,:,0])}")
        
        # Cache local patches (avoid recalculation)
        visited_patch = self._get_local_visited_patch(ix, iy, iz, self.patch_radius)
        if self.use_c_map:
            over_c_patch = self._get_local_over_c_patch(ix, iy, iz, self.patch_radius)
            around_c_patch = self._get_local_around_c_patch(ix, iy, iz, self.patch_radius)

        else:
            over_c_patch = self._zero_c_patch
            around_c_patch = self._zero_c_patch
        
        
        # Update local GT patches only if position changed significantly
        if (ix != old_ix) or (iy != old_iy) or (iz != old_iz):
            self.local_GT = self._get_local_GT_patch(ix, iy, iz, self.patch_radius)
            self.local_GT_around = self._get_local_GT_around_patch(ix, iy, iz, self.patch_radius)
        
        # === OPTIMIZED OBSERVATION CONSTRUCTION ===
        obs = self._build_observation_fast(
            x, y, z, theta, conc,
            visited_patch, over_c_patch, around_c_patch,
            self.old_visited_sum, self.old_c_over_sum, self.old_c_around_sum
        )
        
        # === REWARD CALCULATION ===
        reward, terminated, info = self._compute_reward_and_termination(
            obs, self.old_visited_sum, self.old_c_over_sum, self.old_c_around_sum
        )
        
        return obs, reward, terminated, False, info

    def _check_boundary_violation(self, x, y, z, theta):
        """Vectorized boundary check"""
        eps = 1e-3
        x_lim, y_lim, z_lim = self.domain_limit["x"], self.domain_limit["y"], self.domain_limit["z"]
        
        # Check orthogonal borders
        t_mod = theta % (2 * np.pi)
        x_plus = np.isclose(x, x_lim, atol=eps) and np.isclose(t_mod, 0.0, atol=eps)
        x_minus = np.isclose(x, -x_lim, atol=eps) and np.isclose(t_mod, np.pi, atol=eps)
        y_plus = np.isclose(y, y_lim, atol=eps) and np.isclose(t_mod, 0.5*np.pi, atol=eps)
        y_minus = np.isclose(y, -y_lim, atol=eps) and np.isclose(t_mod, 1.5*np.pi, atol=eps)
        
        return (x_plus or x_minus or y_plus or y_minus or 
                abs(x) > x_lim + self.eps or 
                abs(y) > y_lim + self.eps or 
                abs(z) > z_lim + self.eps)

    def _update_visited_and_concentration_maps(self, ox, oy, oz, nx, ny, nz, conc):
        visited = self.state["visited_map"][:,:,:,0]
        if self.use_c_map and not self.GP_ON:
            c_over = self.state["c_over_threshold_map"][:,:,:,0]
            c_around = self.state["c_around_threshold_map"][:,:,:,0]
        
        # Mark path with 3x3x3 neighborhood around both old and new positions
        for cx, cy, cz in [(ox, oy, oz), (nx, ny, nz)]:
            x_min, x_max = max(0, cx-1), min(self.map_size_x-1, cx+1)
            y_min, y_max = max(0, cy-1), min(self.map_size_y-1, cy+1)
            z_min, z_max = max(0, cz-1), min(self.map_size_z-1, cz+1)
            
            # Check which cells are actually new to count them
            slice_view = visited[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
            newly_visited = ~slice_view  # boolean mask of newly visited cells
            num_new = np.sum(newly_visited)
            
            if num_new > 0:
                slice_view[:] = True
                self._visited_map_version += num_new
                
                if self.use_c_map and not self.GP_ON: #we do not need this in GP mode
                    conc_slice = self.conc[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                    c_over_view = c_over[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                    c_around_view = c_around[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                    
                    c_over_view |= (conc_slice >= self.c_threshold)
                    c_around_view |= ((conc_slice <= self.c_threshold + self.c_around_width) & 
                                    (conc_slice >= self.c_threshold - self.c_around_width))
        
    def _mark_visited(self, ix, iy, iz):
        if not self.state["visited_map"][ix, iy, iz, 0]:
            self.state["visited_map"][ix, iy, iz, 0] = 1
            self._visited_map_version += 1

    def _build_observation_fast(self, x, y, z, theta, conc, 
                            visited_patch, over_c_patch, around_c_patch,
                            old_visited_sum, old_c_over_sum, old_c_around_sum):
        """Optimized observation building with pre-allocated buffers"""
        free = self.state["visited_map"][:,:,:,0].astype(np.float32)
        # Normalize position (avoid division by zero)
        x_norm = x / self.domain_limit["x"]
        y_norm = y / self.domain_limit["y"]
        if self._3D:
            z_norm = z / self.domain_limit["z"]
        else:
            z_norm = 0.0

        # Theta one-hot (vectorized)
        theta_enc = [0, 0, 0, 0]
        t_mod = theta % (2 * np.pi)
        if np.isclose(t_mod, 0.0, atol=1e-4) or np.isclose(t_mod, 2*np.pi, atol=1e-4):
            theta_enc[0] = 1
        elif np.isclose(t_mod, 0.5*np.pi, atol=1e-4):
            theta_enc[1] = 1
        elif np.isclose(t_mod, 1.0*np.pi, atol=1e-4):
            theta_enc[2] = 1
        else:
            theta_enc[3] = 1
        
        # Build obs_state vector efficiently
        buffer = self._obs_state_buffer
        buffer[0] = x_norm
        buffer[1] = y_norm
        buffer[2] = z_norm
        buffer[3:7] = theta_enc
        buffer[7] = conc
        
        # Flatten patches directly into buffer (avoid temp arrays)
        vp_flat = visited_patch.flatten()
        oc_flat = over_c_patch.flatten()
        ac_flat = around_c_patch.flatten()
        
        patch_size = len(vp_flat)
        buffer[8:8+patch_size] = vp_flat
        buffer[8+patch_size:8+2*patch_size] = oc_flat
        buffer[8+2*patch_size:8+3*patch_size] = ac_flat
        
        if self._visited_downsampled is None or self._visited_map_version != getattr(self, '_cached_visit_version', -1):
            self._visited_downsampled = self.downsample_map(free).flatten()
            self._cached_visit_version = self._visited_map_version
        
        # GT maps are static per episode - cache them in reset()
        c_over = self.state["c_over_threshold_map"][:,:,:,0] if self.use_c_map else self._zero_c_map
        c_around = self.state["c_around_threshold_map"][:,:,:,0] if self.use_c_map else self._zero_c_map
        
        # Calculate coverages
        space_cov = np.sum(free) / self.max_return
        plume_cov = np.sum(c_over) / self.maxN_over_thresh if self.maxN_over_thresh > 0.0 else 1.0
        border_cov = np.sum(c_around) / self.maxN_around_thresh if self.maxN_around_thresh > 0.0 else 1.0
        
        #VAR = np.var(self._visited_downsampled)
      
        return {
            "obs_state": buffer.copy(),  # Return copy to avoid reference issues
            "visited_maps_downsampled": self._visited_downsampled.copy(),
            "GT_c_over_threshold_maps_downsampled": self._gt_over_downsampled,
            "GT_c_around_threshold_maps_downsampled": self._gt_around_downsampled,
            "local_GT": self.local_GT.flatten(),
            "local_GT_around": self.local_GT_around.flatten(),
            "space_coverage": np.array([space_cov], dtype=np.float32),
            "plume_coverage": np.array([plume_cov], dtype=np.float32),
            "border_coverage": np.array([border_cov], dtype=np.float32),
            "distance_from_max": np.array([self.distance_from_max], dtype=np.float32),
            #"VAR_visited_downsampled": np.array([VAR], dtype=np.float32),
        }

    def _compute_reward_and_termination(self, obs, old_visited_sum, old_c_over_sum, old_c_around_sum):
        """Separated reward logic for clarity"""
        reward = 0.0
        terminated = False
        info = {"visited_states_count": obs["space_coverage"][0]}
        
        if self.GP_ON or self.HRL:
            return reward, terminated, info
        
        current_space = obs["space_coverage"][0]
        current_plume = obs["plume_coverage"][0]
        current_border = obs["border_coverage"][0]
        
        # Add specific metrics to info
        if self.agent_type in ("BORDER", "META"):
            info["current_around_threshold"] = current_border
        if self.agent_type in ("PLUME", "META"):
            info["current_above_threshold"] = current_plume
        
        # SPACE agent
        if self.agent_type == "SPACE":
            VAR_downsampled = None#obs["VAR_visited_downsampled"][0]
            new_cells = np.sum(self.state["visited_map"][:,:,:,0]) - old_visited_sum
            reward = self.reward_fn_1explore.get_reward(
                self.agent_turns, current_space, new_cells, 
                _3D=self._3D, agent_rises=self.agent_rises, VAR_downsampled=VAR_downsampled
            )
            if current_space >= self.accuracy_agent_goals[0]:
                reward = 40.0
                terminated = True
        
        # PLUME agent
        elif self.agent_type == "PLUME":
            new_cells = np.sum(self.state["c_over_threshold_map"]) - old_c_over_sum
            reward = self.reward_fn_2plume.get_reward(
                self.agent_turns, current_plume, new_cells,
                SUBAGENT_TRAIN_ON_GP=self.GP_ON, 
                diff_new_cells=None, found_source=None,
                _3D=self._3D, agent_rises=self.agent_rises,percentage_visited=current_space
            )
            if current_plume >= self.accuracy_agent_goals[1] and self.c_plume_coverage_was_already_below:
                reward = 40.0
                terminated = True
            elif current_plume < self.accuracy_agent_goals[1]:
                self.c_plume_coverage_was_already_below = True
        
        # BORDER agent
        elif self.agent_type == "BORDER":
            new_cells = np.sum(self.state["c_around_threshold_map"]) - old_c_around_sum
            reward = self.reward_fn_3border.get_reward(
                self.agent_turns, current_border, new_cells,
                SUBAGENT_TRAIN_ON_GP=self.GP_ON,
                diff_new_cells=None, found_source=None,
                _3D=self._3D, agent_rises=self.agent_rises,percentage_visited=current_space
            )
            if current_border >= self.accuracy_agent_goals[2] and self.c_border_coverage_was_already_below:
                reward = 40.0
                terminated = True
            elif current_border < self.accuracy_agent_goals[2]:
                self.c_border_coverage_was_already_below = True
        
        # META agent
        elif self.agent_type == "META":
            visited_count = np.sum(self.state["visited_map"])
            around_count = np.sum(self.state["c_around_threshold_map"])
            over_count = np.sum(self.state["c_over_threshold_map"])
            
            if visited_count >= self.accuracy_agent_goals[0] * self.max_return:
                reward = 40.0
                terminated = True
            elif (around_count >= self.accuracy_agent_goals[1] * self.maxN_around_thresh and 
                over_count >= self.accuracy_agent_goals[2] * self.maxN_over_thresh and 
                self.maxN_around_thresh > 0.0 and 
                self.maxN_over_thresh > 0.0):
                reward = 40.0
                terminated = False
            else:
                reward = 0.0
        
        return reward, terminated, info

    def _handle_boundary_violation(self, old_x, old_y, old_z, old_theta, old_conc, 
                               old_ix, old_iy, old_iz, old_visited_sum):
        """Revert to old state and return penalty observation"""
        self.state["x"] = old_x
        self.state["y"] = old_y
        self.state["z"] = old_z
        self.state["theta"] = old_theta
        self.state["concentration"] = old_conc
        
        # Build observation from old state
        visited_patch = self._get_local_visited_patch(old_ix, old_iy, old_iz, self.patch_radius)
        if self.use_c_map:
            over_c_patch = self._get_local_over_c_patch(old_ix, old_iy, old_iz, self.patch_radius)
            around_c_patch = self._get_local_around_c_patch(old_ix, old_iy, old_iz, self.patch_radius)
        else:
            over_c_patch = self._zero_c_patch
            around_c_patch = self._zero_c_patch
        
        # Reuse observation building
        obs = self._build_observation_fast(
            old_x, old_y, old_z, old_theta, old_conc,
            visited_patch, over_c_patch, around_c_patch,
            old_visited_sum, 
            np.sum(self.state["c_over_threshold_map"]),
            np.sum(self.state["c_around_threshold_map"])
        )
        
        info = {"visited_states_count": old_visited_sum / self.max_return}
        if self.agent_type in ("BORDER", "META"):
            info["current_around_threshold"] = obs["border_coverage"][0]
        if self.agent_type in ("PLUME", "META"):
            info["current_above_threshold"] = obs["plume_coverage"][0]
        
        return obs, np.float32(-4.0), False, False, info


    def _set_corner_cells_visited(self):
        """
        Sett hjørnene i kartet som besøkt (=1).
        Hvis use_c_map er True, sett også c_over_threshold_map = 1 i hjørnene.
        Gjelder 4 hjørner i 2D og 8 hjørner i 3D.
        """
        # Indeksene til hjørnene: 0 og max-indeks i hver dimensjon
        max_x = self.map_size_x - 1
        max_y = self.map_size_y - 1
        max_z = self.map_size_z - 1

        if self._3D:
            # 8 hjørner i 3D
            corners = [
                (0,      0,      0),
                (0,      0,      max_z),
                (0,      max_y,  0),
                (0,      max_y,  max_z),
                (max_x,  0,      0),
                (max_x,  0,      max_z),
                (max_x,  max_y,  0),
                (max_x,  max_y,  max_z),
            ]
        else:
            # 4 hjørner i 2D (z = 0)
            corners = [
                (0,      0,      0),
                (max_x,  0,      0),
                (0,      max_y,  0),
                (max_x,  max_y,  0),
            ]

        for (ix, iy, iz) in corners:
            # visited_map
            self.state["visited_map"][ix, iy, iz, 0] = True

            # oppdater visited-versjonsteller (brukes til caching av downsampled map)
            self._visited_map_version += 1

            if self.use_c_map:
                # Tving hjørnene til over-threshold = 1
                self.state["c_over_threshold_map"][ix, iy, iz, 0] = True
                # Hvis du også vil at "around" skal være 1, kan du legge til:
                self.state["c_around_threshold_map"][ix, iy, iz, 0] = True


    def render(self):
        self.renderer.render(self.model_path)
        # self.renderer.plot_concentration_field(self.concentration_function ,self.gaussian_centers[0], self.gaussian_sigma)
        self.renderer.plot_concentration_field2(self.x,self.y,self.z,self.conc,self.c_threshold-self.c_around_width)

    def step_sim(self):
        self.renderer.step_sim(self.state)#,self.old_state,self.action) # forward also local patch of visited map

        """
        define here how to plot density field and density estimation
        """
        