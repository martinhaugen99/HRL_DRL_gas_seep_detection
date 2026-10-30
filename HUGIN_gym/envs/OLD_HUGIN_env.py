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
        self.gaussian_centers = np.random.uniform(
        low=-np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]]),
        high=np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]]),
        size=(self.N_gaussians, 3))
        # (x0, y0, z0)
        self.gaussian_sigma = 3.5   # spread of the Gaussian blob
        self.gaussian_amplitude = 1.0  # peak concentration
        self.c_threshold = 0.3 # threshold for over threshold map, can be adapted
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

        self._zero_c_patch = np.zeros_like(self._get_local_over_c_patch(1, 1,self.patch_radius))
        
        
        

        ##!###!!!##!!!##!!!##!!!!
        CHANGE_THIS_PLEASE_TO_KERNEL_SIZE_AND_STRIDE_FROM_GP_WRAPPER = True
        ##!###!!!##!!!##!!!##!!!!
        
        self.kernel_size, self.stride = 3,2
        dummy = np.zeros((self.map_size_z, self.map_size_x, self.map_size_y), dtype=np.float32)
        downsampled = self.downsample_map(dummy)
        obs_map_shape = np.array([*downsampled.flatten()]).shape
        
        # -- discrete observation space -- DQN #
        self.observation_space = spaces.Dict({#155
            "obs_state": spaces.Box(low=0, high=1, shape=(155,), dtype=np.float32), # x,y,z, [_ _ _ _], c = 8 params (pos., orientation, conc.) +49 local patch visited +49 local over thresh patch + 49 local patch around thresh
            "visited_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32), # Z,X,Y # here a gigantic version of visited_patch
            #"c_over_threshold_maps_counter": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            #"c_around_threshold_maps_counter": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "GT_c_over_threshold_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "GT_c_around_threshold_maps_downsampled": spaces.Box(low=0, high=1, shape=obs_map_shape, dtype=np.float32),
            "local_GT": spaces.Box(low=0, high=1, shape=(49,), dtype=np.float32),
            "local_GT_around": spaces.Box(low=0, high=1, shape=(49,), dtype=np.float32),
            "space_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "plume_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "border_coverage": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
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
        self.counter_above_threshold = 0 # initially a consecutive counter. Now only a relict

        self.x = np.arange(-self.map_size_x//2, self.map_size_x//2)[:, None, None]
        self.y = np.arange(-self.map_size_y//2, self.map_size_y//2)[None, :, None]
        self.z = np.arange(-self.map_size_z//2, self.map_size_z//2)[None, None, :]
        
        self.conc = self.get_concentration(self.x, self.y, self.z)
        
        # debugging for finding out how many steps may be needed
        #print(f"fields above threshold at start: {np.sum(self.conc>=self.c_threshold)}, around threshold and above: {np.sum( (self.conc >= self.c_threshold - self.c_around_width))}")

        mask_above = (self.conc >= self.c_threshold) 

        self.GT_c_over_threshold_maps = np.transpose(
            np.where(mask_above, 1, 0),
            (2, 0, 1)
        ).astype(bool)
        
        mask_around = (self.conc <= self.c_threshold + self.c_around_width) & (self.conc >= self.c_threshold - self.c_around_width)

        self.GT_c_around_threshold_maps = np.transpose(
            np.where(mask_around, 1, 0),
            (2, 0, 1)
        ).astype(bool)
        self.local_GT_around=self._get_local_GT_around_patch(ix,iy,self.patch_radius)
        self.local_GT=self._get_local_GT_patch(ix,iy,self.patch_radius)
        self.maxN_over_thresh=np.sum(self.GT_c_over_threshold_maps) # careful! only in z plane
        self.maxN_around_thresh=np.sum(self.GT_c_around_threshold_maps) # careful! only in z plane
        self.GP_ON=False # default, no GP active
        self.c_border_coverage_was_already_below = False
        self.c_plume_coverage_was_already_below = False
        
        self.accuracy_agent_goals = [0.9,0.9,0.9]
        self.HRL = False
        
    # -- Define the concentration function -- #   
    def concentration_function(self, x, y, z):
            centers = np.array(self.gaussian_centers)  # (N, 3)
            sigma = self.gaussian_sigma
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

        x = self.avg_pool2d_numpy(x, self.kernel_size, self.stride)
        x = self.avg_pool2d_numpy(x, self.kernel_size, self.stride)
        x = self.avg_pool2d_numpy(x, self.kernel_size, self.stride)

        return x
    def avg_pool2d_numpy(self,x, kernel_size, stride):
            """
            x: (H, W) or (C, H, W)
            """
            if x.ndim == 2:
                x = x[None, ...]  # add channel dim

            C, H, W = x.shape

            out_h = (H - kernel_size) // stride + 1
            out_w = (W - kernel_size) // stride + 1

            shape = (C, out_h, out_w, kernel_size, kernel_size)
            strides = (
                x.strides[0],
                stride * x.strides[1],
                stride * x.strides[2],
                x.strides[1],
                x.strides[2],
            )

            windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
            return windows.mean(axis=(3, 4))
    
    def calculate_distance_to_closest_unvisited(self,agent_pos, visited_map):
        """
        Calculate the Manhattan distance from the agent's transformed position to the closest unvisited cell.
        """
        # Transform the agent position to the [0, N-1] range
        transformed_agent_pos = agent_pos + (np.array(visited_map.shape)//2)
        
        # Find all unvisited cells (value 0 in the visited map)
        unvisited_cells = np.argwhere(~visited_map)  # Find all unvisited cells (value 0)
        
        # Compute the distances from the agent's transformed position to all unvisited cells
        distances = np.abs(unvisited_cells - transformed_agent_pos).sum(axis=1)

        #-- HERE PLEASE euclidean distance when going to continuous space!
        #diff = unvisited_cells - transformed_agent_pos
        #distances = np.sum(diff**2, axis=1)

        # Return the minimum distance
        return min(distances) if distances.any() else 0
  
    
    def _get_local_GT_patch(self, ix, iy, radius=2):
        return self.get_local_patch(
            self.GT_c_over_threshold_maps[0],
            ix, iy, radius
        )
    def _get_local_GT_around_patch(self, ix, iy, radius=2):
        return self.get_local_patch(
            self.GT_c_around_threshold_maps[0],
            ix, iy, radius
        )
    def _get_local_visited_patch(self, ix, iy, radius=2):
        visited = self.state["visited_map"][:, :, 0, 0]
        return self.get_local_patch(visited, ix, iy, radius)

    def _get_local_over_c_patch(self, ix, iy, radius=2):
        return self.get_local_patch(
            self.state["c_over_threshold_map"][:, :, 0, 0],
            ix, iy, radius
        )
    def _get_local_around_c_patch(self, ix, iy, radius=2):
        return self.get_local_patch(
            self.state["c_around_threshold_map"][:, :, 0, 0],
            ix, iy, radius
        )
    def get_local_patch(self, map_2d, ix, iy, radius=2, normalize=True):
        """
        Generic local patch extractor.

        Args:
            map_2d: 2D numpy array
            ix, iy: center index
            radius: patch radius
            normalize: whether to map [-1,1] -> [0,1]

        Returns:
            patch: (2r+1, 2r+1)
        """

        # Ensure float32 (avoid repeated casting elsewhere)
        map_2d = map_2d.astype(np.float32)

        # Pad once
        padded = np.pad(
            map_2d,
            pad_width=radius,
            mode="constant",
            constant_values=-1.0
        )

        # Shift indices because of padding
        ix_p = ix + radius
        iy_p = iy + radius

        # Slice patch (vectorized, no loops)
        patch = padded[
            ix_p - radius: ix_p + radius + 1,
            iy_p - radius: iy_p + radius + 1
        ]

        # Flip (your convention)
        patch = np.flipud(patch)
        patch = np.fliplr(patch)

        # Rotation (if you want it back)
        k = 0  # or: -round(self.state["theta"] / (np.pi / 2))
        patch = np.rot90(patch, k=k)

        if normalize:
            patch = (patch + 1.0) / 2.0

        return patch
    
    def _pos_to_idx(self, x, y, z=0.0):
        ix = int(round(x / self.cell_size)) + self.offset[0]
        iy = int(round(y / self.cell_size)) + self.offset[1]
        iz = int(round(z / self.cell_size)) + self.offset[2]

        ix = np.clip(ix, 0, self.map_size_x - 1)
        iy = np.clip(iy, 0, self.map_size_y - 1)
        iz = np.clip(iz, 0, self.map_size_z - 1)

        return ix, iy, iz

   
    def reset(self, *, seed=None, options=None,goal_distance=None):
        super().reset(seed=seed)

        self.c_border_coverage_was_already_below = False
        self.c_plume_coverage_was_already_below = False
       

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
        
        self.counter_above_threshold = 0
        
        
        if self.train==True or self.random_points==True:
            """
            set random start concentration within domain limits

            +

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
            self.gaussian_centers = np.random.uniform(
            low=-np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]]),
            high=np.array([self.domain_limit["x"]-2, self.domain_limit["y"]-2, self.domain_limit["z"]]),
            size=(self.N_gaussians, 3))
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

        mask_above = (self.conc >= self.c_threshold) 

        self.GT_c_over_threshold_maps = np.transpose(
            np.where(mask_above, 1, 0),
            (2, 0, 1)
        ).astype(bool)
        
        mask_around = (self.conc <= self.c_threshold + self.c_around_width) & (self.conc >= self.c_threshold - self.c_around_width)

        self.GT_c_around_threshold_maps = np.transpose(
            np.where(mask_around, 1, 0),
            (2, 0, 1)
        ).astype(bool)
        #print(f"OVER{self.maxN_over_thresh}, AROUND ={self.maxN_around_thresh}")
        
        self.maxN_over_thresh=np.sum(self.GT_c_over_threshold_maps).astype(np.float32) # carefuL! only in z plane
        self.maxN_around_thresh=np.sum(self.GT_c_around_threshold_maps).astype(np.float32) # carefuL! only in z plane
        # -- visited map update --
        ix, iy, iz = self._pos_to_idx(
            self.state["x"],
            self.state["y"],
            self.state["z"]
        )
        self.state["concentration"] = self.conc[ix, iy, iz]
        
        self.disturbance_dist = self.dynamics.reset() # doesn't do anything rn, cause disturbances are commented out
        
        
        self.local_GT=self._get_local_GT_patch(ix,iy,self.patch_radius)
        self.local_GT_around=self._get_local_GT_around_patch(ix,iy,self.patch_radius)
        # mark visited BEFORE observation
        self.state["visited_map"][ix, iy, iz, :] = 1

        visited_patch = self._get_local_visited_patch(ix, iy,self.patch_radius)
        if self.use_c_map:
            over_c_patch = self._get_local_over_c_patch(ix, iy,self.patch_radius)
            around_c_patch = self._get_local_around_c_patch(ix, iy,self.patch_radius)
        else:
            over_c_patch = self._zero_c_patch
            around_c_patch = self._zero_c_patch

        x_norm = (self.state["x"] ) / (self.domain_limit["x"]) # normalise to [-1,1]
        y_norm = (self.state["y"] ) / (self.domain_limit["y"]) # normalise to [-1,1]
        z_norm = 0.0 # as the space is reduced to 2D plane# (self.state["z"] ) / (self.domain_limit["z"]) # normalise to [-1,1]

        

        

        #theta_norm = (self.state["theta"]/np.pi) /(2) #normalise to [0,1], with 1 being 1.5pi
        concentration = self.state["concentration"]  # already in [0,1]


        free = (self.state["visited_map"][:,:,:,0]).astype(np.float32) # the unvisited fields
        free= np.transpose(free,(2,0,1))
        if concentration >= self.c_threshold:
            self.state["c_over_threshold_map"][ix, iy, iz, 0] = 1 # threshold for over threshold map, can be adapted
            if concentration<=self.c_threshold +self.c_around_width:
                self.state["c_around_threshold_map"][ix, iy, iz, 0] = 1
        if self.use_c_map:
            c_over_threshold = self.state["c_over_threshold_map"][:,:,:,0] # fatching map
            c_over_threshold =np.transpose(c_over_threshold,(2,0,1))
            c_around_threshold = self.state["c_around_threshold_map"][:,:,:,0] # fatching map
            c_around_threshold =np.transpose(c_around_threshold,(2,0,1))
            
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

        #k_ind = np.argmax(theta_encoded)
        #free = np.rot90(free,k=-int(k_ind), axes=(0,1))
        #print(np.shape(np.array([*self.downsample_map(free).flatten()])))
        # -- obs for DQN --
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
            "visited_maps_downsampled": np.array([*self.downsample_map(free).flatten()]).astype(np.float32),# shape= (batch,Z,X,Y) #!! in this way the CNN does not get an information about where not to move!
            #"c_over_threshold_maps": c_over_threshold.astype(np.float32) ,
            #"c_around_threshold_maps": c_around_threshold.astype(np.float32) ,
            "GT_c_over_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_over_threshold_maps).flatten()]).astype(np.float32),
            "GT_c_around_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_around_threshold_maps).flatten()]).astype(np.float32),
            "local_GT": self.local_GT.flatten(),
            "local_GT_around": self.local_GT_around.flatten(),
            "space_coverage": np.array([np.sum(self.state["visited_map"][:,:,:,0])/self.max_return], dtype=np.float32),
            "plume_coverage": np.array([np.sum(c_over_threshold)/self.maxN_over_thresh if self.maxN_over_thresh>0.0 else 1.0], dtype=np.float32),
            "border_coverage": np.array([np.sum(c_around_threshold)/self.maxN_around_thresh if self.maxN_around_thresh>0.0 else 1.0], dtype=np.float32),
            
        }

        
        
        

        # -- obs for PPO --
        # obs = {
        #     "x": np.array([self.state["x"]], dtype=np.float32),
        #     "y": np.array([self.state["y"]], dtype=np.float32),
        #     "z": np.array([self.state["z"]], dtype=np.float32),
        #     "theta": np.array([self.state["theta"]], dtype=np.float32),
        #     "concentration": np.array([self.state["concentration"]], dtype=np.float32),
        #     "visited_map": self.state["visited_map"].astype(np.uint8),
        # }

        #obs["concentration"][0] = self.state["concentration"]

        
        
        #self.reward_fn = Reward() # add hear real concentration GP model!!

        return obs, {}

    def step(self, action):
        #print(type(self.maxN_around_thresh),type(self.maxN_over_thresh)) # floats
        old_x = self.state["x"]
        old_y = self.state["y"]
        old_z = self.state["z"]
        old_theta = self.state["theta"]
        old_conc = self.state["concentration"]
        self.old_visited_sum = np.sum(self.state["visited_map"])
        self.old_c_around_sum = np.sum(self.state["c_around_threshold_map"])
        self.old_c_over_sum = np.sum(self.state["c_over_threshold_map"])
        self.dynamics.step(self.state, self.move_vectors[action],self.conc) # we do NOT update concentration in here, as we need to compute otherwise the new ix,iy,iz
        self.state["theta"] = self._snap_theta(self.state["theta"])
        if action ==3 or action==4:
            self.agent_turns = True
        else:
            self.agent_turns = False
        # -- sanity check for termination of the environment --
            # ---- Check termination FIRST ----
        x_plus_boarder = ((np.isclose(self.state["x"], int(self.domain_limit["x"]), atol=1e-3)) and (np.isclose(abs(self.state["theta"]), 0.0,atol=1e-3) or np.isclose(abs(self.state["theta"]), 2*np.pi,atol=1e-3)))
        x_minus_boarder = ((np.isclose(self.state["x"], -int(self.domain_limit["x"]), atol=1e-3)) and (np.isclose(abs(self.state["theta"]), np.pi,atol=1e-3)))
        y_plus_boarder = ((np.isclose(self.state["y"], int(self.domain_limit["y"]), atol=1e-3)) and (np.isclose(abs(self.state["theta"]), 0.5*np.pi,atol=1e-3)))
        y_minus_boarder = ((np.isclose(self.state["y"], -int(self.domain_limit["y"]), atol=1e-3)) and (np.isclose(abs(self.state["theta"]), 1.5*np.pi,atol=1e-3)))

        #print(f"self.state[\"x\"]={self.state['x']}, self.state[\"y\"]={self.state['y']}, self.state[\"theta\"]={self.state['theta']}, domain_limit={self.domain_limit}, x_plus_boarder={x_plus_boarder}, x_minus_boarder={x_minus_boarder}, y_plus_boarder={y_plus_boarder}, y_minus_boarder={y_minus_boarder}")

        if x_plus_boarder or x_minus_boarder or y_plus_boarder or y_minus_boarder : # with angle acceptance, angle_eps

            border_orthogonal = True
        else:
            border_orthogonal = False
        if (
            border_orthogonal or 
            abs(self.state["x"]) > (self.domain_limit["x"] + self.eps) or
            abs(self.state["y"]) > (self.domain_limit["y"] + self.eps) or
            abs(self.state["z"]) > (self.domain_limit["z"] + self.eps) 
            ):
            # Return old state, negative reward, and let the agent re-act
            ix, iy, iz = self._pos_to_idx(
            old_x,
            old_y,
            old_z)
            
            # - revert to old state -
            self.state["x"] = old_x
            self.state["y"] = old_y
            self.state["z"] = old_z
            self.state["theta"] = old_theta
            self.state["concentration"] = old_conc  

            visited_patch = self._get_local_visited_patch(ix, iy,self.patch_radius)
            if self.use_c_map:
                over_c_patch = self._get_local_over_c_patch(ix, iy,self.patch_radius)
                around_c_patch = self._get_local_around_c_patch(ix, iy,self.patch_radius)
            else:
                over_c_patch = self._zero_c_patch
                around_c_patch = self._zero_c_patch

            x_norm = (self.state["x"] ) / (self.domain_limit["x"]) # normalise to [-1,1]
            y_norm = (self.state["y"]) / (self.domain_limit["y"]) # normalise to [-1,1]
            z_norm = 0.0 # as the space is reduced to 2D plane# (old_state["z"] ) / (self.domain_limit["z"]) # normalise to [-1,1]
            #theta_norm = (old_state["theta"]/np.pi) /(2) #normalise to [0,1], with 1 being 1.5pi
            concentration = self.state["concentration"]   # already in [0,1]

            if np.isclose(self.state["theta"],0.0,atol=1e-4) or np.isclose(self.state["theta"],2*np.pi,atol=1e-4):
                theta_encoded = [1,0,0,0]
            elif np.isclose(self.state["theta"],0.5*np.pi,atol=1e-4):
                theta_encoded = [0,1,0,0]
            elif np.isclose(self.state["theta"],1.0*np.pi,atol=1e-4):
                theta_encoded = [0,0,1,0]
            else:
                theta_encoded = [0,0,0,1]

             # the unvisited fields
            free = (self.state["visited_map"][:,:,:,0]).astype(np.float32)
            free= np.transpose(free,(2,0,1))
            if self.use_c_map:
                if concentration >= self.c_threshold:
                    self.state["c_over_threshold_map"][ix, iy, iz, 0] = 1 
                    self.counter_above_threshold += 1
                    current_above_threshold = True
                else: 
                    self.counter_above_threshold = 0
                    current_above_threshold = False
                
                if concentration <= self.c_threshold + self.c_around_width and concentration >= self.c_threshold - self.c_around_width:
                    self.state["c_around_threshold_map"][ix, iy, iz, 0] = 1
                    current_around_threshold = True
                else:
                    current_around_threshold = False
                    
            else:
                self.counter_above_threshold = 0
                current_above_threshold = False
                current_around_threshold = False

           
            if self.use_c_map:
                c_over_threshold = self.state["c_over_threshold_map"][:,:,:,0] # fatching map
                c_over_threshold =np.transpose(c_over_threshold,(2,0,1))
                c_around_threshold = self.state["c_around_threshold_map"][:,:,:,0] # fatching map
                c_around_threshold =np.transpose(c_around_threshold,(2,0,1))
                
            else:
                c_over_threshold = self._zero_c_map
                c_around_threshold = self._zero_c_map

            self.local_GT=self._get_local_GT_patch(ix,iy,self.patch_radius)
            self.local_GT_around=self._get_local_GT_around_patch(ix,iy,self.patch_radius)
            #k_ind = np.argmax(theta_encoded)
            #free = np.rot90(free,k=-int(k_ind), axes=(0,1))
            # -- obs for DQN --
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
                 "visited_maps_downsampled": np.array([*self.downsample_map(free).flatten()]).astype(np.float32),# shape= (batch,Z,X,Y) #!! in this way the CNN does not get an information about where not to move!
                #"c_over_threshold_maps": c_over_threshold.astype(np.float32) ,
                #"c_around_threshold_maps": c_around_threshold.astype(np.float32) ,
                "GT_c_over_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_over_threshold_maps).flatten()]).astype(np.float32),
                "GT_c_around_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_around_threshold_maps).flatten()]).astype(np.float32),
                "local_GT": self.local_GT.flatten(),
                "local_GT_around": self.local_GT_around.flatten(),
                "space_coverage": np.array([np.sum(self.state["visited_map"][:,:,:,0])/self.max_return], dtype=np.float32),
                "plume_coverage": np.array([np.sum(c_over_threshold)/self.maxN_over_thresh if self.maxN_over_thresh>0.0 else 1.0], dtype=np.float32),
                "border_coverage": np.array([np.sum(c_around_threshold)/self.maxN_around_thresh if self.maxN_around_thresh>0.0 else 1.0], dtype=np.float32),
            }

            
            reward = np.float32(-4.0)   # penalty for leaving domain, reducing what could have been collected
            terminated = False # we want the agent to learn
            truncated = False
            info = {"visited_states_count": np.sum(self.state["visited_map"], dtype= np.float32)/self.max_return }
            if self.GP_ON == True: 
                pass
            else:
                if (self.agent_type == "BORDER") or (self.agent_type =="META"):
                    info["current_around_threshold"] = np.sum(c_around_threshold)/self.maxN_around_thresh
                if (self.agent_type == "PLUME") or (self.agent_type =="META"):
                    info["current_above_threshold"] = np.sum(c_over_threshold)/self.maxN_over_thresh
            return obs, reward, terminated, truncated, info


        # marking visited locations depending on actions
        old_ix, old_iy, old_iz = self._pos_to_idx(
            old_x,
            old_y,
            old_z
        )

      
        if not self.train:
            # in eval mode, we want to be able to visualise the path taken
            #self.old_state = old_state.copy()
            self.action = action
        # -- visited map update --
        ix, iy, iz = self._pos_to_idx(
            self.state["x"],
            self.state["y"],
            self.state["z"]
        )
        self.state["concentration"] = self.conc[ix, iy, iz]
        del_x = abs(ix - old_ix)
        del_y = abs(iy - old_iy)
        del_z = abs(iz - old_iz)

        # mark visited BEFORE observation
        if not self.state["visited_map"][ix, iy, iz,0]:
            current_position_IN_visited_locations = False
        else:
            current_position_IN_visited_locations = True
        
        self.state["visited_map"][ix, iy, iz, :] = 1
        # --- 3D expansion:
        # cell above the  cell
        if old_iz+1 < self.map_size_z:
            self.state["visited_map"][old_ix, old_iy, old_iz+1, :] = 1
        # cell above the cell
        if old_iz-1 >=0:
            self.state["visited_map"][old_ix, old_iy, old_iz-1, :] = 1

        if self.use_c_map:
            if self.state["concentration"] >= self.c_threshold:
                self.state["c_over_threshold_map"][ix,iy,iz, :] = 1 
            
                self.counter_above_threshold += 1
                current_above_threshold = True
            else:
                current_above_threshold = False
            if self.state["concentration"] <= self.c_threshold + self.c_around_width and self.state["concentration"] >= self.c_threshold - self.c_around_width:
                self.state["c_around_threshold_map"][ix, iy, iz, 0] = 1
                current_around_threshold = True
            else:
                current_around_threshold = False
            
        else:
            current_above_threshold = False
            current_around_threshold = False
            self.counter_above_threshold = 0

        # check the concentration for the final state to be higher than the old state:
        if self.state["concentration"] >= old_conc:# only rewarding if above a certain threshold
            current_position_IN_higher_concentration = True
        else:
            current_position_IN_higher_concentration = False

        
        fill_position_IN_visited_locations = None
        current_higher_c_than_fill_position = None
        fill_position_IN_higher_concentration = None
        
        
        

        if del_x >0.8 and (np.isclose(old_theta% (2*np.pi), np.pi,atol=1e-4) or np.isclose(old_theta% (2*np.pi), 0.0,atol=1e-4) or np.isclose(old_theta% (2*np.pi), 2*np.pi,atol=1e-4)): # ensuring that the heading is aligned with x, else increment the y tiles
            # mark states next to the agent as visited, if the agent moves in x direction, as the agent occupies space and can be considered to have visited those states as well

            # 1st: right cell of agent within the map -> mark as visited
            if old_iy+1 < self.map_size_y:
                self.state["visited_map"][old_ix, old_iy+1, old_iz, :] = 1
            # --- 3D expansion:
                # cell above the right cell
                if old_iz+1 < self.map_size_z:
                    self.state["visited_map"][old_ix, old_iy+1, old_iz+1, :] = 1
                # cell above the right cell
                if old_iz-1 >=0:
                    self.state["visited_map"][old_ix, old_iy+1, old_iz-1, :] = 1
            # concentration
                if self.use_c_map:
                    if self.conc[old_ix, old_iy+1, old_iz] >= self.c_threshold:
                        self.state["c_over_threshold_map"][old_ix, old_iy+1, old_iz, :] = 1 
                    if self.conc[old_ix, old_iy+1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                        self.state["c_around_threshold_map"][old_ix, old_iy+1, old_iz, :] = 1 

                # 2nd: if agent moves 2 straight -> mark also 2nd on the right as visited
                if del_x >1.8: 
                    if ix<old_ix:
                        self.state["visited_map"][old_ix-1, old_iy+1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd right cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix-1, old_iy+1, old_iz+1, :] = 1
                        # cell above the 2nd right cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix-1, old_iy+1, old_iz-1, :] = 1

                        if self.use_c_map:
                            if self.conc[old_ix-1, old_iy+1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix-1, old_iy+1, old_iz, :] = 1 
                            if self.conc[old_ix-1, old_iy+1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix-1, old_iy+1, old_iz, :] = 1
                    else:
                        self.state["visited_map"][old_ix+1, old_iy+1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd right cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix+1, old_iy+1, old_iz+1, :] = 1
                        # cell above the 2nd right cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix+1, old_iy+1, old_iz-1, :] = 1

                        if self.use_c_map:
                            if self.conc[old_ix+1, old_iy+1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix+1, old_iy+1, old_iz, :] = 1 
                            if self.conc[old_ix+1, old_iy+1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix+1, old_iy+1, old_iz, :] = 1
        
            # 3rd: left cell of agent within the map -> mark as visited
            if old_iy-1 >=0:
                self.state["visited_map"][old_ix, old_iy-1, old_iz, :] = 1
                # --- 3D expansion:
                # cell above the left cell
                if old_iz+1 < self.map_size_z:
                    self.state["visited_map"][old_ix, old_iy-1, old_iz+1, :] = 1
                # cell above the left cell
                if old_iz-1 >=0:
                    self.state["visited_map"][old_ix, old_iy-1, old_iz-1, :] = 1

                if self.use_c_map:
                    if self.conc[old_ix, old_iy-1, old_iz] >= self.c_threshold:
                        self.state["c_over_threshold_map"][old_ix, old_iy-1, old_iz, :] = 1 
                    if self.conc[old_ix, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                        self.state["c_around_threshold_map"][old_ix, old_iy-1, old_iz, :] = 1

                # 4th: if agent moves 2 straight -> mark also 2nd on the left as visited
                if del_x >1.8:
                    if ix<old_ix: 
                        self.state["visited_map"][old_ix-1, old_iy-1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd left cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix-1, old_iy-1, old_iz+1, :] = 1
                        # cell above the 2nd left cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix-1, old_iy-1, old_iz-1, :] = 1

                        if self.use_c_map:
                            if self.conc[old_ix-1, old_iy-1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix-1, old_iy-1, old_iz, :] = 1 
                            if self.conc[old_ix-1, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix-1, old_iy-1, old_iz, :] = 1
                    else:
                        self.state["visited_map"][old_ix+1, old_iy-1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd left cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix+1, old_iy-1, old_iz+1, :] = 1
                        # cell above the 2nd left cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix+1, old_iy-1, old_iz-1, :] = 1
                        if self.use_c_map:
                            if self.conc[old_ix+1, old_iy-1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix+1, old_iy-1, old_iz, :] = 1 
                            if self.conc[old_ix+1, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix+1, old_iy-1, old_iz, :] = 1

            # 5th: filling the states based on movement along x movement (+reward flags)
            if ix<old_ix:
                if not self.state["visited_map"][old_ix-1, old_iy, old_iz,0]:
                    fill_position_IN_visited_locations = False
                    self.state["visited_map"][old_ix-1, old_iy, old_iz, :] = 1
                    # --- 3D expansion:
                    # cell above the 2nd cell
                    if old_iz+1 < self.map_size_z:
                        self.state["visited_map"][old_ix-1, old_iy, old_iz+1, :] = 1
                    # cell above the 2nd cell
                    if old_iz-1 >=0:
                        self.state["visited_map"][old_ix-1, old_iy, old_iz-1, :] = 1
                    if self.use_c_map:
                        if self.conc[old_ix-1, old_iy, old_iz]>= self.c_threshold:
                            self.state["c_over_threshold_map"][old_ix-1, old_iy, old_iz, :] = 1 
                        if self.conc[old_ix-1, old_iy, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy, old_iz] >= self.c_threshold - self.c_around_width:
                            self.state["c_around_threshold_map"][old_ix-1, old_iy, old_iz, :] = 1
                else:
                    fill_position_IN_visited_locations = True
                
                c_xm1_y_z = self.conc[old_ix-1,old_iy,old_iz]
                # loading the concentration of the transient cell: self.get_concentration(...)
                if c_xm1_y_z>=old_conc:
                    fill_position_IN_higher_concentration = True
                else:
                    fill_position_IN_higher_concentration = False
                
                if c_xm1_y_z>=self.c_threshold: # substituting for only rewarding above a certain threshold
                    fill_above_threshold = True
                else:
                    fill_above_threshold = False
                if self.agent_type =="BORDER":
                        if c_xm1_y_z<= self.c_around_width + self.c_threshold and c_xm1_y_z>= self.c_threshold - self.c_around_width:
                            fill_around_threshold = True
                        else:
                            fill_around_threshold = False
                
                if self.state["concentration"] >= c_xm1_y_z:
                    current_higher_c_than_fill_position = True
                else:
                    current_higher_c_than_fill_position = False
            else:
                if not self.state["visited_map"][old_ix+1, old_iy, old_iz,0]:
                    fill_position_IN_visited_locations = False
                    self.state["visited_map"][old_ix+1, old_iy, old_iz, :] = 1
                    # --- 3D expansion:
                    # cell above the 2nd left cell
                    if old_iz+1 < self.map_size_z:
                        self.state["visited_map"][old_ix+1, old_iy, old_iz+1, :] = 1
                    # cell above the 2nd left cell
                    if old_iz-1 >=0:
                        self.state["visited_map"][old_ix+1, old_iy, old_iz-1, :] = 1
                    if self.use_c_map:
                        if self.conc[old_ix+1, old_iy, old_iz] >= self.c_threshold:
                            self.state["c_over_threshold_map"][old_ix+1, old_iy, old_iz, :] = 1 
                        if self.conc[old_ix+1, old_iy, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy, old_iz] >= self.c_threshold - self.c_around_width:
                            self.state["c_around_threshold_map"][old_ix+1, old_iy, old_iz, :] = 1
                else:
                    fill_position_IN_visited_locations = True
                
                c_xp1_y_z = self.conc[old_ix+1,old_iy,old_iz]
                # loading the concentration of the transient cell: self.get_concentration(...)
                if c_xp1_y_z>=old_conc:
                    fill_position_IN_higher_concentration = True
                else:
                    fill_position_IN_higher_concentration = False
                
                if c_xp1_y_z>=self.c_threshold: # substituting for only rewarding above a certain threshold
                    
                    fill_above_threshold = True
                else:
                    fill_above_threshold = False
                if self.agent_type =="BORDER":
                        if c_xp1_y_z<= self.c_around_width + self.c_threshold and c_xp1_y_z>= self.c_threshold - self.c_around_width:
                            fill_around_threshold = True
                        else:
                            fill_around_threshold = False
                
                if self.state["concentration"] >= c_xp1_y_z:
                    current_higher_c_than_fill_position = True
                else:
                    current_higher_c_than_fill_position = False

        elif del_y >0.8:

            # mark states next to the agent as visited, if the agent moves in x direction, as the agent occupies space and can be considered to have visited those states as well
            if old_ix+1 < self.map_size_x:
                self.state["visited_map"][old_ix+1, old_iy, old_iz, :] = 1
                # --- 3D expansion:
                # cell above the left cell
                if old_iz+1 < self.map_size_z:
                    self.state["visited_map"][old_ix+1, old_iy, old_iz+1, :] = 1
                # cell above the left cell
                if old_iz-1 >=0:
                    self.state["visited_map"][old_ix+1, old_iy, old_iz-1, :] = 1
                if self.use_c_map:
                    if self.conc[old_ix+1, old_iy, old_iz] >= self.c_threshold:
                        self.state["c_over_threshold_map"][old_ix+1, old_iy, old_iz, :] = 1 
                    if self.conc[old_ix+1, old_iy, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy, old_iz] >= self.c_threshold - self.c_around_width:
                        self.state["c_around_threshold_map"][old_ix+1, old_iy, old_iz, :] = 1
                if del_y >1.8:
                    if iy<old_iy:
                        self.state["visited_map"][old_ix+1, old_iy-1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd left cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix+1, old_iy-1, old_iz+1, :] = 1
                        # cell above the 2nd left cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix+1, old_iy-1, old_iz-1, :] = 1
                        if self.use_c_map:
                            if self.conc[old_ix+1, old_iy-1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix+1, old_iy-1, old_iz, :] = 1 
                            if self.conc[old_ix+1, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix+1, old_iy-1, old_iz, :] = 1
                    else:
                        self.state["visited_map"][old_ix+1, old_iy+1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the 2nd left cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix+1, old_iy+1, old_iz+1, :] = 1
                        # cell above the 2nd left cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix+1, old_iy+1, old_iz-1, :] = 1
                        if self.use_c_map:
                            if self.conc[old_ix+1, old_iy+1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix+1, old_iy+1, old_iz, :] = 1 
                            if self.conc[old_ix+1, old_iy+1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix+1, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix+1, old_iy+1, old_iz, :] = 1
            if old_ix-1 >=0:
                self.state["visited_map"][old_ix-1, old_iy, old_iz, :] = 1
                # --- 3D expansion:
                # cell above the right cell
                if old_iz+1 < self.map_size_z:
                    self.state["visited_map"][old_ix-1, old_iy, old_iz+1, :] = 1
                # cell above the right cell
                if old_iz-1 >=0:
                    self.state["visited_map"][old_ix-1, old_iy, old_iz-1, :] = 1
                if self.use_c_map:
                    if self.conc[old_ix-1, old_iy, old_iz] >= self.c_threshold:
                        self.state["c_over_threshold_map"][old_ix-1, old_iy, old_iz, :] = 1 
                    if self.conc[old_ix-1, old_iy, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy, old_iz] >= self.c_threshold - self.c_around_width:
                            self.state["c_around_threshold_map"][old_ix-1, old_iy, old_iz, :] = 1
            
                if del_y >1.8:
                    if iy<old_iy:
                        self.state["visited_map"][old_ix-1, old_iy-1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the right cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix-1, old_iy-1, old_iz+1, :] = 1
                        # cell above the right cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix-1, old_iy-1, old_iz-1, :] = 1
                        if self.use_c_map:
                            if self.conc[old_ix-1, old_iy-1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix-1, old_iy-1, old_iz, :] = 1 
                            if self.conc[old_ix-1, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix-1, old_iy-1, old_iz, :] = 1
                    else:
                        self.state["visited_map"][old_ix-1, old_iy+1, old_iz, :] = 1
                        # --- 3D expansion:
                        # cell above the right cell
                        if old_iz+1 < self.map_size_z:
                            self.state["visited_map"][old_ix-1, old_iy+1, old_iz+1, :] = 1
                        # cell above the right cell
                        if old_iz-1 >=0:
                            self.state["visited_map"][old_ix-1, old_iy+1, old_iz-1, :] = 1
                        if self.use_c_map:
                            if self.conc[old_ix-1, old_iy+1, old_iz] >= self.c_threshold:
                                self.state["c_over_threshold_map"][old_ix-1, old_iy+1, old_iz, :] = 1
                            if self.conc[old_ix-1, old_iy+1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix-1, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                                self.state["c_around_threshold_map"][old_ix-1, old_iy+1, old_iz, :] = 1

            #marking the states based on movement 
            if iy<old_iy:
                if not self.state["visited_map"][old_ix, old_iy-1, old_iz,0]:
                    fill_position_IN_visited_locations = False
                    self.state["visited_map"][old_ix, old_iy-1, old_iz, :] = 1
                    # --- 3D expansion:
                    # cell above the right cell
                    if old_iz+1 < self.map_size_z:
                        self.state["visited_map"][old_ix, old_iy-1, old_iz+1, :] = 1
                    # cell above the right cell
                    if old_iz-1 >=0:
                        self.state["visited_map"][old_ix, old_iy-1, old_iz-1, :] = 1
                    if self.use_c_map:
                        if self.conc[old_ix, old_iy-1, old_iz] >= self.c_threshold:
                            self.state["c_over_threshold_map"][old_ix, old_iy-1, old_iz, :] = 1 
                        if self.conc[old_ix, old_iy-1, old_iz]<= self.c_threshold +self.c_around_width and self.conc[old_ix, old_iy-1, old_iz] >= self.c_threshold - self.c_around_width:
                            self.state["c_around_threshold_map"][old_ix, old_iy-1, old_iz, :] = 1
                else:
                    fill_position_IN_visited_locations = True

                c_x_ym1_z= self.conc[old_ix,old_iy-1,old_iz]
                # loading the concentration of the transient cell: self.get_concentration(...)
                if c_x_ym1_z>=old_conc:
                    fill_position_IN_higher_concentration = True
                else:
                    fill_position_IN_higher_concentration = False
                
                if c_x_ym1_z>=self.c_threshold: # substituting for only rewarding above a certain threshold
                    fill_above_threshold = True
                else:
                    fill_above_threshold = False
                if self.agent_type =="BORDER":
                        if c_x_ym1_z<= self.c_around_width + self.c_threshold and c_x_ym1_z>= self.c_threshold - self.c_around_width :
                            fill_around_threshold = True
                        else:
                            fill_around_threshold = False
                
                if self.state["concentration"] >= c_x_ym1_z:
                    current_higher_c_than_fill_position = True
                else:
                    current_higher_c_than_fill_position = False
            else:
                if not self.state["visited_map"][old_ix, old_iy+1, old_iz,0]:
                    fill_position_IN_visited_locations = False
                    self.state["visited_map"][old_ix, old_iy+1, old_iz, :] = 1
                    # --- 3D expansion:
                    # cell above the middle cell
                    if old_iz+1 < self.map_size_z:
                        self.state["visited_map"][old_ix, old_iy+1, old_iz+1, :] = 1
                    # cell above the middle cell
                    if old_iz-1 >=0:
                        self.state["visited_map"][old_ix, old_iy+1, old_iz-1, :] = 1
                    if self.use_c_map:
                        if self.conc[old_ix, old_iy+1, old_iz] >= self.c_threshold:
                            self.state["c_over_threshold_map"][old_ix, old_iy+1, old_iz, :] = 1 
                        if self.conc[old_ix, old_iy+1, old_iz]<= (self.c_threshold +self.c_around_width) and self.conc[old_ix, old_iy+1, old_iz] >= self.c_threshold - self.c_around_width:
                            self.state["c_around_threshold_map"][old_ix, old_iy+1, old_iz, :] = 1

                else:
                    fill_position_IN_visited_locations = True
                
                c_x_yp1_z = self.conc[old_ix,old_iy+1,old_iz]
                # loading the concentration of the transient cell: self.get_concentration(...)
                if c_x_yp1_z>=old_conc:
                    fill_position_IN_higher_concentration = True
                else:
                    fill_position_IN_higher_concentration = False
                
                if c_x_yp1_z>=self.c_threshold: # substituting for only rewarding above a certain threshold
                    fill_above_threshold = True
                else:
                    fill_above_threshold = False
                if self.agent_type =="BORDER":
                        if c_x_yp1_z<= self.c_around_width + self.c_threshold and c_x_yp1_z>= self.c_threshold - self.c_around_width:
                            fill_around_threshold = True
                        else:
                            fill_around_threshold = False

                if self.state["concentration"] >= c_x_yp1_z:
                    current_higher_c_than_fill_position = True
                else:
                    current_higher_c_than_fill_position = False

        #print(self.state["visited_map"][:,:,0,0])
        
        #print(f"del_x={del_x}, del_y={del_y}, old_state[\"theta\"]% (2*np.pi)={old_state['theta']% (2*np.pi)==np.pi}")
        # if del_z >0.8:
        #     if iz<old_iz:
        #         if not self.state["visited_map"][old_ix, old_iy, old_iz-1,0]:
        #             fill_position_IN_visited_locations = False
        #             self.state["visited_map"][old_ix, old_iy, old_iz-1, :] = 1
        #         else:
        #             fill_position_IN_visited_locations = True
                    
        #     else:
        #         if not self.state["visited_map"][old_ix, old_iy, old_iz+1,0]:
        #             fill_position_IN_visited_locations = False
        #             self.state["visited_map"][old_ix, old_iy, old_iz+1, :] = 1
        #         else:
        #             fill_position_IN_visited_locations = True
                    
    
        visited_patch = self._get_local_visited_patch(ix, iy,self.patch_radius)
        if self.use_c_map:
            over_c_patch = self._get_local_over_c_patch(ix, iy,self.patch_radius)
            around_c_patch = self._get_local_around_c_patch(ix,iy,self.patch_radius)
        else:
            over_c_patch = self._zero_c_patch
            around_c_patch = self._zero_c_patch

        x_norm = (self.state["x"] ) / (self.domain_limit["x"]) # normalise to [-1,1]
        y_norm = (self.state["y"] ) / (self.domain_limit["y"]) # normalise to [-1,1]
        z_norm = 0.0 # as the space is reduced to 2D plane# (self.state["z"] ) / (self.domain_limit["z"]) # normalise to [-1,1]

        

        

        #theta_norm = (self.state["theta"]/np.pi) /(2) #normalise to [0,1], with 1 being 1.5pi
        concentration = self.state["concentration"]  # already in [0,1]

        # the unvisited fields
        free = (self.state["visited_map"][:,:,:,0])
        free= np.transpose(free,(2,0,1))
        if self.use_c_map:
                c_over_threshold = self.state["c_over_threshold_map"][:,:,:,0] # fatching map
                c_over_threshold =np.transpose(c_over_threshold,(2,0,1))
                c_around_threshold = self.state["c_around_threshold_map"][:,:,:,0] # fatching map
                c_around_threshold =np.transpose(c_around_threshold,(2,0,1))
                
        else:
            c_over_threshold = self._zero_c_map
            c_around_threshold = self._zero_c_map

        
        #print(visited_patch)
        #print(np.sum(np.ones(free[:,:,0].shape))-np.sum(free.astype(np.float32)[:,:,0]))
        #print(free[:,:,0].astype(np.float32))
        if np.isclose(self.state["theta"],0.0,atol=1e-4) or np.isclose(self.state["theta"],2*np.pi,atol=1e-4):
            theta_encoded = [1,0,0,0]
        elif np.isclose(self.state["theta"],0.5*np.pi,atol=1e-4):
            theta_encoded = [0,1,0,0]
        elif np.isclose(self.state["theta"],1.0*np.pi,atol=1e-4):
            theta_encoded = [0,0,1,0]
        else:
            theta_encoded = [0,0,0,1]

        #k_ind = np.argmax(theta_encoded)
        #free = np.rot90(free,k=-int(k_ind), axes=(0,1))

        self.local_GT=self._get_local_GT_patch(ix,iy,self.patch_radius)
        self.local_GT_around=self._get_local_GT_around_patch(ix,iy,self.patch_radius)
        #print(visited_patch)
        # -- obs for DQN --
        
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
            "visited_maps_downsampled": np.array([*self.downsample_map(free).flatten()]).astype(np.float32),# shape= (batch,Z,X,Y) #!! in this way the CNN does not get an information about where not to move!
            #"c_over_threshold_maps": c_over_threshold.astype(np.float32) ,
            #"c_around_threshold_maps": c_around_threshold.astype(np.float32) ,
            "GT_c_over_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_over_threshold_maps).flatten()]).astype(np.float32),
            "GT_c_around_threshold_maps_downsampled": np.array([*self.downsample_map(self.GT_c_around_threshold_maps).flatten()]).astype(np.float32),
            "local_GT": self.local_GT.flatten(),
            "local_GT_around": self.local_GT_around.flatten(),
            "space_coverage": np.array([np.sum(self.state["visited_map"][:,:,:,0])/self.max_return], dtype=np.float32),
            "plume_coverage": np.array([np.sum(c_over_threshold)/self.maxN_over_thresh if self.maxN_over_thresh>0.0 else 1.0], dtype=np.float32),
            "border_coverage": np.array([np.sum(c_around_threshold)/self.maxN_around_thresh if self.maxN_around_thresh>0.0 else 1.0], dtype=np.float32),
        }
        
        
        #print(visited_patch)
        
        #print(f"c_over_thresh{obs["c_over_threshold_maps"][0]}")
        # -- debugging checks for over_c_threshold construction --

        #print(obs["c_over_threshold_maps"][0]) # obs space (filling up the 1s)

            #- the reference check -
        # x = np.arange(-self.map_size_x//2, self.map_size_x//2)[:, None, None]
        # y = np.arange(-self.map_size_y//2, self.map_size_y//2)[None, :, None]
        # z = np.arange(-self.map_size_z//2, self.map_size_z//2)[None, None, :]
        # #otherwise we get cartesian confusion
        # print(np.transpose(np.where(self.get_concentration(x,y,z)>=self.c_threshold,1,0),(2,0,1)).astype(np.float32))

        # -- debugging checks for visited map update and concentration comparison logic --
    
        if fill_position_IN_visited_locations is None:
            print(f"old_state['theta']% (2*np.pi)={old_theta% (2*np.pi)}, del_x={del_x}, del_y={del_y}, del_z={del_z}")
            print(f"old_ix={old_ix}, old_iy={old_iy}, old_iz={old_iz}")
            print(f"ix={ix}, iy={iy}, iz={iz}")
            print(self.state["visited_map"][:,:,0,0])
            #print(visited_patch)
            #raise ValueError("Error in visited map update logic.")

        if fill_position_IN_higher_concentration is None:
            print(f"old_state['theta']% (2*np.pi)={old_theta% (2*np.pi)}, del_x={del_x}, del_y={del_y}, del_z={del_z}")
            print(f"old_ix={old_ix}, old_iy={old_iy}, old_iz={old_iz}")
            print(f"ix={ix}, iy={iy}, iz={iz}")
            print(f"old_state['concentration']={old_conc}, self.state['concentration']={self.state['concentration']}")
            #raise ValueError("Error in concentration comparison logic.")

            
        

        



        terminated = False
        truncated = False

        
        if self.HRL == True or self.GP_ON: 
            # waiting for the GP update to calculate the reward and other params
            reward = 0.0
            terminated = False
            info = {}
            info["visited_states_count"] = np.sum(self.state["visited_map"])/self.max_return  
        else:
            # agent: SPACE

            if self.agent_type == "SPACE" :
            # distance_to_closest_unvisited_cell = self.calculate_distance_to_closest_unvisited(np.array([self.state["x"],self.state["y"],self.state["z"]]),self.state["visited_map"][:,:,:,0])
                #print(f"DISTANCE = {distance_to_closest_unvisited_cell}")
                
                reward = self.reward_fn_1explore.get_reward( self.agent_turns, obs["space_coverage"][0],np.sum(self.state["visited_map"][:,:,:,0])-self.old_visited_sum) # pass history to reward function

                if obs["space_coverage"][0]>=self.accuracy_agent_goals[0]: 
                    reward = 40.0 # bonus for visiting 90% of the environment, to encourage exploration, otherwise the agent might get stuck in a local area with high concentration
                    terminated = True
            
            # agent: PLUME

            if self.agent_type == "PLUME":

                #mask = self.GT_c_over_threshold_maps & free # structure: z,x,y ; GT:true=above the range, FREE: true = unvisited -> False values will be accounted in the distance function

                #distance_to_closest_concentration_cell = self.calculate_distance_to_closest_unvisited(np.array([self.state["z"],self.state["x"],self.state["y"]]),(~mask)) # irrelevant
                
                diff_new_cells, found_source = None, None

                reward = self.reward_fn_2plume.get_reward( self.agent_turns,obs["plume_coverage"][0],np.sum(self.state["c_over_threshold_map"])-self.old_c_over_sum,SUBAGENT_TRAIN_ON_GP=self.GP_ON,diff_new_cells=diff_new_cells,found_source=found_source) # pass history to reward function
                
                    # we dropped the consecutive termination condition
                # print(f"STEP_HUGIN; PLUME COV:{obs["plume_coverage"][0] } AND{self.c_plume_coverage_was_already_below}")
                if obs["plume_coverage"][0] >= self.accuracy_agent_goals[1] and self.c_plume_coverage_was_already_below: # if the agent is above the threshold for 5 consecutive steps, it has likely found the source, and we can terminate the episode
                    reward = 40.0 # bonus for finding the source, as the agent might get stuck in local areas with higher concentration but not the source
                    terminated = True
                    #print("WTF PLUME")
                elif obs["plume_coverage"][0] < self.accuracy_agent_goals[1] :
                    self.c_plume_coverage_was_already_below = True

            # agent: BORDER

            if self.agent_type == "BORDER":
                
                #mask = self.GT_c_around_threshold_maps & free # structure: z,x,y ; GT:true=within the range, FREE: true = unvisited -> False values will be accounted in the distance function
                #distance_to_closest_unvisited_cell = self.calculate_distance_to_closest_unvisited(np.array([self.state["z"],self.state["x"],self.state["y"]]),(~mask))
                # CHANGE
                diff_new_cells, found_source = None, None
                #print(f"DISTANCE = {distance_to_closest_unvisited_cell}")
                reward = self.reward_fn_3border.get_reward(self.agent_turns,obs["border_coverage"][0],np.sum(self.state["c_around_threshold_map"])-self.old_c_around_sum,SUBAGENT_TRAIN_ON_GP=self.GP_ON,diff_new_cells=diff_new_cells,found_source=found_source) # pass history to reward function

                if obs["border_coverage"][0] >= self.accuracy_agent_goals[2] and self.c_border_coverage_was_already_below: # if the agent is above the threshold for 5 consecutive steps, it has likely found the source, and we can terminate the episode
                    reward = 40.0 # bonus for finding the source, as the agent might get stuck in local areas with higher concentration but not the source
                    terminated = True# Agent plume boarder!
                    #print("WTF BORDER")
                elif obs["border_coverage"][0] < self.accuracy_agent_goals[2]:# and self.first_step == False:
                    self.c_border_coverage_was_already_below = True
            
            # agent: META
            
            if self.agent_type == "META":
                if(np.sum(self.state["visited_map"])>=0.7*self.max_return): #
                    reward = 40.0 # bonus for fulfilling all 3 objectives
                    terminated = True# Agent plume boarder!
                elif ((np.sum(self.state["c_around_threshold_map"]) >= 0.8*self.maxN_around_thresh) and (np.sum(c_over_threshold) >= 0.8*self.maxN_over_thresh)) and self.maxN_around_thresh>0.0 and self.maxN_over_thresh>0.0:
                    reward = 40.0 # bonus for fulfilling plume objectives
                    terminated = False # we want to explore at least 80% of the rest
                else:
                    reward = 0 # placeholder

        
            info = {}
            info["visited_states_count"] = np.sum(self.state["visited_map"])/self.max_return   
            if self.GP_ON == False and ((self.agent_type == "BORDER") or (self.agent_type =="META")):
                info["current_around_threshold"] = np.sum(c_around_threshold)/self.maxN_around_thresh
            if self.GP_ON == False and ((self.agent_type == "PLUME") or (self.agent_type =="META")):
                info["current_above_threshold"] = np.sum(c_over_threshold)/self.maxN_over_thresh

        #self.first_step = False
        return obs, reward, terminated, truncated, info

    def render(self):
        self.renderer.render(self.model_path)
        # self.renderer.plot_concentration_field(self.concentration_function ,self.gaussian_centers[0], self.gaussian_sigma)
        self.renderer.plot_concentration_field2(self.x,self.y,self.z,self.conc,self.c_threshold,self.c_around_width)

    def step_sim(self):
        self.renderer.step_sim(self.state)#,self.old_state,self.action) # forward also local patch of visited map

        """
        define here how to plot density field and density estimation
        """
        