import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gpytorch
import numpy as np

from gpt_class_exactgpmodel import ExactGPModel
import path
import chem_utils

import numpy as np
import gym
from stable_baselines3 import PPO

# %%
# Definitions

# %%
class GasSurveyEnv(gym.Env):
    def __init__(self, scenario_bank, gp_ls_constraint=gpytorch.constraints.Interval(9, 11), gp_kernel_type='scale_rbf', gp_pred_resolution=None):
        super().__init__()

        self.scenario_bank = scenario_bank
        self.ls_const = gp_ls_constraint
        self.kernel_type = gp_kernel_type

        self.gp_pred_resolution = gp_pred_resolution

        self.reset()
        
        # μ, σ, visited, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of 'visited' include location channels
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        self.channels = np.array([0, 1, 0, 1, 1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.float32
        )

    def reset(self):
        # Draw a random scenario/snapshot
        random_env = self.scenario_bank.sample()
        self.env_xy = random_env['coords']
        self.values = random_env['values']
        self.parameter = random_env['parameter']
        self.depth = random_env['depth']
        self.time = random_env['time']
        
         # Action space is a float anywhere inside this box:
        self.action_space = gym.spaces.Box(low=np.array([0., 0.]),
                                high=np.array([self.env_xy[:, 0].max(), self.env_xy[:, 1].max()]),
                                dtype=np.float32)

        # Init GP model
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        
        # Init sample memory and location
        self.sampled_coords = torch.tensor([])
        self.sampled_vals = torch.tensor([])
        self.location = torch.tensor((0, 0, 0))

        # Init prediction tensors
        self.mu_flat         = torch.tensor([])
        self.sigma_flat      = torch.tensor([])
        
        # Init observation channels
        self._create_obs_coords()

        self.mu_norm    = np.zeros((self.obs_y, self.obs_x), dtype=np.float32)
        self.sigma_norm = np.zeros_like(self.mu_norm)
        self.visited    = np.zeros_like(self.mu_norm)
        
        obs = self._render_layers()
        
        return obs, {}

    def step(self, action, speed, sample_freq):
        # action = absolute (x,y) or Δx,Δy; clip, update GP, rewards...
        start_time = '2020-01-01T02:10:00.000000000' # dummy time
        synoptic = True
        old_loc = self.location
        new_loc = action

        sample_coords = path.path([old_loc, new_loc], start_time, speed, sample_freq, synoptic)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = torch.tensor([(float(t[0]), float(t[1])) for t in sample_coords])

        measurements = torch.zeros(len(sample_coords_xy), dtype=torch.float32)
        radius = 1.0 # Radius of sample averaging
        for c, coord in enumerate(sample_coords_xy):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.x, self.y, self.val, coord.numpy(), radius)
        
        self.sampled_coords = torch.cat((self.sampled_coords, sample_coords_xy))
        self.sampled_vals = torch.cat((self.sampled_vals, measurements))

        self.estimate() # fill self.mu and self.sigma
        
        

        obs  = self._render_layers()
        done = ...
        info = {}
        return obs, reward, done, False, info
    
    def estimate(self):
        if self.mdl is None:
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        
        self.mdl.set_train_data(self.sampled_coords, self.sampled_vals, strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(self._coords_flat))
    
        self.mu = current_pred.mean
        self.sigma = current_pred.variance

        return

    def _render_layers(self) -> np.ndarray:
        """
        Assemble the observation tensor.

        Channels (fixed order):
            0: μ‑field  (self.mu_norm)
            1: σ‑field  (self.sigma_norm)
            2: visited  mask (self.visited)
            3: Coord‑Y  channel (self.coord_y)
            4: Coord‑X  channel (self.coord_x)

        Only the layers whose corresponding entry in `self.channels`
        is truthy (1 / True) are stacked.
        """
        # List all *possible* layers in a canonical order
        candidate_layers = [
            self.mu_norm,     # idx 0
            self.sigma_norm,  # idx 1
            self.visited,     # idx 2
            self.coord_y_norm,     # idx 3
            self.coord_x_norm      # idx 4
        ]

        # Select the ones flagged by `self.channels`
        chosen_layers = [
            layer for layer, flag in zip(candidate_layers, self.channels) if flag
        ]

        # Sanity‑check: number of layers matches observation_space
        assert len(chosen_layers) == self.channels.sum(), \
            "Mismatch between channel mask and selected layers"

        # Stack into (C, H, W) NumPy array expected by Gym
        stacked = np.stack(chosen_layers, axis=0).astype(np.float32)
        return stacked
    
    def _create_obs_coords(self):
        
        if self.gp_pred_resolution:
            # Downsampling based on given pred_resolution

            self.obs_x, self.obs_y = self.gp_pred_resolution
            # -- 1. grid of query points -----------------
            #   (H*W, 2) tensor that GPyTorch will accept.
            xs = np.linspace(0, self.action_space.high[0], self.obs_x, dtype=np.float32)
            ys = np.linspace(0, self.action_space.high[1], self.obs_y, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)                        # shape (H, W)

            # Save as 2‑D field for coord‑channels and as flat list for GP queries
            self._coord_x = gx           # (H, W)
            self._coord_y = gy           # (H, W)
            self._coords = np.stack([gx, gy], axis=-1)      # (H, W, 2)
            self._coords_flat = torch.from_numpy(
                    self._coords.reshape(-1, 2)             # (H*W, 2)
            )
            
            # -- 2. static coordinate channels, normalised (0‒1) --
            self.coord_x_norm = (gx / xs.max()).astype(np.float32)   # (H, W)
            self.coord_y_norm = (gy / ys.max()).astype(np.float32)   # (H, W)

            return
        
        if self.gp_pred_resolution is None:
            # No downsampling, predict env_xy coords from environment

            self.obs_x = len(self.env_xy[:, 0])
            self.obs_y = len(self.env_xy[:, 1])
            
            # -- 1. grid of query points ------------------
            self._coords_flat = self.env_xy

            if isinstance(self._coords_flat, torch.Tensor):
                coords = self._coords_flat.view(self.obs_y, self.obs_x, 2) # torch view/reshape
                self._coords = coords.cpu().numpy() # if you still need NumPy
            else: # already NumPy array
                self._coords = self._coords_flat.reshape(self.obs_y, self.obs_x, 2)

            # -- 2. split into per‑axis images ---------------------------------------
            self._coord_x = self._coords[:, 0]   # shape (H, W)
            self._coord_y = self._coords[:, 1]   # shape (H, W)

            # -- 3. normalised static coordinate channels
            self.coord_x_norm = (self._coord_x / self._coord_x.max()).astype(np.float32)
            self.coord_y_norm = (self._coord_y / self._coord_y.max()).astype(np.float32)
        
            return
    
    def _tensor_to_obs_channel(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.view(self.obs_y, self.obs_x, 2).cpu().numpy()
        
        return tensor.reshape(self.obs_y, self.obs_x, 2)
    
