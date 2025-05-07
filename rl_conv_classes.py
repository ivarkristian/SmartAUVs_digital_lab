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
        self.min_concentration, self.max_concentration = self.scenario_bank.get_minmax()
        self.mu_all, self.sigma2_all = self.scenario_bank.get_mu_sigma2()
        self.ls_const = gp_ls_constraint
        self.kernel_type = gp_kernel_type

        self.gp_pred_resolution = gp_pred_resolution

        self.n_episodes = 0
        # reset draws a random scenario, initializes GP model and sample memory
        self.reset()
        
        # μ, σ, visited, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of 'visited' include location channels
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        self.channels = np.array([0, 1, 1, 0, 0])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.float32
        )

         # Action space is a float anywhere inside this box:
        self.action_space = gym.spaces.Box(low=np.array([self.env_y[:, 0].min(), self.env_xy[:, 1].min()]),
                                high=np.array([self.env_x_max, self.env_y_max]),
                                dtype=np.float32)

    def reset(self):
        # Draw a random scenario/snapshot
        random_env = self.scenario_bank.sample()
        self.env_xy = random_env['coords']
        self.values = random_env['values']
        self.parameter = random_env['parameter']
        self.depth = random_env['depth']
        self.time = random_env['time']

        self.env_x_max = self.env_xy[:, 0].max()
        self.env_y_max = self.env_xy[:, 1].max()

        self.n_steps = 0
        self.n_episodes += 1

        # Init observation channels
        self._create_obs_coords()

        self.pred_mu_norm    = np.zeros((self.obs_y, self.obs_x), dtype=np.float32)
        self.pred_var_norm = np.zeros_like(self.mu_pred_norm)
        self.visited    = np.zeros_like(self.mu_pred_norm)

        # Init GP model
        self.ls_const_norm = self.ls_const/self.env_x_max
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_type, lengthscale_constraint=self.ls_const_norm)
        self.mdl.covar_module.outputscale = torch.tensor(1.0)   # fixed prior variance 1
        self.variance_prior = self.mdl.get_outputscale()   # = 1.0

        # Init normalized coords and values for GP model
        # coords_flat_norm is [0, 1]
        self.coords_flat_norm = np.stack([self.coord_x_norm, self.coord_y_norm], axis=-1)
        # values
        self.values_norm_minmax = self._norm_minmax()
        self.values_norm_zscale = self._norm_zscale()
        
        # Init sample memory. Could include lawnmower path samples.
        self.sampled_coords = torch.tensor([])
        self.sampled_vals = torch.tensor([])
        
        # Init location. Should be random
        self.location = torch.tensor((0, 0, 0))

        # Init prediction tensors
        self.pred_mu = torch.tensor([])
        self.pred_var = torch.tensor([])
        
        obs = self._render_layers()
        info = {}
        
        return obs, info

    def step(self, action, speed, sample_freq):
        self.n_steps += 1
        # action = absolute (x,y) or Δx,Δy; clip, update GP, rewards...
        start_time = '2020-01-01T02:10:00.000000000' # dummy time
        synoptic = True
        old_loc = self.location
        old_var = self.pred_var_norm

        # expects action to be a pair of locs within [0, 255]
        new_loc = action

        sample_coords = path.path([old_loc, new_loc], start_time, speed, sample_freq, synoptic)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = torch.tensor([(float(t[0]), float(t[1])) for t in sample_coords])
        sample_coords_norm = sample_coords_xy / torch.tensor([self.env_x_max, self.env_y_max])

        measurements = torch.zeros(len(sample_coords_xy), dtype=torch.float32)
        radius = 1.0/self.env_x_max # Radius of sample averaging
        for c, coord in enumerate(sample_coords_norm):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.coord_x_norm, self.coord_y_norm, self.values_norm_zscale, coord.numpy(), radius)
        
        self.sampled_coords = torch.cat((self.sampled_coords, sample_coords_xy))
        self.sampled_coords_norm = torch.cat((self.sampled_coords, sample_coords_norm))
        self.sampled_vals = torch.cat((self.sampled_vals, measurements))

        self.estimate() # fill self.pred_mu self.pred_var and normalized equivalents
        
        # compute reward (based on decrease in overall variance)
        reward = (old_var - self.pred_var_norm).sum()

        obs  = self._render_layers()
        done = (self.n_steps >= 25)
        info = {}

        return obs, reward, done, info
    
    def estimate(self):
        if self.mdl is None:
            self.mdl = ExactGPModel(self.sampled_coords_norm, self.sampled_vals, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        
        self.mdl.set_train_data(self.sampled_coords_norm, self.sampled_vals, strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(self.coords_flat_norm))
    
        self.pred_mu = current_pred.mean
        self.pred_var = current_pred.variance/self.variance_prior

        pred_mu_dezscale = self.pred_mu * self.sigma2_all + self.mu_all
        pred_mu_norm = (pred_mu_dezscale - self.values.min()) / (self.values.max() - self.values.min() + 1e-8)
        
        self.pred_mu_norm = np.clip(self._tensor_to_obs_channel(pred_mu_norm), 0.0, 1.0)
        self.pred_var_norm = np.clip(self._tensor_to_obs_channel(self.pred_var), 0.0, 1.0)

        return

    def _norm_minmax(self):
        return (self.values - self.min_concentration)/(self.max_concentration - self.min_concentration)
    
    def _norm_zscale(self):
        return (self.values - self.mu_all)/self.sigma2_all

    def _render_layers(self) -> np.ndarray:
        """
        Assemble the observation tensor.

        Channels (fixed order):
            0: μ‑field  (self.mu_norm)
            1: σ‑field  (self.sigma_norm)
            2: location  mask (self.location) (could be all visited locations)
            3: Coord‑Y  channel (self.coord_y)
            4: Coord‑X  channel (self.coord_x)

        Only the layers whose corresponding entry in `self.channels`
        is truthy (1 / True) are stacked.
        """
        # List all *possible* layers in a canonical order
        candidate_layers = [
            self.pred_mu_norm,     # idx 0
            self.pred_var_norm,  # idx 1
            self.location,     # idx 2
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
            xs = np.linspace(0, self.env_x_max, self.obs_x, dtype=np.float32)
            ys = np.linspace(0, self.env_y_max, self.obs_y, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)                        # shape (H, W)

            # Save as 2‑D field for coord‑channels and as flat list for GP queries
            self._coord_x = gx           # (H, W)
            self._coord_y = gy           # (H, W)
            self._coords = np.stack([gx, gy], axis=-1)      # (H, W, 2)
            self._coords_flat = torch.from_numpy(
                    self._coords.reshape(-1, 2)             # (H*W, 2)
            )
            
            # -- 2. static coordinate channels, normalised [0, 1] --
            self.coord_x_norm = (gx / xs.max()).astype(np.float32)  # (H, W)
            self.coord_y_norm = (gy / ys.max()).astype(np.float32)  # (H, W)
        
        if self.gp_pred_resolution is None:
            # No downsampling, predict env_xy coords from environment

            self.obs_x = len(self.env_xy[:, 0])
            self.obs_y = len(self.env_xy[:, 1])
            
            # -- 1. grid of query points ------------------
            self._coords_flat = self.env_xy

            if isinstance(self._coords_flat, torch.Tensor):
                #coords = self._coords_flat.view(self.obs_y, self.obs_x, 2) # torch view/reshape
                coords = torch.tensor(self._coords[:, 1], self._coords[:, 0])
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
    
