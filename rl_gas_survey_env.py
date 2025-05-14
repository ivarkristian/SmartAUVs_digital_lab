import gpytorch.constraints
import torch
import gpytorch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import time

from gpt_class_exactgpmodel import ExactGPModel
import path
import chem_utils

# %%
# Definitions

# %%
class GasSurveyEnv(gym.Env):
    def __init__(self, scenario_bank, gp_ls_constraint=gpytorch.constraints.Interval(9, 11), gp_kernel_type='scale_rbf', gp_pred_resolution=[100, 100], timer=False):
        super(GasSurveyEnv, self).__init__()
        self.timer = timer
        # Load scenario bank
        self.scenario_bank = scenario_bank
        
        # Get statistics for environments
        min_concentration, max_concentration = self.scenario_bank.get_minmax()
        self.min_concentration = min_concentration.item()
        self.max_concentration = max_concentration.item()
        mu_all, sigma2_all = self.scenario_bank.get_mu_sigma2()
        self.mu_all = mu_all.item()
        self.sigma2_all = sigma2_all.item()
        
        # GP model parameters
        self.ls_const = gp_ls_constraint
        self.kernel_type = gp_kernel_type
        self.gp_pred_resolution = gp_pred_resolution

        # Steps until truncated=True (done)
        self.n_episodes = 0
        self.n_steps_max = 30
        self.acc_reward = 0
        
        # μ, σ, visited, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of 'visited' include location channels
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        self.channels = np.array([0, 1, 1, 0, 0])

        # reset draws a random scenario, initializes GP model and sample memory
        self.reset()
    
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.uint8
        )

         # Action space is a float anywhere inside this box:
        self.action_space = spaces.Box( low=np.array([-1.0, -1.0]),
                                        high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Summary:
        # Observations are 0-255. Prediction means and variances must be scaled in two steps; first
        # to scale with the overall mean and variance of the entire dataset, secondly to 0-255.
        # Location (or all visited locations) is 255, non-visited locations is 0.
        # Action is [-1 1], and must be scaled to [env_min, env_max]. When converting from action to
        # new location, action will be discretized to the resolution of [obs_x, obs_y]
        # Sampling is done along the computed locations, which do not necessarily adhere to the 
        # [obs_x, obs_y] raster. Prediction is done with the [obs_x, obs_y] raster.

    def reset(self, seed=None, options=None):
        t = time.process_time()
        self.n_episodes += 1
        self.n_steps = 0

        if self.n_episodes % 1 == 0:
            print(f'Ep {self.n_episodes}, mean reward = {(self.acc_reward/self.n_episodes):.3}')

        # Draw a random scenario/snapshot
        random_env = self.scenario_bank.sample()
        self.env_xy = random_env['coords']
        self.values = random_env['values']
        self.parameter = random_env['parameter']
        self.depth = random_env['depth']
        self.time = random_env['time']
        self.cur_dir = random_env['cur_dir']
        self.cur_str = random_env['cur_str']

        self.env_x_max = self.env_xy[:, 0].max().item()
        self.env_y_max = self.env_xy[:, 1].max().item()
        
        self.done = False

        # Init observation channels
        self._create_obs_coords()

        self.pred_mu_norm = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_mu_norm_clipped = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_var_norm = np.zeros_like(self.pred_mu_norm)
        self.pred_var_norm_clipped = np.zeros_like(self.pred_mu_norm)
        self.location = np.zeros_like(self.pred_mu_norm)

        # Init GP model
        #lower_norm = self.ls_const.lower_bound/self.env_x_max
        #upper_norm = self.ls_const.upper_bound/self.env_x_max
        self.ls_const = gpytorch.constraints.Interval(self.ls_const.lower_bound, self.ls_const.upper_bound)
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        self.mdl.covar_module.outputscale = torch.tensor(1.0)   # fixed prior variance 1
        self.variance_prior = self.mdl.get_outputscale()   # = 1.0
       
        # mu_all, sigma2_all min_all and max_all
        # are now retrived from scenario bank based on downsampled values!
        self.values_norm_minmax = self._norm_minmax()
        self.values_norm_zscale = self._norm_zscale()
        
        # Init sample memory. Could include lawnmower path samples.
        self.sampled_coords = torch.tensor([])
        self.sampled_coords_norm = torch.tensor([])
        self.sampled_vals = torch.tensor([])
        
        # Init location. Should be random
        loc_x = self.env_x_max * random.random()
        loc_y = self.env_y_max * random.random()
        self.old_loc = torch.tensor([loc_x, loc_y, self.depth], dtype=torch.float32)
        self.location[round(loc_y)][round(loc_x)] = 255

        # Init prediction tensors
        self.pred_mu = torch.tensor([])
        self.pred_var = torch.tensor([])
        
        obs = self._render_layers()
        info = {}
        
        if self.timer:
            print(f'reset took: {time.process_time() - t}')

        return obs, info

    def step(self, action, speed=1.0, sample_freq=1.0):
        tt = time.process_time()
        t = time.process_time()
        self.n_steps += 1
        # action = absolute (x,y) or Δx,Δy; clip, update GP, rewards...
        start_time = '2020-01-01T02:10:00.000000000' # dummy time
        synoptic = True
        old_loc_y, old_loc_x = np.argwhere(self.location)[0]
            #old_loc = torch.tensor([int(old_loc_x), int(old_loc_y)], dtype=torch.int)
        old_var = self.pred_var_norm # remember to compare old_var with new pred_norm, not new pred

        # expects action to be [-1.0, -1.0] [1.0, 1.0], convert to locs within [0, 255]
        out_of_bounds = not self.action_space.contains(action)
        if out_of_bounds:
            action = np.clip(action, self.action_space.low, self.action_space.high)

        new_loc = torch.tensor((action+1.0)/2.0*[self.env_x_max, self.env_y_max], dtype=torch.float32)

        #old_loc = self._append_z_to_xy(old_loc)
        new_loc = self._append_z_to_xy(new_loc)
        if self.timer:
            print(f't0 step: {time.process_time()-t}')

        if torch.equal(self.old_loc, new_loc):
            obs, truncated, info = self._get_obs_truncated_info()
            reward = -1.0
            self.acc_reward += reward
            return obs, float(reward), False, truncated, info

        t = time.process_time()
        sample_coords = path.path([self.old_loc, new_loc], start_time, speed, sample_freq, synoptic)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = torch.tensor([(float(t[0]), float(t[1])) for t in sample_coords])
        #sample_coords_norm = sample_coords_xy / torch.tensor([self.env_x_max, self.env_y_max])
        if self.timer:
            print(f't1 step: {time.process_time()-t}')
        
        measurements = torch.zeros(len(sample_coords_xy), dtype=torch.float32)
        radius = 1.0 # Radius of sample averaging
        
        t = time.process_time()
        # Sampling from the z-scaled values
        for c, coord in enumerate(sample_coords_xy):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self._coords_flat[:, 0], self._coords_flat[:, 1], self.values_norm_zscale, coord.numpy(), radius)
            #print(f'{coord} - {measurements[c]}')
        if self.timer:
            print(f't2 step: {time.process_time()-t}')
        
        self.sampled_coords = torch.cat((self.sampled_coords, sample_coords_xy))
        #self.sampled_coords_norm = torch.cat((self.sampled_coords_norm, sample_coords_norm))
        self.sampled_vals = torch.cat((self.sampled_vals, measurements))

        t = time.process_time()
        self._estimate() # fill self.pred_mu self.pred_var and normalized equivalents
        if self.timer:
            print(f't3 step: {time.process_time()-t}')
        
        # Update location
        self.old_loc = new_loc
        self.location[old_loc_y, old_loc_x] = 0

        x_indice = (self.obs_x-1)/self.obs_x
        y_indice = (self.obs_y-1)/self.obs_y
        self.location[int(np.round(new_loc[1]*y_indice)), int(np.round(new_loc[0]*x_indice))] = 255
        
        # compute reward (based on decrease in overall variance)
        reward = (old_var - self.pred_var_norm).sum()

        obs, truncated, info = self._get_obs_truncated_info()
        self.acc_reward += reward
        
        if self.timer:
            print(f'step took: {time.process_time()-tt}')
        
        return obs, float(reward), False, truncated, info
    
    def render():
        pass

    def close():
        pass
    
    def _get_obs_truncated_info(self):
        obs  = self._render_layers()
        truncated = (self.n_steps >= self.n_steps_max)
        info = {}
        return obs, truncated, info

    def _estimate(self):
        if self.mdl is None:
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        
        self.mdl.set_train_data(self.sampled_coords, self.sampled_vals, strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(self._coords_flat))
    
        self.pred_mu = self._tensor_to_obs_channel(current_pred.mean)
        self.pred_var = self._tensor_to_obs_channel(current_pred.variance)

        # De-zscale prediction means
        self.pred_mu_dezscale = self.pred_mu * self.sigma2_all + self.mu_all
        # Scale to 0-255 ([min_conc, max_conc] from scenario bank)
        self.pred_mu_norm = (self.pred_mu_dezscale - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm = self.pred_var/self.variance_prior * 255
        #pred_mu_norm = (pred_mu_dezscale - self.values.min()) / (self.values.max() - self.values.min() + 1e-8)

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
        
        self._ensure_normalization()

        # List all *possible* layers in a canonical order
        candidate_layers = [
            self.pred_mu_norm_clipped,     # idx 0
            self.pred_var_norm_clipped,  # idx 1
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
        stacked = np.stack(chosen_layers, axis=0).astype(np.uint8)
        return stacked
    
    def _ensure_normalization(self):
        self.pred_mu_norm_clipped = np.clip(self._tensor_to_obs_channel(self.pred_mu_norm), 0, 255)
        self.pred_var_norm_clipped = np.clip(self._tensor_to_obs_channel(self.pred_var_norm), 0, 255)
    
    def _create_obs_coords(self):
        
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
        self.coords_flat_norm = self._coords_flat/torch.tensor([self.env_x_max, self.env_y_max])
        self.coord_x_norm = (gx / xs.max())  # (H, W)
        self.coord_y_norm = (gy / ys.max())  # (H, W)

        return
    
    def _tensor_to_obs_channel(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.view(self.obs_y, self.obs_x).cpu().numpy()
        
        return tensor.reshape(self.obs_y, self.obs_x)
    
    def _print_info(self):
        print(f'Currently loaded env: {self.parameter} ({self.depth} {self.time})')
    
    def _append_z_to_xy(self, xy):
        if len(xy) == 2:
            return torch.cat((xy, torch.tensor([self.depth], dtype=torch.int)))
        else:
            return xy

    def plot_env(self, x=None, y=None, c=None, path=None, x_range=[0, 250], y_range=[0, 250]):

        if x is None:
            x = self._coord_x
        if y is None:
            y = self._coord_y
        if c is None:
            c = self.values

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=c, cmap='coolwarm', s=1, vmin=c.min(), vmax=c.max())
        if path is not None:
            ax.scatter(path[:, 0], path[:, 1], c='black', s=1)
        
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Value')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title(f"Time {self.time}, {self.parameter} at {self.depth}m depth ()")

        return fig, ax
        
        #print(f"Could not plot dataset = {self.dataset}, parameter = {env['parameter']}")

    
