from memory_profiler import profile
import gpytorch.constraints
import torch
import gpytorch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

from gpt_class_exactgpmodel import ExactGPModel
import path
import chem_utils

# %%
# Definitions

# %%
class GasSurveyEnv(gym.Env):
    def __init__(self, scenario_bank, gp_ls_constraint=gpytorch.constraints.Interval(9, 11), gp_kernel_type='scale_rbf', gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], timer=False, debug=False):
        super(GasSurveyEnv, self).__init__()
        self.debug = debug
        self.timer = timer
        self.a_var, self.a_dist = r_weights
        # Device selection supporting CUDA, MPS (Apple Silicon), or CPU
        #if torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load scenario bank
        self.scenario_bank = scenario_bank
        self.min_concentration, self.max_concentration = map(
            float, self.scenario_bank.get_minmax()
        )
        self.mu_all, self.sigma2_all = map(float, self.scenario_bank.get_mu_sigma2())
        
        # GP model parameters
        self.ls_const = gp_ls_constraint
        self.kernel_type = gp_kernel_type
        self.obs_x, self.obs_y = gp_pred_resolution

        # Steps until truncated=True (done)
        self.n_episodes = 0
        self.n_steps_max = 20
        self.acc_reward = 0.0
        
        # μ, σ, location, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of location include 'visited' channel
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        self.channels = np.array([0, 1, 0, 1, 1])

        self.max_samples = 0
        # reset draws a random scenario, initializes GP model and sample memory
        obs, _ = self.reset()
    
        # observation space
        self.observation_layers = spaces.Box(
            low=0, high=255, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.uint8
        )
        
        self.observation_space = spaces.Dict({
            "map": self.observation_layers,
            "loc": spaces.Box(-1.0, 1.0, (2,), np.float32),
        })

         # Action space is a float anywhere inside this box:
        self.action_space = spaces.Box( low=np.array([-1.0, -1.0]),
                                        high=np.array([1.0, 1.0]), dtype=np.float32)

    #@profile
    def reset(self, seed=None, options=None):
        if self.n_episodes % 10 == 0 and self.n_episodes:
            print(f'Ep {self.n_episodes}, mean reward = {(self.acc_reward/self.n_episodes):.3}')

        t = time.process_time()
        self.n_episodes += 1
        self.n_steps = 0
        self.terminated = False

        # Draw a random scenario/snapshot
        random_env = self.scenario_bank.sample()

        self.env_xy = random_env['coords'].to(self.device)
        self.values = random_env['values'].to(self.device)
        self.env_x_np = self.env_xy[:, 0].cpu().numpy()
        self.env_y_np = self.env_xy[:, 1].cpu().numpy()
        self.env_vals_np = self.values.cpu().numpy()

        self.parameter = random_env['parameter']
        self.depth = random_env['depth']
        self.time = random_env['time']
        self.cur_dir = random_env['cur_dir']
        self.cur_str = random_env['cur_str']

        if self.debug:
            print(f"Sampled env '{self.parameter}', depth {self.depth}', time {self.time}")

        self.env_x_max = float(self.env_xy[:, 0].max())
        self.env_y_max = float(self.env_xy[:, 1].max())
        self.maxdist=((self.env_x_max**2 + self.env_y_max**2)**0.5)

        # Init observation channels
        self._create_obs_coords()
        #self._get_cached_grid(self.env_x_max, self.env_y_max)

        self.pred_mu_norm = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_mu_norm_clipped = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_var_norm = np.zeros_like(self.pred_mu_norm) + self.sigma2_all
        self.pred_var_norm_clipped = np.zeros_like(self.pred_mu_norm) + self.sigma2_all
        self.location = np.zeros_like(self.pred_mu_norm)

        # Init GP model
        if hasattr(self, 'mdl'):
            del self.mdl
            
        if hasattr(self, 'llh'):
            del self.llh
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.llh = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        empty_x = torch.empty((0, 2), device=self.device)
        empty_y = torch.empty(0, device=self.device)
        self.mdl = ExactGPModel(
            empty_x,
            empty_y,
            self.llh,
            type=self.kernel_type,
            lengthscale_constraint=self.ls_const,
        ).to(self.device)
        self.mdl.covar_module.outputscale = self.sigma2_all
        self.mdl.eval()
        self.llh.eval()
       
        self.values_submuall = self.values-self.mu_all
        
        # Init sample memory. Could include lawnmower path samples.
        self.max_samples_old = self.max_samples
        self.max_samples = int(self.maxdist*self.n_steps_max)
        self.sample_idx = 0
        self.sample_idx_mdl = 0

        if (hasattr(self, 'sampled_coords') is False) or (self.max_samples != self.max_samples_old):
            # Preallocate memory (on GPU)
            self.sampled_coords = torch.empty((self.max_samples, 2), device=self.device)
        
        if (hasattr(self, 'sampled_vals') is False) or (self.max_samples != self.max_samples_old):
            # Preallocate memory (on GPU)
            self.sampled_vals = torch.empty(self.max_samples, device=self.device)
        
        #self.sampled_coords = torch.empty((0, 2), device=self.device)
        #self.sampled_vals = torch.empty(0, device=self.device)
        
        # Init location. Should be random
        loc_x = (self.env_x_max-1) * random.random()
        loc_y = (self.env_y_max-1) * random.random()
        self.obs_x_len=self.env_x_max/self.obs_x
        self.obs_y_len=self.env_y_max/self.obs_y

        self.loc = torch.tensor([loc_x, loc_y, self.depth], device=self.device)
        if self.debug:
            print(f'reset loc: {loc_x}, {loc_y}')

        ind_x, ind_y = self.loc_to_ind((loc_x, loc_y))
        self.location[ind_y, ind_x] = 255

        # Init prediction tensors
        self.pred_mu = np.zeros((self.obs_y, self.obs_x), dtype=np.float32)
        self.pred_var = np.full_like(self.pred_mu, self.sigma2_all)
        
        if self.debug:
            self._assert_gpu_consistency()

        #obs = self._render_layers()
        #info = {}
        obs, _, info = self._get_obs_truncated_info()
        
        if self.timer:
            print(f'reset took: {time.process_time() - t}')

        return obs, info

    #@profile
    def step(self, action, speed=1.0, sample_freq=1.0):
        tt = time.process_time()
        t = time.process_time()
        self.n_steps += 1
        reward = 0.0
        # action = absolute (x,y) or Δx,Δy; clip, update GP, rewards...
        start_time = '2020-01-01T02:10:00.000000000' # dummy time
        synoptic = True
        old_ind_y, old_ind_x = np.argwhere(self.location)[0]
        old_var = self.pred_var_norm_clipped # remember to compare with correct new var  (norm, clipped etc.)

        if self.debug:
            print(f'action: {action}', end=' ')

        # expects action to be [-1.0, -1.0] [1.0, 1.0], convert to locs within [0, 255]
        out_of_bounds = not self.action_space.contains(action)
        if out_of_bounds:
            action = np.clip(action, self.action_space.low, self.action_space.high)

        new_xy = ((action + 1.0) / 2.0) * np.array(
            [self.env_x_max, self.env_y_max], dtype=np.float32
        )
        self.new_loc = torch.as_tensor([*new_xy, self.depth], dtype=torch.float32, device=self.device)
        
        if self.timer:
            print(f't0 step: {time.process_time()-t}')

        if torch.allclose(self.loc, self.new_loc):
            obs, truncated, info = self._get_obs_truncated_info()
            reward += -5.0
            self.acc_reward += reward
            if self.debug:
                print(f'torch.allclose = True')
            return obs, float(reward), self.terminated, truncated, info

        t = time.process_time()
        sample_coords = path.path([self.loc.cpu(), self.new_loc.cpu()], start_time, speed, sample_freq, synoptic)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = [(float(k[0]), float(k[1])) for k in sample_coords]
        
        if self.timer:
            print(f't1 step: {time.process_time()-t}')
        
        measurements = np.zeros(len(sample_coords_xy), dtype=np.float32)
        radius = 1.0 # Radius of sample averaging
        
        t = time.process_time()
        # Sampling from the z-scaled values
        for c, coord in enumerate(sample_coords_xy):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.env_x_np, self.env_y_np, self.env_vals_np, coord, radius)

        if self.timer:
            print(f't2 step: {time.process_time()-t}')
        
        if self.debug:
            print(f'#Smp: {len(sample_coords_xy)}', end=' ')

        end_idx = self.sample_idx + len(sample_coords_xy)
        if end_idx > self.max_samples:
            raise RuntimeError(f"Exceeded maximum number of samples ({self.max_samples})")

        # Store new samples into the preallocated tensors
        self.sampled_coords[self.sample_idx:end_idx] = torch.as_tensor(sample_coords_xy, device=self.device, dtype=self.sampled_coords.dtype)
        self.sampled_vals[self.sample_idx:end_idx] = torch.as_tensor(measurements, device=self.device, dtype=self.sampled_vals.dtype)
        self.sample_idx = end_idx

        t = time.process_time()
        self._estimate() # fill self.pred_mu, self.pred_var and norms
        if self.timer:
            print(f't3 step: {time.process_time()-t}')
        
        # Update location
        self.loc = self.new_loc.detach()
        self.location[old_ind_y, old_ind_x] = 0

        ind_x, ind_y = self.loc_to_ind((self.loc[0].item(), self.loc[1].item()))
        self.location[ind_y][ind_x] = 255
        
        # compute reward (based on decrease in overall variance)
        if self.debug:
            print(f'old_var.mean: {old_var.mean():.4} pred_var_norm.mean: {self.pred_var_norm.mean():.4}')

        var_red = (old_var.mean() - self.pred_var_norm.mean())
        #r_var = 1 + 10*var_red.mean()/old_var.mean()
        r_var = var_red # reward for reducing variance
        #r_var = 2*var_red/(float(self.mdl.get_lengthscale())*len(sample_coords_xy)*old_var.mean())
        r_dist = -2.0 # penalty for changing course
        r_term = 0.0

        if self.pred_var.mean() <= 100:
            r_term = self.n_steps_max - self.n_steps
            self.terminated = True

        # IMPLEMENT REWARD SCALING ~1
        reward += self.a_var*r_var + self.a_dist*r_dist + r_term

        obs, truncated, info = self._get_obs_truncated_info()
        self.acc_reward += reward
        
        if self.timer:
            print(f'step took: {time.process_time()-tt}')
        if self.debug:
            print(f'r_var: {r_var:.4}, r_dist: {r_dist:.4}, r_tot: {reward:.4}')
            #self._assert_gpu_consistency()

        return obs, float(reward), self.terminated, truncated, info
    
    def render():
        pass

    def close():
        pass
    
    def loc_to_ind(self, loc: Tuple[float, float]) -> Tuple[int, int]:
        x_idx = min(int(round(loc[0] / (self.env_x_max / self.obs_x))), self.obs_x - 1)
        y_idx = min(int(round(loc[1] / (self.env_y_max / self.obs_y))), self.obs_y - 1)
        return x_idx, y_idx

    def _get_obs_truncated_info(self):
        layers_uint8  = self._render_layers()
        loc_x = ((self.loc[0]/self.env_x_max)*2.0 - 1.0).cpu()
        loc_y = ((self.loc[1]/self.env_y_max)*2.0 - 1.0).cpu()

        obs_dict = {
            "map": layers_uint8,           # (C,H,W)
            "loc": np.array([loc_x, loc_y], np.float32),
        }

        truncated = (self.n_steps >= self.n_steps_max)
        info = {}
        return obs_dict, truncated, info

    def _assert_gpu_consistency(self):

        for p in self.mdl.parameters():
            if str(p.device.type) != str(self.device):
                print(f'p.device.type: {p.device.type}, self.device: {self.device}')
                print(f'{p} - {p.device}')
        
        for t in self.mdl.train_inputs + (self.mdl.train_targets,):
            if str(t.device.type) != str(self.device):
                print(f'{t} - {t.device}')
        
        for p in self.llh.parameters():
            if str(p.device.type) != str(self.device):
                print(f'{p} - {p.device}')

        # 1. parameters
        assert all(str(p.device.type) == str(self.device) for p in self.mdl.parameters()), \
            "Some model parameters are not on the target device"

        # 2. training data
        for t in self.mdl.train_inputs + (self.mdl.train_targets,):
            assert str(t.device.type) == str(self.device), "GP training tensor on wrong device"

        # 3. likelihood parameters
        assert all(str(p.device.type) == str(self.device) for p in self.llh.parameters()), \
            "Likelihood parameters not on target device"

    #@profile
    def _estimate(self):
        if self.mdl is None:
            if self.debug:
                print(f'Created model in ._estimate()')
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals-self.mu_all, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const)
        
        t = time.process_time()
        #self.mdl.set_train_data(
        #    inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
        if self.n_steps > 1:
            # not first prediction, so choose between using fantasy mdl or set_train_data
            #if self.sample_idx - self.sample_idx_mdl > 500:
                # set train data
            #    self.mdl.set_train_data(
            #        inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
            #    self.sample_idx_mdl = self.sample_idx
            #    use_self_mdl = True
            #    if self.debug:
            #        print(f"Set train data")
            #else:
                # use fantasy model, adding samples from idx_mdl to idx
            #self.mdl = self.mdl.get_fantasy_model(self.sampled_coords[self.sample_idx_mdl:self.sample_idx], self.sampled_vals[self.sample_idx_mdl:self.sample_idx]-self.mu_all)
            self.mdl = self.mdl.get_fantasy_model(self.sampled_coords[self.sample_idx_mdl:self.sample_idx], self.sampled_vals[self.sample_idx_mdl:self.sample_idx]-self.mu_all)
            self.sample_idx_mdl = self.sample_idx
            use_self_mdl = True
        else:
            # first prediction must have train data
            self.mdl.set_train_data(
                inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
            self.sample_idx_mdl = self.sample_idx
            if self.debug:
                self.mdl.print_named_parameters()
            #use_self_mdl = True
            
        if self.timer:
            print(f't3.1 step: {time.process_time()-t}')

        # Then predict
        t = time.process_time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #if use_self_mdl:
            current_pred = self.mdl(self._coords_flat)
            #else:
            #    current_pred = mdl_fantasy(self._coords_flat)

        if self.timer:
            print(f't3.2 step: {time.process_time()-t}')

        t = time.process_time()
        self.pred_mu = self._tensor_to_obs_channel(current_pred.mean + self.mu_all)
        self.pred_var = self._tensor_to_obs_channel(current_pred.variance)
        if self.timer:
            print(f't3.3 step: {time.process_time()-t}')
        
        # Scale to 0-255 ([min_conc, max_conc] from scenario bank)
        t = time.process_time()
        self.pred_mu_norm = (self.pred_mu - self.min_concentration) / (self.max_concentration - self.min_concentration) * 255
        self.pred_var_norm = self.pred_var/self.sigma2_all * 255
        if self.timer:
            print(f't3.4 step: {time.process_time()-t}')

        return

    def _norm_minmax(self):
        return (self.values - self.min_concentration)/(self.max_concentration - self.min_concentration)
    
    def _norm_zscale(self):
        return (self.values - self.mu_all)/(self.sigma2_all**(0.5))

    #@profile
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
        self.pred_mu_norm_clipped = np.clip(self.pred_mu_norm, 0, 255).astype(np.uint8)
        self.pred_var_norm_clipped = np.clip(self.pred_var_norm, 0, 255).astype(np.uint8)
    
    def _get_cached_grid(self, x_max: float, y_max: float) -> torch.Tensor:
        """Return (H*W,2) tensor on self.device; cache between envs."""
        key = (x_max, y_max, self.obs_x, self.obs_y, self.device.type)
        if key not in GasSurveyEnv._grid_cache:
            xs = torch.linspace(0, x_max, self.obs_x, device=self.device)
            ys = torch.linspace(0, y_max, self.obs_y, device=self.device)
            gx, gy = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
            grid = torch.stack((gx, gy), dim=-1).view(-1, 2)  # (H*W,2)
            GasSurveyEnv._grid_cache[key] = grid
        return GasSurveyEnv._grid_cache[key]

    #@profile
    def _create_obs_coords(self):

        # -- 1. grid of query points -----------------
        #   (H*W, 2) tensor that GPyTorch will accept.
        xs = np.linspace(0, self.env_x_max, self.obs_x, dtype=np.float32)
        ys = np.linspace(0, self.env_y_max, self.obs_y, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)                        # shape (H, W)

        # Save as 2‑D field for coord‑channels and as flat list for GP queries
        self._coord_x = gx           # (H, W)
        self._coord_y = gy           # (H, W)
        self._coords = np.stack([gx, gy], axis=-1).reshape(-1, 2)      # (H, W, 2)
        self._coords_flat = torch.as_tensor(self._coords, device=self.device)
        
        # -- 2. static coordinate channels, normalised [0, 255] --
        self.coord_x_norm = (gx / self.env_x_max) * 255  # (H, W)
        self.coord_y_norm = (gy / self.env_y_max) * 255  # (H, W)

        return
    
    #@profile
    def _tensor_to_obs_channel(self, t: torch.Tensor) -> np.ndarray:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(t)}")

        # Ensure tensor is flat with expected size
        expected_size = self.obs_x * self.obs_y
        if t.numel() != expected_size:
            raise ValueError(f"Tensor has {t.numel()} elements, expected {expected_size}")

        return t.view(self.obs_y, self.obs_x).detach().cpu().numpy()
    
    def _print_info(self):
        print(f'Currently loaded env: {self.parameter} ({self.depth} {self.time})')
    
    def _append_z_to_xy(self, xy):
        if len(xy) == 2:
            return torch.cat((xy, torch.tensor([self.depth], dtype=torch.int)))
        else:
            return xy

    def plot_env(self, x=None, y=None, c=None, path=None, x_range=[0, 250], y_range=[0, 250]):

        if x is None:
            x = self.env_xy[:, 0]
        if y is None:
            y = self.env_xy[:, 1]
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

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

class MapPlusLocExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space: spaces.Dict, features_dim=512):
        super().__init__(obs_space, features_dim)
        self.cnn = NatureCNN(obs_space["map"], features_dim=256)
        self.linear = torch.nn.Linear(256 + 2, features_dim)

    def forward(self, obs):
        map_feats = self.cnn(obs["map"])
        x = torch.cat([map_feats, obs["loc"]], dim=1)
        return torch.relu(self.linear(x))    
