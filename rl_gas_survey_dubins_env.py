from memory_profiler import profile
import gpytorch.constraints
import torch
import gpytorch
import gc
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import time
from typing import Tuple
from stable_baselines3.common.buffers import DictReplayBuffer, DictReplayBufferSamples

import dubins

from gpt_class_exactgpmodel import ExactGPModel
import chem_utils
import agents

# %%
# Definitions

# %%
class GasSurveyDubinsEnv(gym.Env):
    def __init__(self, scenario_bank=None, gp_ls_constraint=gpytorch.constraints.Interval(9, 11), gp_kernel_type='scale_rbf', gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], turn_radius=250, channels=np.array([0, 1, 0, 0, 0]), timer=False, debug=False, device=torch.device("cpu")):
        super(GasSurveyDubinsEnv, self).__init__()
        self.debug = debug
        self.timer = timer
        self.print_info_rate = 10
        self.turn_radius = turn_radius
        self.path_planner = dubins.Dubins(self.turn_radius-2, 1.0) #1.0 - sample every meter

        self.a_var, self.a_dist = r_weights
        
        self.device = device
        # Load scenario bank
        if not scenario_bank:
            print(f"You have to provide a scenario bank!")
            return
        
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
        self.n_steps = 0
        self.n_episodes = 0
        self.n_steps_max = 100
        self.total_steps = 0
        self.acc_reward = 0.0
        
        # μ, σ, location, coord‑Y, coord‑X  → 5 possible channels
        # Could instead of location include 'visited' channel
        # Including coord_x/y channels is a bit dangerous, should
        # randomize direction of scenarios, e.g. rotate by 90/180 deg
        # to avoid 'learning the coordinate system'
        self.channels = channels

        self.max_samples = 0
        self.location_noise = 0.05
        self.location_radius = self.ls_const.upper_bound.item()
        # reset draws a random scenario, initializes GP model and sample memory
        obs, _ = self.reset()
    
        # observation space
        self.observation_layers = spaces.Box(
            low=0, high=255, shape=(self.channels.sum(), self.obs_x, self.obs_y), dtype=np.uint8
        )
        
        self.observation_space = spaces.Dict({
            "map": self.observation_layers,
            "loc": spaces.Box(-1.0, 1.0, (2,), np.float32),
            "hdg": spaces.MultiBinary(4)
        })

        #Discrete action space, left - 0, straight - 1, right - 2:
        self.action_space = spaces.Discrete(3)

        print(f'Init dubins env, \ndevice: {self.device}\nturn_radius: {self.turn_radius}\nchannels: {self.channels}')

    #@profile
    def reset(self, seed=None, options=None):
        
        if self.n_episodes % self.print_info_rate == 0 and self.n_episodes:
            print(f'Ep {self.n_episodes}, mean reward = {(self.acc_reward/self.print_info_rate):.3}')
            self.acc_reward = 0

        t = time.process_time()
        self.n_episodes += 1
        self.total_steps += self.n_steps
        self.n_steps = 0
        self.terminated = False

        # Draw a random scenario/snapshot
        rotation = random.choice([-90, 0, 90, 180])
        random_scenario = self.scenario_bank.sample()

        self.env_xy = self._rotate_xy(random_scenario['coords'].to(self.device), rotation)
        self.values = random_scenario['values'].to(self.device)
        self.env_x_np = self.env_xy[:, 0].cpu().numpy()
        self.env_y_np = self.env_xy[:, 1].cpu().numpy()
        self.env_vals_np = self.values.cpu().numpy()

        self.parameter = random_scenario['parameter']
        self.depth = random_scenario['depth']
        self.time = random_scenario['time']
        self.cur_dir = random_scenario['cur_dir'] + rotation
        self.cur_str = random_scenario['cur_str']

        if self.debug:
            print(f"Sampled env '{self.parameter}', depth {self.depth}', time {self.time}")

        self.env_x_max = float(self.env_xy[:, 0].max())
        self.env_y_max = float(self.env_xy[:, 1].max())

        self.maxdist=2*math.pi*self.turn_radius/4.0

        # Init observation channels
        self._create_obs_coords()
        #self._get_cached_grid(self.env_x_max, self.env_y_max)

        self.pred_mu_norm = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_mu_norm_clipped = np.zeros((self.obs_y, self.obs_x), dtype=np.uint8)
        self.pred_var_norm = np.zeros_like(self.pred_mu_norm) + 255#self.sigma2_all
        self.pred_var_norm_clipped = np.zeros_like(self.pred_mu_norm) + 255#self.sigma2_all
        self.location = np.zeros_like(self.pred_mu_norm)

        # Init GP model
        if hasattr(self, 'mdl'):
            self.mdl.cpu()
            del self.mdl
            
        if hasattr(self, 'llh'):
            self.llh.cpu()
            del self.llh
        
        gc.collect()
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
        self.max_samples = int((round(self.maxdist+0.5) + 1) * (1 + self.location_noise) *(self.n_steps_max + 1))
        self.sample_idx = 0
        self.sample_idx_mdl = 0

        if (hasattr(self, 'sampled_coords') is False) or (self.max_samples != self.max_samples_old):
            self.sampled_coords = torch.empty((self.max_samples, 2), device=self.device)
        
        if (hasattr(self, 'sampled_vals') is False) or (self.max_samples != self.max_samples_old):
            self.sampled_vals = torch.empty(self.max_samples, device=self.device)
        
        # Init location and heading. Should be random
        loc_x = (self.env_x_max-1) * random.random()
        loc_y = (self.env_y_max-1) * random.random()
        self.heading = np.zeros(4)
        self.heading[random.choice([0, 1, 2, 3])] += 1

        self.obs_x_len=self.env_x_max/self.obs_x
        self.obs_y_len=self.env_y_max/self.obs_y

        self.loc = torch.tensor([loc_x, loc_y, self.depth], device=self.device)
        if self.debug:
            print(f'reset loc: {loc_x}, {loc_y}')

        self.make_circle(self.loc[0].cpu().numpy(), self.loc[1].cpu().numpy(), self.location_radius)

        # Init prediction tensors
        self.pred_mu = np.zeros((self.obs_y, self.obs_x), dtype=np.float32)
        self.pred_var = np.full_like(self.pred_mu, self.sigma2_all)
        
        if self.debug:
            self._assert_gpu_consistency()

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
        #old_ind_y, old_ind_x = np.argwhere(self.location)[0]
        old_var = self.pred_var_norm # remember to compare with correct new var  (norm, clipped etc.)

        # expects action to be up, down, left, right

        delta_xy, new_heading = self._dubins_delta(action, self.heading, self.turn_radius)
        noise = self._delta_add_noise(delta_xy, self.turn_radius)
        new_xy = self.loc[:2].cpu().numpy() + delta_xy + noise

        if self.debug:
            print(f'step: {self.n_steps} action: {delta_xy} ({noise}) new_xy: {new_xy} new_hdg: {new_heading}', end=' ')
        out_of_bounds = not ((0 <= new_xy[0] <= self.env_x_max) and (0 <= new_xy[1] <= self.env_y_max))
        if out_of_bounds:
            obs, truncated, info = self._get_obs_truncated_info()
            reward += -5.0
            self.acc_reward += reward
            if self.debug:
                print(f'out_of_bounds = True')
            return obs, float(reward), self.terminated, truncated, info
        else:
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
        #sample_coords = path.path([self.loc.cpu(), self.new_loc.cpu()], start_time, speed, sample_freq, synoptic)
        start = (self.loc[0], self.loc[1], self._onehot_to_rad(self.heading))
        end = (new_xy[0], new_xy[1], self._onehot_to_rad(new_heading))
        sample_coords_xy = self.path_planner.dubins_path(start, end)
        if self.debug:
            print(f'sample_coords: {sample_coords_xy}')
            agents.plot_n(x=sample_coords_xy[:, 0], y=sample_coords_xy[:, 1], data_list=[np.ones_like(sample_coords_xy[:, 0])], path=sample_coords_xy)
        
        if self.timer:
            print(f't1 step: {time.process_time()-t}')
        
        measurements = np.zeros(len(sample_coords_xy), dtype=np.float32)
        radius = 1.0 # Radius of sample averaging
        
        t = time.process_time()
        # Sampling from the z-scaled values
        for c, coord in enumerate(sample_coords_xy):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.env_x_np, self.env_y_np, self.env_vals_np, coord, radius)

        if self.debug:
            print(f'measurements: {measurements}')

        if self.timer:
            print(f't2 step: {time.process_time()-t}')
        
        if self.debug:
            print(f'#Smp: {len(sample_coords_xy)}', end=' ')

        end_idx = self.sample_idx + len(sample_coords_xy)
        if end_idx > self.max_samples:
            raise RuntimeError(f"Exceeded maximum number of samples ({end_idx} > {self.max_samples})")

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
        self.heading = new_heading
        #self.location[old_ind_y, old_ind_x] = 0
        self.make_circle(self.loc[0].cpu().numpy(), self.loc[1].cpu().numpy(), self.location_radius)
        
        # compute reward (based on decrease in overall variance)
        if self.debug:
            print(f'old_var.mean: {old_var.mean():.4} pred_var_norm.mean: {self.pred_var_norm.mean():.4}')

        var_red = min(2.0, (old_var.mean() - self.pred_var_norm.mean()))#2.0 is max possible reward for step length 20
        r_var = var_red # reward for reducing variance
        #r_var = var_red/float(len(sample_coords_xy)*0.0694)
        r_dist = -1.0 # step penalty (for changing course)
        r_term = 0.0

        if self.pred_var_norm.mean() <= 100:
            #r_term = self.n_steps_max - self.n_steps
            r_term = 5.0
            self.terminated = True
        
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
    
    def _onehot_to_rad(self, heading_1hot):
        '''
        Parameters
        ----------
        onehot - [north, south, west, east]

        Returns
        ----------
        one of [math.pi/2, -math.pi/2, -math.pi, math.pi]
        '''
        rads = [math.pi/2, -math.pi/2, math.pi, 0]
        try:
            idx = list(heading_1hot).index(1)
        except ValueError:
            raise ValueError("heading_1hot must have exactly one 1") from None

        return rads[idx]


    def _dubins_delta(self, action, heading_1hot, turn_radius: float | int):
        """
        Parameters
        ----------
        action : {"left", "straight", "right"}
        heading_1hot : iterable of length 4 [north, south, west, east]  e.g. [1,0,0,0] for north.
        turn_radius : positive float (or int)
        dtype, device : passed to the returned tensor

        Returns
        -------
        Δ : shape (2,) torch.Tensor
            (dx, dy) after executing the action.
        """
        # decode the action
        actions = ["left", "straight", "right"]
        act = actions[action]

        # validate & decode the heading ------------------------------------------
        headings = ('north', 'south', 'west', 'east')
        try:
            h_idx = list(heading_1hot).index(1)
        except ValueError:
            raise ValueError("heading_1hot must have exactly one 1") from None

        heading = headings[h_idx]

        # canonical mapping -------------------------------------------------------
        r = float(turn_radius)
        mapping = {
            ('north', 'straight'): (( 0,  r), [1, 0, 0, 0]),
            ('north', 'left')    : ((-r,  r), [0, 0, 1, 0]),
            ('north', 'right')   : (( r,  r), [0, 0, 0, 1]),

            ('south', 'straight'): (( 0, -r), [0, 1, 0, 0]),
            ('south', 'left')    : (( r, -r), [0, 0, 0, 1]),
            ('south', 'right')   : ((-r, -r), [0, 0, 1, 0]),

            ('west',  'straight'): ((-r,  0), [0, 0, 1, 0]),
            ('west',  'left')    : ((-r, -r), [0, 1, 0, 0]),
            ('west',  'right')   : ((-r,  r), [1, 0, 0, 0]),

            ('east',  'straight'): (( r,  0), [0, 0, 0, 1]),
            ('east',  'left')    : (( r,  r), [1, 0, 0, 0]),
            ('east',  'right')   : (( r, -r), [0, 1, 0, 0]),
        }

        try:
            (dx, dy), new_heading = mapping[(heading, act)]
        except KeyError:
            raise ValueError(f"invalid action '{act}'") from None

        return np.array([dx, dy]), np.array(new_heading)
    
    def _delta_add_noise(self, delta_xy, step, max_percentage=0.05):
        max_noise = max_percentage * step
        x_noise = (random.random() - 0.5)*2 * max_noise
        y_noise = (random.random() - 0.5)*2 * max_noise
        
        return np.array([x_noise, y_noise])

    def loc_to_ind(self, loc: Tuple[float, float]) -> Tuple[int, int]:
        x_idx = min(int(round(loc[0] / (self.env_x_max / self.obs_x))), self.obs_x - 1)
        y_idx = min(int(round(loc[1] / (self.env_y_max / self.obs_y))), self.obs_y - 1)
        return x_idx, y_idx
    
    def _rotate_xy(self, env_xy: torch.Tensor, d: float | int) -> torch.Tensor:
        """
        Rotate 2-D coordinates `env_xy` by `d` degrees **clockwise**.

        Returns
        -------
        rotated : (N, 2) torch.Tensor
            Rotated coordinates, same dtype and device as `env_xy`.
        """
        t = torch.tensor([env_xy[:, 0].mean(), env_xy[:, 1].mean()], device=env_xy.device)
        env_xy_zero_translated = env_xy - t
        # ensure float dtype on the same device as the input
        theta = torch.deg2rad(torch.as_tensor(d, dtype=env_xy.dtype,
                                            device=env_xy.device))

        c, s = torch.cos(theta), torch.sin(theta)
        rot_mat = torch.stack((torch.stack(( c,  -s)),
                            torch.stack((s,  c))))
        
        env_xy_rot = env_xy_zero_translated @ rot_mat.T

        return env_xy_rot + t

    def _get_obs_truncated_info(self):
        layers_uint8  = self._render_layers()
        loc_x = ((self.loc[0]/self.env_x_max)*2.0 - 1.0).cpu()
        loc_y = ((self.loc[1]/self.env_y_max)*2.0 - 1.0).cpu()

        obs_dict = {
            "map": layers_uint8,           # (C,H,W)
            "loc": np.array([loc_x, loc_y], np.float32),
            "hdg": self.heading
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
            self.mdl = ExactGPModel(self.sampled_coords, self.sampled_vals-self.mu_all, self.llh, self.kernel_type, lengthscale_constraint=self.ls_const).to(self.device)
        
        t = time.process_time()
        #self.mdl.set_train_data(
        #    inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
        if len(self.mdl.train_targets) > 0:
            # not first prediction, use fantasy mdl
            self.mdl = self.mdl.get_fantasy_model(self.sampled_coords[self.sample_idx_mdl:self.sample_idx], self.sampled_vals[self.sample_idx_mdl:self.sample_idx]-self.mu_all).to(self.device)
            self.sample_idx_mdl = self.sample_idx
        else:
            # first prediction must have train data
            self.mdl.set_train_data(
                inputs=self.sampled_coords[:self.sample_idx], targets=self.sampled_vals[:self.sample_idx]-self.mu_all, strict=False)
            self.sample_idx_mdl = self.sample_idx
            if self.debug:
                self.mdl.print_named_parameters()
            
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
        if key not in GasSurveyDiscEnv._grid_cache:
            xs = torch.linspace(0, x_max, self.obs_x, device=self.device)
            ys = torch.linspace(0, y_max, self.obs_y, device=self.device)
            gx, gy = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
            grid = torch.stack((gx, gy), dim=-1).view(-1, 2)  # (H*W,2)
            GasSurveyDiscEnv._grid_cache[key] = grid
        return GasSurveyDiscEnv._grid_cache[key]

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
    
    def make_circle(self, x: int, y: int, r: int):
        """
        Set all pixels inside radius `r` of (x, y) to 0 and the rest to 255.

        Parameters
        ----------
        self.location : np.ndarray              # 2-D array (H, W) you want to modify
        x, y  : int                     # centre coordinate (column, row)
        r     : int                     # radius in pixels (inclusive)

        Returns
        -------
        """
        # Boolean mask for the circle (≤ r²)
        mask = (self._coord_x - x)**2 + (self._coord_y - y)**2 <= r*r

        # Write values in place
        self.location.fill(255)     # everything white
        self.location[mask] = 0     # black disk

        return

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
        ax.set_title(f"Time {self.time}, {self.parameter} at -{self.depth}m. ({self.cur_str:.2}m/s @ {round(self.cur_dir)} deg)")

        return fig, ax

def get_q_values(model, obs):
    """
    Return the Q-value vector (one value per discrete action) for a single observation.
    """
    # 1. Convert raw obs (np array, dict, …) to a batched torch.Tensor on the
    #    same device as the policy
    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    # 2. Extract features (CNN/MLP) exactly as the policy does
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor)          # shape (1, n_actions)

    return q_values.cpu().numpy().squeeze(0)             # -> (n_actions,)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

class MapPlusLocExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space: spaces.Dict, features_dim=512):
        super().__init__(obs_space, features_dim)
        self.cnn = NatureCNN(obs_space["map"], features_dim=256)
        self.linear = torch.nn.Linear(256 + 2, features_dim)

    def forward(self, obs):
        device = self.linear.weight.device          # extractor is on same device as policy
        map_t = obs["map"].to(device).float().div(255.0)  # scale 0-1
        loc_t = obs["loc"].to(device)
        map_feats = self.cnn(map_t)
        return torch.relu(self.linear(torch.cat([map_feats, loc_t], dim=1)))

class CpuDictReplayBuffer(DictReplayBuffer):
    def __init__(self, *args, sample_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_device = torch.device(sample_device) if sample_device else None

    @staticmethod
    def _to_device(batch: DictReplayBufferSamples, device: torch.device):
        """Return a *new* DictReplayBufferSamples living on `device`."""
        obs        = {k: v.to(device) for k, v in batch.observations.items()}
        next_obs   = {k: v.to(device) for k, v in batch.next_observations.items()}
        actions    = batch.actions.to(device)
        rewards    = batch.rewards.to(device)
        dones      = batch.dones.to(device)
        return DictReplayBufferSamples(obs, actions, next_obs, dones, rewards)

    # override -----------------------------------------------------------
    def sample(self, batch_size: int, env=None, device=None):
        batch = super().sample(batch_size, env=env)   # still on CPU

        target_device = device or self.sample_device
        if target_device is not None:
            batch = self._to_device(batch, target_device)
        return batch

def show_conv3_maps(model, obs):
    conv3 = model.policy.q_net.features_extractor.cnn.cnn[4]  # 3rd Conv2d
    feature_bank = {}
    def _save_features(_, __, output):
        feature_bank["conv3"] = output.detach().cpu()
    h = conv3.register_forward_hook(_save_features)

    # ---- 3. forward pass through the extractor -----------------------
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        _ = model.policy.q_net.features_extractor(obs_tensor)

    h.remove()
    # fmap from the forward hook: shape (1, 64, 9, 9)
    fmap = feature_bank["conv3"].squeeze(0)          # (64, 9, 9)  remove batch dim

    # per-channel activation energy
    energy = fmap.abs().mean(dim=(1, 2)).cpu().numpy()   # (64,)

    # bar plot
    channels = np.arange(len(energy))        # x-positions: 0 … 63
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 2.0), dpi=300)
    axes.bar(channels, energy, width=0.8)
    plt.xlabel("Channel", fontsize=8)
    plt.ylabel("mean |activation|", fontsize=8)
    axes.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.show()

    # Top activation channels
    k = 9
    top_idx = energy.argsort()[-k:]
    rows = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(rows, rows, figsize=(rows*2, rows*2))

    for ax, idx in zip(axes.flat, top_idx):
        ax.imshow(fmap[idx], cmap="inferno")
        ax.set_title(f"ch {idx}", fontsize=6)
        ax.axis("off")
    plt.tight_layout(); plt.show()