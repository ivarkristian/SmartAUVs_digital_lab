import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gpytorch

import chem_utils
import path
from gpt_class_exactgpmodel import ExactGPModel

# %%
# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# Step 1: Define the environment wrapper
class EnvironmentWrapper:
    def __init__(self, data_dir = '../scenario_1c_medium/'):
        self.data_dir = data_dir
        self.data_file = None
        self.dataset = None
        self.parameter = None
        self.depth = None
        self.time = None
        self.val = None
        self.x = None
        self.y = None
        self.sampled_coords = torch.tensor([])
        self.sampled_vals = torch.tensor([])

        # Read and clean list of .nc files
        files = os.listdir(data_dir)
        # Create a new list with strings that end with '.nc'
        self.nc_files = [s for s in files if s.endswith('.nc')]
        self.nc_files.sort()
        self.print_data_files()
    
    def print_data_files(self):
        print(f'Directory {self.data_dir} containts these files of type .nc:\n{self.nc_files}')
        print(f'Run .load_dataset(nc_file) to load a file as a dataset')

    def load_dataset(self, nc_file=None):
        if isinstance(nc_file, int):
            data_file = self.data_dir + self.nc_files[nc_file]
        elif isinstance(nc_file, str):
            if nc_file in self.nc_files:
                data_file = self.data_dir + nc_file
            else:
                print(f'File {nc_file} not found in data_dir {self.data_dir}')
                return
        else:
            print(f'EnvironmentWrapper.load_dataset: Parameter {nc_file} not recognized as int or str')
            return

        self.data_file = data_file
        self.dataset = chem_utils.load_chemical_dataset(self.data_file)
    
    def set_env(self, parameter='pH', depth=67, time=1):
        self.parameter = parameter
        self.depth = depth
        self.time = time
        val_dataset = self.dataset[self.parameter].isel(time=self.time, siglay=self.depth)
        self.val = val_dataset.values[:72710]
        x = val_dataset['x'].values[:72710]
        y = val_dataset['y'].values[:72710]
        self.x = x - x.min()
        self.y = y - y.min()
        self.env_xy = torch.tensor(np.column_stack((self.x, self.y)), dtype=torch.float32)
        self.print_info()
        
    def print_info(self):
        print(f'Current directory: {self.data_dir}')
        print(f'Data file: {self.data_file}')
        print(f'Loaded dataset: {self.dataset}')
        print(f'Loaded parameter: {self.parameter} (depth={self.depth}, time={self.time})')

    def plot_env(self):
        if self.parameter:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(self.y, self.x, c=self.val, cmap='coolwarm', s=2, vmin=self.val.min(), vmax=self.val.max())
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Value')

            # Add labels and title
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.set_title(f'Time {self.time}, {self.parameter} at {self.depth}m depth')

            return fig, ax
        
        print(f'Could not plot dataset = {self.dataset}, parameter = {self.parameter}')


    def reset(self):
        self.sampled_coords = torch.tensor([])
        self.sampled_vals = torch.tensor([])
        return

    def step(self, old_loc, new_loc, speed, sampling_freq):
        # Move to new location while sampling
        # Action should specify old and new location and speed so that sampling points
        # and reward can be computed
        start_time = '2020-01-01T02:10:00.000000000' # dummy time

        # Perhaps make a more flexible function for non-synoptic sampling
        # (although that is much slower)
        synoptic = True
        print(f'old_loc: {old_loc}, new_loc: {new_loc}')
        sample_coords = path.path([old_loc, new_loc], start_time, speed, sampling_freq, synoptic)
        print(f'sample_coords: {sample_coords}')
        #sample_coords_xy = [(item[0], item[1]) for item in sample_coords]
        #sample_coords_xy = torch.tensor(sample_coords_xy)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = torch.tensor([(float(t[0]), float(t[1])) for t in sample_coords])
        print(f'sample_coords_xy: {sample_coords_xy}')

        measurements = torch.zeros(len(sample_coords_xy), dtype=torch.float32)
        radius = 1.0 # Radius of sample averaging
        for c, coord in enumerate(sample_coords_xy):
            measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.x, self.y, self.val, coord.numpy(), radius)
        
        self.sampled_coords = torch.cat((self.sampled_coords, sample_coords_xy))
        self.sampled_vals = torch.cat((self.sampled_vals, measurements))

        # next_state, reward, done, _ 
        return len(measurements)

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space.n

    @property
    def observation_space(self):
        return self.env.observation_space.shape[0]

# Step 2: Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Step 3: Define the Agent
class GPAgent:
    def __init__(self, input_dim, action_space, env_xy, speed, sampling_freq, small_grid_bins=5, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        self.length_constraint = gpytorch.constraints.Positive()
        self.kernel_name = 'scale_rbf'
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)
        self.env_xy = env_xy
        self.current_pred = None
        self.small_pred_mean = torch.zeros(len(env_xy))
        self.small_pred_std_dev = torch.zeros(len(env_xy))
        self.small_grid_location = torch.tensor((0, 0))
        self.location = torch.tensor((0, 0, 0))
        self.speed = speed
        self.sampling_freq = sampling_freq
        self.small_grid_bins = small_grid_bins

    def reset(self):
        self.mdl = None
        self.current_pred = None
        self.small_pred_mean = torch.zeros(len(self.env_xy))
        self.small_pred_std_dev = torch.zeros(len(self.env_xy))
        self.small_grid_location = torch.tensor((0, 0))
        self.location = torch.tensor((0, 0))
        return
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.fc3.out_features - 1)
        else:
            with torch.no_grad():
                #state = torch.cat((self.small_pred_mean, self.location))
                state = torch.cat((self.small_pred_std_dev, self.location))

                return torch.argmax(self.policy_net(state)).item()
    
    def new_small_grid_location_from_action(self, action):
        if action == 0:
            movement = torch.tensor((1, 0))# right
        elif action == 1:
            movement = torch.tensor((-1, 0))# left
        elif action == 2:
            movement = torch.tensor((0, -1))# down
        elif action == 3:
            movement = torch.tensor((0, 1))# up
        
        return self.small_grid_location + movement

    def print_action(self, action):
        if action == 0:
            print('right')
        elif action == 1:
            print('left')
        elif action == 2:
            print('down')
        elif action == 3:
            print('up')
        else:
            print(f'Action {action} not recognized')
    
        
    def estimate_env(self, env_xy, sampled_coords, sampled_vals):
        print(self.mdl)
        if self.mdl is None:
            self.mdl = ExactGPModel(torch.tensor(sampled_coords), torch.tensor(sampled_vals), self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)
        
        self.mdl.set_train_data(torch.tensor(sampled_coords), torch.tensor(sampled_vals), strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(env_xy))
    
        return current_pred
    
    def compute_reward(self, current_prediction, next_prediction):
        reward = (current_prediction - next_prediction).abs().sum()
        
        return reward

    def make_small_grid(self, coords, pred):
        # Compute small_grid from predictions
        # coords is a 1D torch tensor of (coord_x, coord_y) in the grid
        # pred is a 1D torch tensor of values belonging to the coordinates

        max_x = coords[:, 0].max()
        max_y = coords[:, 1].max()
        bin_width_x = max_x/self.small_grid_bins
        bin_width_y = max_y/self.small_grid_bins
        bin_borders_x = torch.arange(0, max_x + bin_width_x, bin_width_x)
        bin_borders_y = torch.arange(0, max_y + bin_width_y, bin_width_y)
        
        # Initialize the small grid with zeros
        small_pred = torch.zeros((self.small_grid_bins**2), dtype=torch.float32)

        # Fill the small_grid with mean values for each bin,
        # by applying the bin_borders_x and bin_borders_y
        c = 0
        for i in range(self.small_grid_bins):
            for j in range(self.small_grid_bins):
                # Find the bounds for the current bin
                x_min, x_max = bin_borders_x[i], bin_borders_x[i + 1]
                y_min, y_max = bin_borders_y[j], bin_borders_y[j + 1]

                # Identify points within the current bin
                in_bin = (
                    (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
                    (coords[:, 1] >= y_min) & (coords[:, 1] < y_max)
                )

                # Compute the mean value of current_pred within the bin
                if in_bin.any():
                    small_pred[c] = pred[in_bin].mean()
                else:
                    small_pred[c] = 0.0  # Default to 0 if no points fall in the bin
                
                c = c + 1

        return small_pred

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.policy_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

