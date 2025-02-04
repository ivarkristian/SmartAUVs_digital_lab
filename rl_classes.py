import os
import gpytorch.constraints
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

    def plot_env(self, title_postfix=None, path=None):
        if self.parameter:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(self.x, self.y, c=self.val, cmap='coolwarm', s=2, vmin=self.val.min(), vmax=self.val.max())
            if path:
                ax.scatter(self.sampled_coords[:, 0], self.sampled_coords[:, 1], c='black', s=2)
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Value')

            # Add labels and title
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.set_title(f'Time {self.time}, {self.parameter} at {self.depth}m depth ({title_postfix})')

            return fig, ax
        
        print(f'Could not plot dataset = {self.dataset}, parameter = {self.parameter}')


    def reset(self):
        self.sampled_coords = torch.tensor([])
        self.sampled_vals = torch.tensor([])
        return
    
    def append_z_to_xy(self, xy):
        if len(xy) == 2:
            return torch.cat((xy, torch.tensor([self.depth])))
        else:
            return xy

    def step(self, old_loc, new_loc, speed, sampling_freq):
        # Move to new location while sampling
        # Action should specify old and new location and speed so that sampling points
        # and reward can be computed
        start_time = '2020-01-01T02:10:00.000000000' # dummy time

        # Perhaps make a more flexible function for non-synoptic sampling
        # (although that is much slower)
        synoptic = True
        old_loc = self.append_z_to_xy(old_loc)
        new_loc = self.append_z_to_xy(new_loc)
        old_loc_cpu = old_loc.cpu()
        new_loc_cpu = new_loc.cpu()
        #print(f'old_loc: {old_loc}, new_loc: {new_loc}')
        sample_coords = path.path([old_loc_cpu, new_loc_cpu], start_time, speed, sampling_freq, synoptic)
        # Extract the first two elements of each tuple and convert to a torch tensor
        sample_coords_xy = torch.tensor([(float(t[0]), float(t[1])) for t in sample_coords])

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
    def __init__(self, input_dim, action_space, env_xy, speed, sampling_freq, small_grid_bins=5, learning_rate=1e-3, gamma=0.99, gp_lengthscale_constraint=gpytorch.constraints.Positive(), nn_filename=None, device='mps'):
        # Init the static parameters
        self.device = torch.device(device)
        self.nn_filename = nn_filename
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.policy_net = PolicyNetwork(input_dim, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        if self.nn_filename:
            self.load_model(self.nn_filename)

        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        #self.lengthscale_initial = gp_lengthscale_initial
        self.length_constraint = gp_lengthscale_constraint
        self.kernel_name = 'scale_rbf'
        self.env_xy = env_xy
        self.small_grid_bins = small_grid_bins
        self.speed = speed
        self.sampling_freq = sampling_freq
        # Init the parameters that resets for every episode
        self.reset()

    def reset(self):
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)
        self.current_pred_mean = torch.zeros(len(self.env_xy))
        self.current_pred_variance = torch.ones(len(self.env_xy))*5
        
        self.small_grid_mean = torch.zeros(self.small_grid_bins**2, device=self.device)
        self.small_grid_variance = torch.zeros(self.small_grid_bins**2, device=self.device)
        self.small_grid_location = torch.tensor((0, 0), device=self.device)
        self.location = torch.tensor((0, 0, 0), device=self.device)
        return
    
    def save_model(self, filename=None):
        """Save the model weights and optimizer state to a file."""
        if filename is None:
            filename = 'GPAgent.nn'
        
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename=None):
        """Load model weights and optimizer state from a file."""
        try:
            checkpoint = torch.load(filename, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}, starting fresh.")
        except Exception as e:
            print(f'Error loading model: {e}')

    def select_action(self, state, epsilon=0.1, train_mode=True):
        # state parameter not used, but should be in the future for a more flexible function
        if train_mode and (random.random() < epsilon):
            return random.randint(0, self.policy_net.fc3.out_features - 1)
        else:
            with torch.no_grad():
                #state = torch.cat((self.small_grid_mean, self.small_grid_location))
                state = torch.cat((self.small_grid_variance, self.small_grid_location))

                return torch.argmax(self.policy_net(state)).item()
    
    def new_small_grid_location_from_action(self, action):
        if action == 0:
            movement = torch.tensor((1, 0), device=self.device)# right
        elif action == 1:
            movement = torch.tensor((-1, 0), device=self.device)# left
        elif action == 2:
            movement = torch.tensor((0, -1), device=self.device)# down
        elif action == 3:
            movement = torch.tensor((0, 1), device=self.device)# up
        
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
    
    def print_grid(self, grid):
        # Ensure vector length is a perfect square
        n = grid.numel()  # Get the total number of elements in the tensor
        sqrt_n = int(torch.sqrt(torch.tensor(n, dtype=torch.float32)))
        if sqrt_n ** 2 != n:
            raise ValueError("The length of the vector must be a perfect square.")

        # Reshape into a 2D grid
        grid = grid.view(sqrt_n, sqrt_n)

        # Reverse the rows to make the first sqrt(n) values the bottom row
        grid = torch.flip(grid, dims=[0])

        # Print the grid
        for row in grid:
            print(" ".join(f"{v:.{4}f}" for v in row.tolist()))

        
    def estimate_env(self, env_xy, sampled_coords, sampled_vals):
        if self.mdl is None:
            self.mdl = ExactGPModel(sampled_coords, sampled_vals, self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)
        
        self.mdl.set_train_data(sampled_coords, sampled_vals, strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(env_xy))
    
        return current_pred
    
    def plot_estimate(self, env_xy, val, path=None, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(env_xy[:, 0], env_xy[:, 1], c=val, cmap='coolwarm', s=2, vmin=val.min(), vmax=val.max())
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Value')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        if title:
            ax.set_title(title)

        return fig, ax

    def compute_reward(self, current_prediction, next_prediction):
        total_change = (current_prediction - next_prediction).abs().sum()
        reward = total_change
        
        return reward

    def make_small_grid(self, coords_cpu, pred_cpu):
        # Compute small_grid from predictions
        # coords is a 1D torch tensor of (coord_x, coord_y) in the grid
        # pred is a 1D torch tensor of values belonging to the coordinates
        coords = coords_cpu.to(self.device)
        pred = pred_cpu.to(self.device)

        max_x = coords[:, 0].max()
        max_y = coords[:, 1].max()
        bin_width_x = max_x/self.small_grid_bins
        bin_width_y = max_y/self.small_grid_bins
        bin_borders_x = torch.arange(0, max_x + bin_width_x, bin_width_x, device=self.device)
        bin_borders_y = torch.arange(0, max_y + bin_width_y, bin_width_y, device=self.device)
        
        # Initialize the small grid with zeros
        small_pred = torch.zeros((self.small_grid_bins**2), dtype=torch.float32, device=self.device)

        # Fill the small_grid with mean values for each bin,
        # by applying the bin_borders_x and bin_borders_y
        c = 0
        for i in range(self.small_grid_bins):
            for j in range(self.small_grid_bins):
                # Find the bounds for the current bin
                x_min, x_max = bin_borders_x[j], bin_borders_x[j + 1]
                y_min, y_max = bin_borders_y[i], bin_borders_y[i + 1]

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
    
    def normalize(self, tensor):
        if tensor.min() == tensor.max():
            return torch.zeros_like(tensor)
        
        zero_adj = tensor - tensor.min()
        return zero_adj/zero_adj.max()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack([state.to(self.device) for state in states]),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.stack([next_state.to(self.device) for next_state in next_states]),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def train(self, training_per_episode=1):
        for _ in range(training_per_episode):
            if len(self.memory) < self.batch_size:
                return

            states, actions, rewards, next_states, dones = self.sample_memory()

            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze().to(self.device)
            next_q = self.policy_net(next_states).max(1)[0].detach().to(self.device)
            target_q = rewards + self.gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

