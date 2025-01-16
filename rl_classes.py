import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import chem_utils
import path

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
        
        return self.env.reset()

    def step(self, action):
        # Move to new location while sampling
        # Action should specify old and new location and speed so that sampling points
        # and reward can be computed
        start_time = '2020-01-01T02:10:00.000000000' # dummy time

        old_loc = action[0]
        new_loc = action[1]
        speed = action[2]
        sampling_freq = action[3]
        # Perhaps make a more flexible function for non-synoptic sampling
        # (although that is much slower)
        synoptic = True

        sample_locs = path.path([old_loc, new_loc], start_time, speed, sampling_freq, synoptic)
        chem_utils.extract_synoptic_chemical_data_from_depth()
        return self.env.step(action)

    def render(self):
        
        self.env.render()

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
class Agent:
    def __init__(self, input_dim, action_space, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim, action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.fc3.out_features - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return torch.argmax(self.policy_net(state)).item()

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

