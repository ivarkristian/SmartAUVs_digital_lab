# %%
import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np

import chem_utils

# %%
# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# Step 1: Define the environment wrapper
class ScenarioBank:
    def __init__(self, data_dir = '../scenario_1c_medium/'):
        self.data_dir = data_dir
        self.data_file = None
        self.dataset = None
        #self.parameter = None
        #self.depth = None
        #self.time = None
        #self.val = None
        #self.x = None
        #self.y = None
        #self.sampled_coords = torch.tensor([])
        #self.sampled_vals = torch.tensor([])
        self.environments = []

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
    
    def get_env(self, parameter='pH', depth=67, time=1):
        self.parameter = parameter
        self.depth = depth
        self.time = time
        val_dataset = self.dataset[self.parameter].isel(time=self.time, siglay=self.depth)
        values = val_dataset.values[:72710]
        x_dataset = val_dataset['x'].values[:72710]
        y_dataset = val_dataset['y'].values[:72710]
        x = x_dataset - x_dataset.min()
        y = y_dataset - y_dataset.min()
        env_xy = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
        metadata = {'parameter': parameter, 'depth': depth, 'time': time, 'data_file': self.data_file}

        return env_xy, values, metadata
    
    def add_env(self, parameter='pH', depth=67, time=1):
        env_xy, values, metadata = self.get_env(parameter, depth, time)
        self.environments.append({'coords': env_xy, 'values': values, 'parameter': metadata['parameter'], 'depth': metadata['depth'], 'time': metadata['time']})
        print(f"Loaded environment: {parameter} (depth={depth}, time={time})")
        
    def print_info(self):
        print(f'Current directory: {self.data_dir}')
        print(f'Data file: {self.data_file}')
        print(f'Loaded dataset: {self.dataset}')
        self.print_env_info()
    
    def print_envs_info(self):
        print(f"Loaded environments:")
        for env in self.environments:
            print(f"{env['parameter']} (depth={env['depth']}, time={env['time']})")

    def plot_env(self, env_num=0, title_postfix=None, path=None):
        if len(self.environments) <= env_num:
            print(f'Bank contains only {len(self.environments)}. Tried to plot #{env_num}')
            return
        
        env = self.environments[env_num]
        if env['parameter']:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(env['coords'][:, 0], env['coords'][:, 1], c=env['values'], cmap='coolwarm', s=2, vmin=env['values'].min(), vmax=env['values'].max())
            if path:
                ax.scatter(path[:, 0], path[:, 1], c='black', s=2)
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Value')

            # Add labels and title
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Northing [m]')
            ax.set_title(f"Time {env['time']}, {env['parameter']} at {env['depth']}m depth ({title_postfix})")

            return fig, ax
        
        print(f"Could not plot dataset = {self.dataset}, parameter = {env['parameter']}")

    def sample(self):
        return random.choice(self.environments)

    def get_minmax(self):
        maxes = torch.zeros(len(self.environments))
        mins = torch.zeros(len(self.environments))
        for c, env in enumerate(self.environments):
            maxes[c] = env['values'].max()
            mins[c] = env['values'].min()
        
        return mins.min(), maxes.max()
    
    def get_mu_sigma2(self):
        ns = torch.zeros(len(self.environments))
        mus = torch.zeros(len(self.environments))
        sigma2s = torch.zeros(len(self.environments))
        for c, env in enumerate(self.environments):
            ns[c] = len(env['values'])
            mus[c] = env['values'].mean()
            sigma2s[c] = env['values'].var()
        
        N = ns.sum()
        mu_all = (ns * mus).sum() / N
        ss = ns * (sigma2s + mus**2)
        sigma2_all = ss.sum() / N - mu_all**2

        return mu_all, sigma2_all
    
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

