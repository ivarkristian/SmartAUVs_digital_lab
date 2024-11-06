# %%
import os
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from datetime import datetime
from linetimer import CodeTimer
import importlib
import lawnmower_path as lp
import path_utils
import chem_utils
import path
from gpt_class_exactgpmodel import ExactGPModel
import gpt_utils

# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# Setup processor usage, parsing of arguments
#torch.set_num_threads = 4
# Add argument parser to receive index
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, help="Index to identify the loop iteration")
#args = parser.parse_args()
#index = args.index
index = 1

# %%
importlib.reload(path)
importlib.reload(chem_utils)

# %%
# Setup experiment parameters
sample_variable = 'pH'
sample_radius = 1.0
n_per_spacing = 5
grid_spacings = [10, 30]
kernel_type = 'scale_rbf_ard'
kernel_training_iter = 100
early_stopping = None
experiment_time_offset = 4 # *10 minutes intervals
synoptic_sampling = True
depth = 67
advection_angle = 0
background_threshold = 0.1
# rotation is random from uniform probability [-180, 180]
# travel speed and sample rate is constant at 1.0 m/s, 1.0 sample/s

# %%
# Load data set from path
data_dir = '../scenario_1c_medium/'
# Read and clean list of .nc files
files = os.listdir(data_dir)
# Create a new list with strings that end with '.nc'
nc_files = [s for s in files if s.endswith('.nc')]
nc_files.sort()
print(f'Files of type .nc:\n{nc_files}')

data_file = data_dir + nc_files[0]
dataset = chem_utils.load_chemical_dataset(data_file)
dataset_start_time = dataset['time'].values[0]
experiment_start_time = dataset['time'].values[0 + experiment_time_offset]
print(f'Loaded dataset from {data_file}')
print(f'Dataset starts at {dataset_start_time}, experiment starts at {ts}')

# %%
# Define size and resolution of environment
scenario_x_min = 0
scenario_x_max = 250
scenario_y_min = 0
scenario_y_max = 250

sc_len = scenario_x_max - scenario_x_min
path_len = sc_len/2.0 * np.sqrt(2.0)
path_delta = (sc_len - path_len)/2.0

path_x_min = scenario_x_min + path_delta
path_x_max = scenario_x_max - path_delta
path_y_min = scenario_y_min + path_delta
path_y_max = scenario_y_max - path_delta

resolution = 3000
x_data = np.linspace(path_x_min, path_x_max, resolution)
y_data = np.linspace(path_y_min, path_y_max, resolution)

val_dataset = dataset[sample_variable].isel(time=experiment_time_offset, siglay=depth)
env_values = gpt_utils.normalize_tensor(torch.tensor(val_dataset.values[:72710]))
if sample_variable == 'pH':
    env_values = 1 - env_values

num_above_threshold_env = (env_values > background_threshold).int().sum()

x_np = val_dataset['x'].values[:72710] - min(val_dataset['x'].values[:72710])
y_np = val_dataset['y'].values[:72710] - min(val_dataset['y'].values[:72710])
x = torch.from_numpy(x_np)
y = torch.from_numpy(y_np)
xy_to_predict = torch.from_numpy(np.column_stack((x, y)))

# %%
# Run experiments
print(f'Experiment parameters:')
print(f'spacings = {grid_spacings}')
print(f'n_per_spacing = {n_per_spacing}')
print(f'Time is ', end='')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
file_ts = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
fname = file_ts + '_' + str(index) + '_output.pickle'
print(f'Starting experiment, writing to file \'{fname}\'')

# %%
with open(fname, 'wb') as file_out:
    for i, spacing in enumerate(grid_spacings):

        print(f'Grid spacing: {spacing}...')
        # Generate waypoints for the lawnmower path with specified parameters 
        waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
            x_data, y_data, width=10, min_turn_radius=5, siglay=depth, direction='x'
        )

        # remove duplicate waypoints
        waypoints = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)

        # Compute means, min and max for x and y axes
        wp_x = waypoints[:, 0]
        wp_y = waypoints[:, 1]

        wp_x_min, wp_x_max = np.min(wp_x), np.max(wp_x)
        wp_y_min, wp_y_max = np.min(wp_y), np.max(wp_y)

        x_mean = wp_x_min + (wp_x_max - wp_x_min)/2.0
        y_mean = wp_y_min + (wp_y_max - wp_y_min)/2.0

        x_min = x_mean - (x_mean - np.min(wp_x))*np.sqrt(2.0)
        y_min = y_mean - (y_mean - np.min(wp_y))*np.sqrt(2.0)
        x_max = x_mean + (np.max(wp_x) - x_mean)*np.sqrt(2.0)
        y_max = y_mean + (np.max(wp_y) - y_mean)*np.sqrt(2.0)
        
        # Extract using interpolating extract function path.path()
        ts = experiment_start_time
        speed = 1.5
        sample_freq = 1
        measurements, sample_coords_list = path.path(dataset, waypoints, ts, speed, sample_freq, data_variable='pH', synoptic=synoptic_sampling)

        # Convert the sample_coords list of tuples to a NumPy array
        sample_coords_no_rot = np.array([tup[:2] for tup in sample_coords_list])

        rng = np.random.default_rng()
        rot_dirs = rng.integers(low=-179, high=180, endpoint=True, size=n_per_spacing)  # [-180 180]*n

        xy_to_rotate = sample_coords_no_rot[:, :2] - [x_mean, y_mean]
        for j, rot_dir in enumerate(rot_dirs):
            xy_rot = path_utils.rotate_points(xy_to_rotate, rot_dir) + [x_mean, y_mean]

            sample_values = torch.zeros(len(xy_rot))
            # sample waypoints_rot
            for k, coord in enumerate(xy_rot):
                sample_values[k] = chem_utils.extract_synoptic_chemical_data_from_depth(x, y, env_values, coord, sample_radius)

            ct_train = CodeTimer('Train', unit='s')
            ct_predict = CodeTimer('Predict', unit='s')
            # Demonstrate training with early stopping on a single kernel
            llh = gpytorch.likelihoods.GaussianLikelihood()
            mdl = ExactGPModel(torch.from_numpy(xy_rot), sample_values, llh, kernel_type)
            #mdl.mean_module.constant.data.fill_(ch4_training.min().item())
            with ct_train:
                gpt_utils.train_model(torch.from_numpy(xy_rot), sample_values, mdl, iter=kernel_training_iter, early_delta=(early_stopping, 'mll', None, None, 10), debug=False)

            # Then predict
            mdl.eval()
            mdl.likelihood.eval()

            with ct_predict, torch.no_grad(), gpytorch.settings.fast_pred_var():
                single_pred = mdl.likelihood(mdl(xy_to_predict))
            
            # Compare single_pred with env_values
            values_diff = env_values - single_pred.mean
            RMSE = values_diff.pow(2).mean().sqrt()

            # Number of samples above threshold
            num_above_threshold = (sample_values > background_threshold).int().sum()

            dict = {    
                'spacing_num': i,
                'rot_num': j,
                'spacing': spacing,
                'rot': rot_dir,
                'angle_delta': advection_angle - rot_dir,
                'ct_train': ct_train.took,
                'ct_predict': ct_predict.took,
                'RMSE': RMSE.item(),
                'plume_samples': num_above_threshold,
                'plume_percentage': num_above_threshold/num_above_threshold_env
            }
                
            #experiment.append(dict)
            pickle.dump(dict, file_out)

# %%
