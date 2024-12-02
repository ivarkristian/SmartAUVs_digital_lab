# %%
import os
import gpytorch.constraints
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
import gpt_class_environment
import gpt_utils

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, help="Index to identify the loop iteration")
parser.add_argument("--directory", type=str, help="Path to output file directory, e.g. \'../out_files\'")
#args = parser.parse_args()
#index = args.index
index = 84
#directory = args.directory
directory = '../out_files/run2_test'

torch.set_default_dtype(torch.float64)

# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# Setup processor usage, parsing of arguments
#torch.set_num_threads = 4

# %%
importlib.reload(path)
importlib.reload(chem_utils)
importlib.reload(path_utils)
importlib.reload(gpt_utils)
importlib.reload(gpt_class_environment)

# %%
# Setup experiment parameters
environment_type = 'elliptical'

# 'netCDF' parameters
sample_variable = 'pH'
#experiment_time_offset = 5 # *10 minutes intervals
synoptic_sampling = True
times = [89]#[4, 81, 85, 89]
depths = [68]#[67, 66, 66, 68]
advection_angles = [-20]#[5, -75, -50, -20] # Inspect files
background_threshold = 0.1
# 'EnvClass' parameters
d = 0.01 
num_c = 50
experiment_start_time = np.datetime64('1970-01-01T13:00:00.000000000')
# common parameters
sample_radius = 1.0
grid_spacings = [10, 20, 40]
rot_dirs = range(0, 91, 5)
grid_types = ['plain', 'cross']
kernel_types = ['SE', 'SE-ARD']
kernel_training_iter = 100
early_stopping = None
# travel speed and sample rate is constant at 1.0 m/s, 1.0 sample/s

# %%
# Define size and resolution of environment
scenario_x_min = 0
scenario_x_max = 250
scenario_y_min = 0
scenario_y_max = 250

sc_len_x = scenario_x_max - scenario_x_min
sc_len_y = scenario_y_max - scenario_y_min
path_len = sc_len_x/2.0 * np.sqrt(2.0)
path_delta = (sc_len_x - path_len)/2.0

path_x_min = scenario_x_min + path_delta
path_x_max = scenario_x_max - path_delta
path_y_min = scenario_y_min + path_delta
path_y_max = scenario_y_max - path_delta

resolution = 3000
x_data = np.linspace(path_x_min, path_x_max, resolution, dtype=np.float64)
y_data = np.linspace(path_y_min, path_y_max, resolution, dtype=np.float64)

sc_mean_x = scenario_x_min + sc_len_x/2.0
sc_mean_y = scenario_y_min + sc_len_y/2.0

# %%
if environment_type == 'netCDF':
    # Load netCDF data set from path
    data_dir = '../scenario_1c_medium'
    # Read and clean list of .nc files
    files = os.listdir(data_dir)
    # Create a new list with strings that end with '.nc'
    nc_files = [s for s in files if s.endswith('.nc')]
    nc_files.sort()
    print(f'Files of type .nc:\n{nc_files}')
    env_list = []
    env_desc = []
    for i, time in enumerate(times):

        file_num = int(time/12)
        experiment_time_offset = time%12
        data_file = os.path.join(data_dir, nc_files[file_num])

        dataset = chem_utils.load_chemical_dataset(data_file)
        dataset_start_time = dataset['time'].values[0]
        experiment_start_time = dataset['time'].values[0 + experiment_time_offset]
        print(f'Loaded dataset from {data_file}')
        print(f'File starts at {dataset_start_time}, experiment starts at {experiment_start_time}')

        val_dataset = dataset[sample_variable].isel(time=experiment_time_offset, siglay=depths[i])

        x_adj = val_dataset['x'].values[:72710] - min(val_dataset['x'].values[:72710])
        y_adj = val_dataset['y'].values[:72710] - min(val_dataset['y'].values[:72710])
        env_xy = torch.tensor(np.column_stack((x_adj, y_adj)), dtype=torch.float64)
        x = torch.tensor(x_adj, dtype=torch.float64)
        y = torch.tensor(y_adj, dtype=torch.float64)
        env_list.append(val_dataset.values[:72710])
        env_desc.append(f'file: {data_file} time step: {experiment_time_offset} depth: {depths[i]}')
        
        # plot netCDF data set
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x, y, c=env_list[i], cmap='coolwarm', s=2)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Value')

        # Add labels and title
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title(f'TS {file_num*12 + experiment_time_offset}, {sample_variable} at {depths[i]}m depth')

        plt.show()


elif environment_type == 'elliptical':
    advection_angle = 0
    
    # Create environment x and y locations
    env_x = np.linspace(scenario_x_min, scenario_x_max, sc_len_x, dtype=np.float64)
    env_y = np.linspace(scenario_y_min, scenario_y_max, sc_len_y, dtype=np.float64)
    env_xy = torch.tensor(np.column_stack((env_x, env_y)), dtype=torch.float64)
    env_xy = gpytorch.utils.grid.create_data_from_grid(env_xy)
    x = env_xy[:, 0]
    y = env_xy[:, 1]

    # Define c parameter (determines anisotropy)
    rng_c = np.random.default_rng(seed=index)
    env_list = rng_c.integers(low=1, high=2000, endpoint=True, size=num_c)/10.0  # Generate envs based on these
    #env_list = [1.0, 180.0]
    env_desc = [str(c) for c in env_list]

    depths = np.ones_like(env_list)

    # Define x and y offset of location
    max_spacings_half = max(grid_spacings)/2.0
    rng_x_offset = np.random.default_rng(seed=index*3)
    rng_y_offset = np.random.default_rng(seed=index**2)
    x_offset_list = rng_x_offset.integers(low=-max_spacings_half, high=max_spacings_half, size=len(env_list)) # Generate envs based on these
    y_offset_list = rng_y_offset.integers(low=-max_spacings_half, high=max_spacings_half, size=len(env_list)) # Generate envs based on these
    #x_offset_list = [0, 0]
    #y_offset_list = [0, 0]


# %%
# Run experiments
print(f'Experiment parameters:')
print(f'num_envs = {len(env_list)}')
print(f'spacings = {grid_spacings}')
print(f'rotations = {rot_dirs}')
print(f'Time is ', end='')
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
file_ts = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
fname = file_ts + '_' + str(index) + '_output.pickle'
fname_env = fname.split('.')[0] + '_environments.pickle'
fname_pred = fname.split('.')[0] + '_predictions.pickle'

output_dir = directory
output_dir_figs = directory + '/figures'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_figs, exist_ok=True)

file_path = os.path.join(output_dir, fname)
file_path_env = os.path.join(output_dir_figs, fname_env)
file_path_pred = os.path.join(output_dir_figs, fname_pred)
print(f'Starting experiment, writing to file \'{file_path}\'')

# %%
with open(file_path, 'wb') as file_out, open(file_path_env, 'wb') as file_out_env, open(file_path_pred, 'wb') as file_out_pred:
    for i, c in enumerate(env_list):

        if environment_type == 'netCDF':
            env_values = gpt_utils.normalize_tensor(torch.tensor(c, dtype=torch.float64))
            advection_angle = advection_angles[i]
            x_offset = 0
            y_offset = 0
            if sample_variable == 'pH':
                env_values = 1 - env_values
            title = f'Simulated feature: {env_desc[i]}'
                
        if environment_type == 'elliptical':
            x_offset = x_offset_list[i]
            y_offset = y_offset_list[i]
            rel_dists = env_xy - torch.tensor([sc_mean_x - 25 + x_offset, sc_mean_y + y_offset])
            distances = torch.norm(rel_dists, dim=1) # scale with dilution
            angles = torch.atan2(rel_dists[:, 1], rel_dists[:, 0]) # [-pi, pi]
    
            angles_cos = (-torch.cos(angles) + 1)*c
            angles_cos += 1
            env_values = gpt_utils.normalize_tensor(torch.exp(-(distances*d*(angles_cos))))
            title = f'Elliptical feature (c = {env_desc[i]}, x_offset = {x_offset}, y_offset = {y_offset})'
        
        print(f'Running environment - {title} ({i}/{len(env_list)})')
        # Create plots of environment
        fig = gpt_utils.plot_env(x, y, env_values, vmin=0.0, vmax=1.0, title=title)
        pickle.dump(fig, file_out_env)

        for j, spacing in enumerate(grid_spacings):
            print(f'Grid spacing: {spacing}...')

            # Generate waypoints for the lawnmower path with specified parameters 
            waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
                x_data, y_data, width=spacing, min_turn_radius=int(spacing/2), siglay=depths[i], direction='x'
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

            #x_min = x_mean - (x_mean - np.min(wp_x))*np.sqrt(2.0)
            #y_min = y_mean - (y_mean - np.min(wp_y))*np.sqrt(2.0)
            #x_max = x_mean + (np.max(wp_x) - x_mean)*np.sqrt(2.0)
            #y_max = y_mean + (np.max(wp_y) - y_mean)*np.sqrt(2.0)
            
            ts = experiment_start_time
            speed = 1.5
            sample_freq = 1
            plain_sample_coords_list = path.path(waypoints, ts, speed, sample_freq, synoptic=synoptic_sampling)

            # Prepare for cross grid_type
            cross_waypoints_with_turns, cross_x_coords, cross_y_coords, cross_z_coords = lp.generate_lawnmower_waypoints(
                        x_data, y_data, width=spacing, min_turn_radius=int(spacing/2), siglay=depths[i], direction='y'
            )
            
            # remove duplicate waypoints
            cross_waypoints = lp.remove_consecutive_duplicate_wps(cross_waypoints_with_turns, 1e-3)
            cross_sample_coords_list = path.path(cross_waypoints, ts, speed, sample_freq, synoptic=synoptic_sampling)
            
            for k, rot_dir in enumerate(rot_dirs):
                
                for grid_type in grid_types:
                    if grid_type == 'cross':        
                        sample_coords_list = plain_sample_coords_list + cross_sample_coords_list
                    else:
                        sample_coords_list = plain_sample_coords_list

                    # Convert the sample_coords list of tuples to a NumPy array
                    sample_coords_no_rot = np.array([tup[:2] for tup in sample_coords_list])

                    xy_to_rotate = torch.tensor(sample_coords_no_rot[:, :2] - [x_mean, y_mean], dtype=torch.float64) 
                    xy_rot = path_utils.rotate_points(xy_to_rotate, rot_dir) + torch.tensor([x_mean, y_mean], dtype=torch.float64)

                    sample_values = torch.zeros(len(xy_rot), dtype=torch.float64)
                    # sample waypoints_rot
                    for m, coord in enumerate(xy_rot):
                        sample_values[m] = chem_utils.extract_synoptic_chemical_data_from_depth(x, y, env_values, coord, sample_radius)

                    for kernel_type in kernel_types:
                        
                        if kernel_type == 'SE-ARD':
                            # Rotate system for ard kernel
                            xy_to_rot_ard = xy_rot - torch.tensor([x_mean, y_mean])
                            xy_to_train = path_utils.rotate_points(xy_to_rot_ard, advection_angle) + torch.tensor([x_mean, y_mean])
                            x_mean_env = scenario_x_min + (scenario_x_max - scenario_x_min)/2.0
                            y_mean_env = scenario_y_min + (scenario_y_max - scenario_y_min)/2.0
                            env_xy_to_rot_ard = env_xy - torch.tensor([x_mean_env, y_mean_env])
                            xy_to_predict = path_utils.rotate_points(env_xy_to_rot_ard, advection_angle) + torch.tensor([x_mean_env, y_mean_env])
                            kernel_name = 'scale_rbf_ard'
                        else:
                            xy_to_train = xy_rot
                            xy_to_predict = env_xy
                            kernel_name = 'scale_rbf'
                    
                        ct_train = CodeTimer('Train', unit='s')
                        ct_predict = CodeTimer('Predict', unit='s')
                        
                        llh = gpytorch.likelihoods.GaussianLikelihood()
                        #length_constraint = gpytorch.constraints.GreaterThan(spacing)
                        length_constraint = gpytorch.constraints.Positive()
                        mdl = ExactGPModel(xy_to_train, sample_values, llh, kernel_name, lengthscale_constraint=length_constraint)
                        
                        with ct_train:
                            gpt_utils.train_model(xy_to_train, sample_values, mdl, iter=kernel_training_iter, early_delta=(early_stopping, 'mll', None, None, 10), debug=False)

                        # Then predict
                        mdl.eval()
                        mdl.likelihood.eval()

                        with ct_predict, torch.no_grad(), gpytorch.settings.fast_pred_var():
                            pred = mdl.likelihood(mdl(xy_to_predict))
                        
                        
                        # Compare single_pred with env_values
                        xy_to_predict_to_rot = xy_to_predict - torch.tensor([x_mean, y_mean])
                        xy_to_predict_rotated = path_utils.rotate_points(xy_to_predict_to_rot, -rot_dir) + torch.tensor([x_mean, y_mean])
                        indices_low_x = xy_to_predict_rotated[:, 0] >= wp_x_min
                        indices_high_x = xy_to_predict_rotated[:, 0] <= wp_x_max
                        indices_low_y = xy_to_predict_rotated[:, 1] >= wp_y_min
                        indices_high_y = xy_to_predict_rotated[:, 1] <= wp_y_max
                        ind = indices_low_x & indices_high_x & indices_low_y & indices_high_y
                        
                        values_diff = env_values[ind] - pred.mean[ind]
                        RMSE = values_diff.pow(2).mean().sqrt()
                        
                        # Number of samples above threshold
                        num_above_threshold = (sample_values >= background_threshold).int().sum()
                        num_above_threshold_env = (env_values[ind] >= background_threshold).int().sum()

                        # Create plots of prediction
                        title = f'Prediction (grid {spacing}, rot. {rot_dir}, {kernel_type}) RMSE: {RMSE:.4}'
                        fig = gpt_utils.plot_env(x, y, pred.mean, xy_rot, vmin=0.0, vmax=1.0, title=title)
                        pickle.dump(fig, file_out_pred)

                        data_dict = {
                            'index': index,
                            'env_desc': env_desc[i],
                            'x_offset': x_offset,
                            'y_offset': y_offset,
                            'env_num': i,
                            'spacing_num': j,
                            'rot_num': k,
                            'spacing': spacing,
                            'rot': rot_dir,
                            'angle_delta': advection_angle - rot_dir,
                            'ct_train': ct_train.took,
                            'ct_predict': ct_predict.took,
                            'kernel_type': kernel_type,
                            'grid_type' : grid_type,
                            'RMSE': RMSE.item(),
                            'plume_samples': num_above_threshold.item(),
                            'plume_fraction': (num_above_threshold/num_above_threshold_env).item()
                        }

                        pickle.dump(data_dict, file_out)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Done!')
print(f'Experiment parameters:')
print(f'num_envs = {len(env_list)}')
print(f'spacings = {grid_spacings}')
print(f'rotations = {rot_dirs}')
print(f'Wrote to file \'{fname}\'')

# %%
