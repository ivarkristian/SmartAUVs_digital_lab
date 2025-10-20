# The purpose of this script is to support paper 4 - investigating simulated patterns
# and application in field study.

# In particular, this script runs tests that compare different preplanned patterns
# for focused sampling near points of interest. We plan to compare patterns that
# a) does not take into account currents - double bowtie, dubins square, spiral
# b) patterns that do take the current into account - tilted bowtie, drifting circle

# The aim is to find a best practice for known and unknown current, and compare with
# field measurements.

# %%
import importlib
import torch
import numpy as np
import random
import lawnmower_path as lp
import rl_scenario_bank
import path
import chem_utils
import rl_gas_survey_dubins_env
import analysis_utils
import path_utils
from patterns import *

# %%
importlib.reload(rl_scenario_bank)
importlib.reload(path)
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(analysis_utils)
importlib.reload(path_utils)

# %%
# Device selection supporting CUDA, MPS (Apple Silicon), or CPU
device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')
envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
random_scenario = bank.sample()
rotation = random.choice([-90, 0, 90, 180])
env_xy = rl_gas_survey_dubins_env.rotate_xy(random_scenario['coords'].to(device), rotation)
values = random_scenario['values'].to(device)

# Offset so that source is not always in the middle
max_x_off = int(env_xy[:, 0].max()/2 * 0.7)
max_y_off = int(env_xy[:, 1].max()/2 * 0.7)
x_off = random.randint(-max_x_off, max_x_off)
y_off = random.randint(-max_y_off, max_y_off)

x_max = env_xy[:, 0].max()
y_max = env_xy[:, 1].max()
x_min = env_xy[:, 0].min()
y_min = env_xy[:, 1].min()

env_xy[:, 0] += x_off
env_xy[:, 1] += y_off

values[env_xy[:, 0] > x_max] = values.min()
values[env_xy[:, 0] < x_min] = values.min()
values[env_xy[:, 1] > y_max] = values.min()
values[env_xy[:, 1] < y_min] = values.min()
env_xy[:, 0][env_xy[:, 0] > x_max] -= x_max
env_xy[:, 0][env_xy[:, 0] < x_min] += x_max
env_xy[:, 1][env_xy[:, 1] > y_max] -= y_max
env_xy[:, 1][env_xy[:, 1] < y_min] += y_max

# Get numpy copies of environment for later sampling
env_xy_np = env_xy.cpu().numpy()
values_np = values.cpu().numpy()

# %%
# Define sizes for lawnmower pattern
scenario_x_min = env_xy_np[:, 0].min()
scenario_x_max = env_xy_np[:, 0].max()
scenario_y_min = env_xy_np[:, 1].min()
scenario_y_max = env_xy_np[:, 1].max()

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

#sc_mean_x = scenario_x_min + sc_len_x/2.0
#sc_mean_y = scenario_y_min + sc_len_y/2.0

# %%
# Example simulation parameters
sample_radius = 2.0
line_spacing = 25
speed = 1.0  # m/s
data_var = 'pCO2'
depth = 68
start_time = '2020-01-01T02:10:00.000000000'
sample_freq = 1.0  # Sample frequency in Hz (samples per second)
threshold = 550  # Threshold for the chemical data variable
# We can compare the following patterns:
#[bowtie, four-leaf (double bowtie), severdighet (dubins square), spiral, drifting circle]
# (If we have no info about the current, compare four-leaf, severdighet and spiral)
# (If we use current direction info, compare tilted bowtie and drifting circle)
#pattern_funcs = [bowtie, cross, crisscross, drifting_circle, square, leaf_clover, spiral]
pattern_funcs = [bowtie_double, square_dubins]
#pattern_func = bowtie
trigger_dist = 100

# %%
# Generate waypoints
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=line_spacing, min_turn_radius=25, siglay=depth, direction='x')

# remove duplicate waypoints
waypoints_with_turns = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)


plain_sample_coords_times_list = path.path(waypoints_with_turns, start_time, speed, sample_freq, synoptic=True)
sample_coords_xy = [row[:2] for row in plain_sample_coords_times_list]

# %%
for pattern_func in pattern_funcs:
    print(f'Pattern: {pattern_func.__name__}')

    # Simulate AUV path and collect data
    measurements = []
    measurement_coords = []
    sample_coords_xy_pattern_coords = []
    triggered_locs = []
    triggered_patterns = []

    #new_trigger = False
    #c = 0
    for coord in sample_coords_xy:
        msr = chem_utils.extract_synoptic_chemical_data_from_depth(env_xy_np[:, 0], env_xy_np[:, 1], values_np, coord, sample_radius)
        measurements.append(msr)
        measurement_coords.append(coord)
        #c = c+1

        # Check if the chemical data exceeds the threshold and sample in a pattern if true
        if msr > threshold:
            # Check if leakage region is new
            new_trigger = True # assume new trigger
            for (loc_x, loc_y) in triggered_locs:
                dx = coord[0] - loc_x
                dy = coord[1] - loc_y
                if dx*dx + dy*dy <= trigger_dist*trigger_dist:
                    new_trigger = False # remove trigger if too close
                    break
            
            if new_trigger:
                #new_trigger = False
                triggered_locs.append(coord)
                pattern_waypoints = pattern_func(np.array([coord[0], coord[1], depth]))
                pattern_wp_xy = np.array([(row[0], row[1]) for row in pattern_waypoints])
                
                # rotate pattern accourding to heading
                heading = np.atan2((coord[1] - coord_prev[1]),(coord[0] - coord_prev[0]))
                pattern_wp_rotated = path_utils.rotate_points(pattern_wp_xy, np.rad2deg(heading), rot_coord=coord)
                pattern_wp_xyz = np.array([(row[0], row[1], depth) for row in pattern_wp_rotated])

                pattern_sample_coords_times_list = path.path(pattern_wp_xyz, start_time, speed, sample_freq, synoptic=True)
                sample_coords_xy_pattern = [(row[0], row[1]) for row in pattern_sample_coords_times_list]
                #measurements = np.pad(measurements, (0, len(sample_coords_xy_pattern)), mode='constant', constant_values=0)

                #msrs_tmp = np.zeros(len(sample_coords_xy_pattern))
                #msr_locs_tmp = np.zeros_like(msrs_tmp)
                for i, pattern_coord in enumerate(sample_coords_xy_pattern):
                    msr = chem_utils.extract_synoptic_chemical_data_from_depth(env_xy_np[:, 0], env_xy_np[:, 1], values_np, pattern_coord, sample_radius)
                    measurements.append(msr)
                    measurement_coords.append(pattern_coord)
                
                #triggered_patterns.append(sample_coords_xy_pattern)
        
        coord_prev = coord

    #pattern_xy = [x for sublist in triggered_patterns for x in sublist]
    #sample_coords_xy_total = torch.tensor(sample_coords_xy + pattern_xy)
    measurements = torch.tensor(measurements)

    
    # Plot the final path
    title = f"{pattern_func.__name__}, {random_scenario['parameter']} at -{random_scenario['depth']}m. ({random_scenario['cur_str']:.2}m/s @ {round(random_scenario['cur_dir'])} deg)"
    analysis_utils.plot_env(x=env_xy[:, 0], y=env_xy[:, 1], c=values, path=np.array(measurement_coords), title=title)


# %%
