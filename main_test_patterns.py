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

# %%
importlib.reload(rl_scenario_bank)
importlib.reload(path)
importlib.reload(rl_gas_survey_dubins_env)

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

# %%
# Define sizes for lawnmower pattern
scenario_x_min = env_xy[:, 0].min().cpu().numpy()
scenario_x_max = env_xy[:, 0].max().cpu().numpy()
scenario_y_min = env_xy[:, 1].min().cpu().numpy()
scenario_y_max = env_xy[:, 1].max().cpu().numpy()

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
sphere_radius = 4
line_spacing = 30
speed = 1.0  # m/s
data_var = 'pCO2'
depth = 68
start_time = '2020-01-01T02:10:00.000000000'
sample_freq = 0.1  # Sample frequency in Hz (samples per second)
threshold = 550  # Threshold for the chemical data variable

# %%
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=line_spacing, min_turn_radius=25, siglay=depth, direction='x')

# remove duplicate waypoints
waypoints_with_turns = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)


plain_sample_coords_list = path.path(waypoints_with_turns, start_time, speed, sample_freq, synoptic=True)

# %%
""" Run the path function with the bowtie pattern as an example """
# Simulate AUV path and collect data
measurements = np.zeros(len(sample_coords_xy), dtype=np.float32)
radius = 1.0 # Radius of sample averaging

for c, coord in enumerate(sample_coords_xy):
    measurements[c] = chem_utils.extract_synoptic_chemical_data_from_depth(self.env_x_np, self.env_y_np, self.env_vals_np, coord, radius)


measurements, sample_coords = path.path_with_samples(chemical_dataset, bubble_dataset, start_time, speed, way_points, sample_frequency, threshold, bowtie, data_var, sphere_radius=sphere_radius)
