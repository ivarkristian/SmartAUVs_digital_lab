# The purpose of this script is to support paper 3 - agents in a digital ocean laboratory
# The script samples from the database of leakage scenarios, and runs simulated surveys
# with the different agents in every scenario. The acquired sets of samples and the
# resulting GP estimates are compared across the agents.

# %%
import importlib
import random
import torch
import numpy as np
from stable_baselines3 import DQN

import rl_scenario_bank
import lawnmower_path as lp
import chem_utils
import path
import rl_gas_survey_dubins_agent_env
import rl_gas_survey_dubins_env
import agents

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
# Important parameters
max_offset_factors = (0.7, 0.7)
turn_radius = 25
gp_pred_resolution = [100, 100]

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])
bank.gas_coverage_cutoff(cutoff_concentration=550, cutoff_percentage=6)

# %%
# Sample a scenario for init, with random rotation and offset
random_scenario = bank.sample()
rotation = random.choice([-90, 0, 90, 180])

env_xy = bank.rotate_xy(random_scenario['coords'].to(device), rotation)
values = random_scenario['values'].to(device)

env_xy, values = bank.offset_xy(env_xy, values, max_offset_factors)

# numpy versions
env_xy_np = env_xy.cpu().numpy()
env_vals_np = values.cpu().numpy()

# %%
# Lawnmower sampling (without knowing the flow direction)
# Example simulation parameters
sample_radius = 2.0
line_spacing = 25
speed = 1.0  # m/s
data_var = 'pCO2'
depth = 68
start_time = '2020-01-01T02:10:00.000000000'
sample_freq = 1.0  # Sample frequency in Hz (samples per second)

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

# Generate waypoints
waypoints_with_turns, x_coords, y_coords, z_coords = lp.generate_lawnmower_waypoints(
    x_data, y_data, width=line_spacing, min_turn_radius=25, siglay=depth, direction='x')

# remove duplicate waypoints
waypoints_with_turns = lp.remove_consecutive_duplicate_wps(waypoints_with_turns, 1e-3)


plain_sample_coords_times_list = path.path(waypoints_with_turns, start_time, speed, sample_freq, synoptic=True)
sample_coords_xy = [row[:2] for row in plain_sample_coords_times_list]

# %%
# Start loop here
# Sample a scenario for init, with random rotation and offset
random_scenario = bank.sample()
rotation = random.choice([-90, 0, 90, 180])

env_xy = bank.rotate_xy(random_scenario['coords'].to(device), rotation)
values = random_scenario['values'].to(device)

env_xy, values = bank.offset_xy(env_xy, values, max_offset_factors)

# numpy versions
env_xy_np = env_xy.cpu().numpy()
env_vals_np = values.cpu().numpy()

measurements_lawnmower = []
measurement_coords_lawnmower = []
for coord in sample_coords_xy:
    msr = chem_utils.extract_synoptic_chemical_data_from_depth(env_xy_np[:, 0], env_xy_np[:, 1], env_vals_np, coord, sample_radius)
    measurements_lawnmower.append(msr)
    measurement_coords_lawnmower.append(coord)

# Do a GP estimate here!

# %%
# Adaptive sampling agent
adaptive_channels = np.array([1, 1, 0, 1, 1])
kappa = 255/20.0
gamma = -1.0
n_samples_lim = len(measurements_lawnmower)

# init agent env class
ducb_env = rl_gas_survey_dubins_agent_env.GasSurveyDubinsAgentEnv(bank, gp_pred_resolution=gp_pred_resolution, r_weights=[1.0, 1.0, 1.0], turn_radius=turn_radius, channels=adaptive_channels, timer=False, debug=False)

# set env to env_xy
obs = ducb_env.reset(env_xy=env_xy, values=values)
ducb_ag = agents.adaptive_agents(model_type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(obs=obs)
while ducb_env.sample_idx < n_samples_lim:
    ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(ducb_env.step(waypoints=ducb_wps, heading=ducb_hdg))
    #agents.plot_n(ducb_env._coord_x, ducb_env._coord_y, [ducb_env.pred_mu, ducb_env.pred_mu_norm_clipped], titles=["pred_mu", "pred_mu_norm_clipped"], path=ducb_env.sampled_coords[:ducb_env.sample_idx])
    #agents.plot_n(ducb_env._coord_x, ducb_env._coord_y, [ducb_ag.gas_scaled, ducb_ag.map], titles=["gas_scaled", "ducb map"], path=ducb_env.sampled_coords[:ducb_env.sample_idx])
ducb_env.sample_idx = n_samples_lim
ducb_env._estimate()

ducb_env.plot_env(x=ducb_env.sampled_coords[:, 0][:n_samples_lim], y=ducb_env.sampled_coords[:, 1][:n_samples_lim], c=ducb_env.sampled_vals[:n_samples_lim])

# %%
# RL agent
env_device = torch.device("cpu")
action_mode = ['relative', 20, 20]
channels_rl = np.array([1, 1, 0, 0, 0])
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[5.0, 1.0, 1.0], channels=channels_rl, turn_radius = turn_radius, timer=False, debug=True, device=env_device)

load_model = '1760001506_dunder_0_12259909' # DQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
models_dir = f"models"
agent = DQN.load(f"{models_dir}/{load_model}", env=env, device=env.device)

# Run an episode
obs, _ = env.reset(env_xy=env_xy, values=values)
done = False
rewards = np.array([])
q_values = []
for i in range(len(measurement_coords_lawnmower)):
    if env.debug:
        #q_vec = rl_gas_survey_discrete_env.get_q_values(agent, obs)
        q_vec = rl_gas_survey_dubins_env.get_q_values(agent, obs)
        q_values.append(q_vec)

    action, _step = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    rewards = np.append(rewards, reward)
    done = terminated or truncated

q_values = np.vstack(q_values)
