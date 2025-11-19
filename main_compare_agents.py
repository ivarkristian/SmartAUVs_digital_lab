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
import matplotlib.pyplot as plt
import gpytorch

import rl_scenario_bank
import lawnmower_path as lp
import chem_utils
import path
import rl_gas_survey_dubins_agent_env
import rl_gas_survey_dubins_env
import agents
#from gpt_class_exactgpmodel import ExactGPModel
import variograms
import gpt_ard32

# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "Dejavu Serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
importlib.reload(rl_gas_survey_dubins_agent_env)
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(variograms)
importlib.reload(gpt_ard32)

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
    
# Important parameters
max_offset_factors = (0.7, 0.7)
turn_radius = 25
gp_pred_resolution = [100, 100]

# Setup scenario bank
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

# Setup GP model for evaluation
kernel_type = "matern"
nu = 0.5 # matern nu

# Find GP parameters from variogram
z = values.view(250, 250).detach().cpu().numpy()
ell_par, ell_perp, sig_par, sig_perp, *rest = variograms.fit_and_plot_anisotropic_variogram(
    z, direction_deg=random_scenario['cur_dir'],   # e.g., 30.0
    tol_deg=20.0,                    # angular tolerance for binning
    nbins=50,
    kernel=kernel_type,
    nu=nu,
    plot=False,
    report=True
    )

base_model, lik = gpt_ard32.build_base_model(random_scenario['cur_dir'],
                                             ell_par, ell_perp, max(sig_par, sig_perp), z.mean(), nu=nu)
threshold = 550 # gas plume threshold

# %%
# Setup lawnmower pattern, DUCB agent and RL agent
strategy_names = ["Lawnmower", "DUCB", "RL"]
results = gpt_ard32.init_strategy_results(strategy_names)

# Setup lawnmower sampling (without knowing the flow direction)
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
#path_delta = (sc_len_x - path_len)/2.0
path_delta = 10
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

# Setup adaptive sampling agent
adaptive_channels = np.array([1, 1, 0, 1, 1])
kappa = 255/20.0
gamma = -1.0
n_samples_lim = len(sample_coords_xy)

# init agent env class
env_ducb = rl_gas_survey_dubins_agent_env.GasSurveyDubinsAgentEnv(bank, gp_pred_resolution=gp_pred_resolution, r_weights=[1.0, 1.0, 1.0], turn_radius=turn_radius, channels=adaptive_channels, timer=False, debug=False)

# Setup RL agent
env_device = torch.device("cpu")
action_mode = ['relative', 20, 20]
channels_rl = np.array([1, 1, 0, 0, 0])
env_rl = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[5.0, 1.0, 1.0], channels=channels_rl, turn_radius = turn_radius, timer=False, debug=False, device=env_device)

#load_model = '1760001506_dunder_0_12259909' # DQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_model = '1761655432_dunder_0_10329902.zip'
load_model = '1761655432_dunder_0_14219947' # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
models_dir = f"models"
agent = DQN.load(f"{models_dir}/{load_model}", env=env_rl, device=env_rl.device)

# Sample limits and sample intervals for GP testing
sample_start = 1000
sample_step = 200
sample_end = (n_samples_lim)-((n_samples_lim-sample_start)%sample_step)

def stepped_array(n, m, s):
    vals = list(np.arange(n, m, s))
    vals.append(m)
    return np.array(vals)

gp_iterator = stepped_array(sample_start, n_samples_lim, sample_step)
n_steps = len(gp_iterator)

# %%
# Start loop here
rmse_lawnmower = torch.zeros(n_steps)
rmse_ducb = torch.zeros(n_steps)
rmse_rl = torch.zeros(n_steps)

cumsum_lawnmower = torch.zeros(n_samples_lim)
cumsum_ducb = torch.zeros(n_samples_lim)
cumsum_rl = torch.zeros(n_samples_lim)

rmse_lawnmower_all = []
rmse_ducb_all = []
rmse_rl_all = []

cumsum_lawnmower_all = []
cumsum_ducb_all = []
cumsum_rl_all = []

i = 0
print('Running..')
while i < 30:
    print(f'Scenario {i}...')
    
    # Sample a scenario
    random_scenario = bank.sample()
    rotation = random.choice([-90, 0, 90, 180])

    env_xy = bank.rotate_xy(random_scenario['coords'].to(device), rotation)
    values = random_scenario['values'].to(device)
    random_scenario['cur_dir'] += rotation

    env_xy, values = bank.offset_xy(env_xy, values, max_offset_factors)

    # numpy versions
    env_xy_np = env_xy.cpu().numpy()
    env_vals_np = values.cpu().numpy()

    # Find GP parameters from variogram
    z = values.view(250, 250).detach().cpu().numpy()
    ell_par, ell_perp, sig_par, sig_perp, *rest = variograms.fit_and_plot_anisotropic_variogram(
    z, direction_deg=random_scenario['cur_dir'],   # e.g., 30.0
    tol_deg=20.0,                    # angular tolerance for binning
    nbins=50,
    kernel=kernel_type,                # "matern" or "rbf"
    nu=nu,
    plot=True,
    report=True
    )

    # Lawnmower sampling
    measurements_lawnmower = []
    measurement_coords_lawnmower = []
    for coord in sample_coords_xy:
        msr = chem_utils.extract_synoptic_chemical_data_from_depth(env_xy_np[:, 0], env_xy_np[:, 1], env_vals_np, coord, sample_radius)
        measurements_lawnmower.append(msr)
        measurement_coords_lawnmower.append(coord)

    measurements_lawnmower = torch.tensor(measurements_lawnmower, dtype=torch.float32)
    measurement_coords_lawnmower = torch.tensor(measurement_coords_lawnmower, dtype=torch.float32)

    # DUCB sampling
    obs = env_ducb.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
    ducb_init_loc = env_ducb.loc
    ducb_init_hdg = env_ducb.heading
    ducb_ag = agents.adaptive_agents(model_type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

    ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(obs=obs)
    while env_ducb.sample_idx < n_samples_lim:
        ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(env_ducb.step(waypoints=ducb_wps, heading=ducb_hdg))

    measurements_ducb = env_ducb.sampled_vals[:n_samples_lim]
    measurement_coords_ducb = env_ducb.sampled_coords[:n_samples_lim]

    # RL agent sampling
    obs, _ = env_rl.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
    env_rl.loc = ducb_init_loc
    env_rl.heading = ducb_init_hdg
    truncated = False
    rewards = np.array([])
    q_values = []

    while env_rl.sample_idx < n_samples_lim and not truncated:
        if env_rl.debug:
            q_vec = rl_gas_survey_dubins_env.get_q_values(agent, obs)
            q_values.append(q_vec)

        action, _step = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_rl.step(int(action))
        #rewards = np.append(rewards, reward)
        if truncated:
            print(f'RL truncated: {truncated}')

    if len(q_values):
        q_values = np.vstack(q_values)

    measurements_rl = env_rl.sampled_vals[:min(env_rl.sample_idx, n_samples_lim)]
    measurement_coords_rl = env_rl.sampled_coords[:min(env_rl.sample_idx, n_samples_lim)]

    # Compare GP estimates with truth
    
    # Get obs_truth from DUCB env for comparison with GP prediction
    obs_truth_coords = env_ducb._coords_flat
    obs_truth_values = torch.as_tensor(env_ducb.obs_truth)

    # Update GP hyperparams according to current scenario
    base_model.set_hyperparams(ell_par, ell_perp, angle_deg=random_scenario['cur_dir'], outputscale=max(sig_par, sig_perp), mean_value=obs_truth_values.mean().item())

    results_intermediate = gpt_ard32.init_strategy_results(strategy_names)
    for j in gp_iterator:
        print(f'intermediate prediction (j = {j})')
        strategy_samples = {
            "Lawnmower": (measurement_coords_lawnmower[:j], measurements_lawnmower[:j]),
            "DUCB":      (measurement_coords_ducb[:j], measurements_ducb[:j]),
            "RL":        (measurement_coords_rl[:j], measurements_rl[:j]),
        }

        results_intermediate = gpt_ard32.evaluate_strategies_for_field_lognorm_gridspec(
            base_model, lik,
            obs_truth_coords, obs_truth_values,
            strategy_samples,
            results_intermediate,
            obs_x=env_rl.obs_x,
            obs_y=env_rl.obs_y,
            make_plot=True,   # or False for batch runs
            title_prefix=f"GP predictions for depth {random_scenario['depth']}, t={random_scenario['time']*10} ({j} samples)"
        )

    strategy_samples = {
            "Lawnmower": (measurement_coords_lawnmower[:n_samples_lim], measurements_lawnmower[:n_samples_lim]),
            "DUCB":      (measurement_coords_ducb[:n_samples_lim], measurements_ducb[:n_samples_lim]),
            "RL":        (measurement_coords_rl[:n_samples_lim], measurements_rl[:n_samples_lim]),
        }
    
    gpt_ard32.plot_sampling_comparison_lognorm_gridspec(
        env_xy=env_xy,
        values=values,
        measurement_coords_lawnmower=measurement_coords_lawnmower,
        measurements_lawnmower=measurements_lawnmower,
        ducb_coords=measurement_coords_ducb,
        ducb_vals=measurements_ducb,
        rl_coords=measurement_coords_rl,
        rl_vals=measurements_rl,
        obs_x=250, obs_y=250
    )
    
    # Comparison statistics
    #   1. ACCUMULATE RMSE PER FIELD
    # -------------------------------
    rmse_lawn  = torch.tensor(results_intermediate["Lawnmower"]["rmse"], dtype=torch.float32)
    rmse_du    = torch.tensor(results_intermediate["DUCB"]["rmse"], dtype=torch.float32)
    rmse_rl_v  = torch.tensor(results_intermediate["RL"]["rmse"], dtype=torch.float32)

    rmse_lawnmower_all.append(rmse_lawn)
    rmse_ducb_all.append(rmse_du)
    rmse_rl_all.append(rmse_rl_v)

    #   2. ACCUMULATE GAS-DETECTION STATS
    # -------------------------------
    # Lawnmower (fixed length):
    c_lawn = torch.cumsum((measurements_lawnmower > threshold).int(), dim=0)
    cumsum_lawnmower_all.append(c_lawn)

    # DUCB (fixed length):
    c_du = torch.cumsum((measurements_ducb > threshold).int(), dim=0)
    cumsum_ducb_all.append(c_du)

    # RL (variable length):
    L = measurements_rl.shape[0]
    detect_rl = (measurements_rl > threshold).int()
    c_rl = torch.cumsum(detect_rl, dim=0)

    # Pad to full length n_samples_lim by holding the last value
    if L < n_samples_lim:
        pad_val = c_rl[-1].item()
        padded = torch.cat([c_rl, pad_val * torch.ones(n_samples_lim - L, dtype=torch.int)])
    else:
        padded = c_rl[:n_samples_lim]

    cumsum_rl_all.append(padded)

    #rmse_lawnmower += torch.tensor(results_intermediate['Lawnmower']["rmse"])
    #rmse_ducb += torch.tensor(results_intermediate['DUCB']["rmse"])
    #rmse_rl += torch.tensor(results_intermediate['RL']["rmse"])

    #threshold = 550
    #x = torch.arange(0, n_samples_lim)
    #cumsum_lawnmower += torch.cumsum((measurements_lawnmower > threshold).int(), dim=0)
    #cumsum_ducb += torch.cumsum((measurements_ducb > threshold).int(), dim=0)
    #L = measurements_rl.shape[0]                  # length of the current measurement vector
    #cs_rl = torch.cumsum((measurements_rl > threshold).int(), dim=0)
    #cumsum_rl[:L] += cs_rl
    #cumsum_rl[L:] += cs_rl[-1].item()

    i += 1

# Compute final statistics
rmse_lawnmower_all = torch.stack(rmse_lawnmower_all).to(dtype=torch.float)   # (N, 6)
rmse_ducb_all      = torch.stack(rmse_ducb_all).to(dtype=torch.float)        # (N, 6)
rmse_rl_all        = torch.stack(rmse_rl_all).to(dtype=torch.float)          # (N, 6)

cumsum_lawnmower_all = torch.stack(cumsum_lawnmower_all).to(dtype=torch.float)  # (N, T)
cumsum_ducb_all      = torch.stack(cumsum_ducb_all).to(dtype=torch.float)
cumsum_rl_all        = torch.stack(cumsum_rl_all).to(dtype=torch.float)

# Means
rmse_mean_lm  = rmse_lawnmower_all.mean(dim=0)
rmse_mean_du  = rmse_ducb_all.mean(dim=0)
rmse_mean_rl  = rmse_rl_all.mean(dim=0)

# Variances
rmse_var_lm   = rmse_lawnmower_all.var(dim=0)
rmse_var_du   = rmse_ducb_all.var(dim=0)
rmse_var_rl   = rmse_rl_all.var(dim=0)

cumsum_mean_lm = cumsum_lawnmower_all.mean(dim=0)
cumsum_var_lm  = cumsum_lawnmower_all.var(dim=0)
cumsum_mean_du = cumsum_ducb_all.mean(dim=0)
cumsum_var_du  = cumsum_ducb_all.var(dim=0)
cumsum_mean_rl = cumsum_rl_all.mean(dim=0)
cumsum_var_rl  = cumsum_rl_all.var(dim=0)

# %%
# Plotting
gpt_ard32.plot_rmse_with_confidence(rmse_lawnmower_all, rmse_ducb_all, rmse_rl_all, gp_iterator)
gpt_ard32.plot_cumsum_with_variance(cumsum_lawnmower_all, cumsum_ducb_all, cumsum_rl_all)
gpt_ard32.plot_rmse_and_cumsum_panels(rmse_lawnmower_all, rmse_ducb_all, rmse_rl_all,
                                cumsum_lawnmower_all, cumsum_ducb_all, cumsum_rl_all,
                                gp_iterator)
'''
# Cumsum gas plume samples
plt.style.use("seaborn-v0_8-whitegrid")  # minimal grid style
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

ax.plot(x, cumsum_lawnmower, lw=2.5, color="#1f77b4", label="Lawnmower")
ax.plot(x, cumsum_ducb, lw=2.5, color="#ff7f0e", label="DUCB")
ax.plot(x, cumsum_rl, lw=2.5, color="#2ca02c", label="RL")

# --- Beautify ---
ax.set_xlabel("Sample", fontsize=13)
ax.set_ylabel("Cumulative sum", fontsize=13)
ax.set_title("Gas plume samples", fontsize=15, pad=6)
ax.legend(frameon=False, fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# GP RMSE mean plot
plt.style.use("seaborn-v0_8-whitegrid")  # minimal grid style
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

ax.plot(gp_iterator, rmse_lawnmower, lw=2.5, color="#1f77b4", label="Lawnmower")
ax.plot(gp_iterator, rmse_ducb, lw=2.5, color="#ff7f0e", label="DUCB")
ax.plot(gp_iterator, rmse_rl, lw=2.5, color="#2ca02c", label="RL")

# --- Beautify ---
ax.set_xlabel("Sample", fontsize=13)
ax.set_ylabel("GP RMSE", fontsize=13)
ax.set_title("GP RMSE vs number of samples", fontsize=15, pad=6)
ax.legend(frameon=False, fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
'''
# %%
# Print summary
for name in strategy_names:
    rmse_arr = np.array(results_intermediate[name]["rmse"])
    nll_arr  = np.array(results_intermediate[name]["nll"])
    print(f"\n{name}:")
    print(f"  RMSE: mean={rmse_arr.mean():.4f}, std={rmse_arr.std():.4f}")
    print(f"  NLL:  mean={nll_arr.mean():.4f}, std={nll_arr.std():.4f}")

# %%
# Plotting
#env_rl.plot_env(x=env_xy[:, 0], y=env_xy[:, 1], c=values) # original env
#env_rl.plot_env(x=measurement_coords_lawnmower[:, 0], y=measurement_coords_lawnmower[:, 1], c=measurements_lawnmower)
#env_ducb.plot_env(x=env_ducb.sampled_coords[:, 0][:env_ducb.sample_idx], y=env_ducb.sampled_coords[:, 1][:env_ducb.sample_idx], c=env_ducb.sampled_vals[:env_ducb.sample_idx])
#env_rl.plot_env(x=env_rl.sampled_coords[:, 0][:env_rl.sample_idx], y=env_rl.sampled_coords[:, 1][:env_rl.sample_idx], c=env_rl.sampled_vals[:env_rl.sample_idx])

# %%

# %%
