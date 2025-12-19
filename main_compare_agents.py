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
#import gpytorch
import copy
import time

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
importlib.reload(agents)

# %%
# Device selection supporting CUDA, MPS (Apple Silicon), or CPU
device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
# Important parameters
max_offset_factors = (0.7, 0.7)
turn_radius = 25
gp_pred_resolution = [100, 100]

# Setup scenario bank
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

#envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
envs_file = 'tensor_envs/2c_pCO2_112.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])
#bank.gas_coverage_cutoff(cutoff_concentration=550, cutoff_percentage=6)

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
#threshold = 550 # gas plume threshold
threshold = 405.0

# %%
# Setup lawnmower pattern, DUCB agent and RL agent
# Setup lawnmower sampling (without knowing the flow direction)
# Example simulation parameters
sample_radius = 1.0
line_spacing = 25
speed = 1.0  # m/s
data_var = 'pCO2'
depth = 112 #68
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
#kappas = [255/15.0, 255/20.0, 255/25.0, 255/30.0]
kappa_scale = 5.0/255
kappa_scale_back = 1/kappa_scale
#kappas = np.array([1.8]) * (5.0/255.0)
kappas = np.array([0.2, 0.6, 1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 4.0]) * (5.0/255.0)
gammas = np.array([0.0, -0.1, -0.25, -0.5, -1.0, -2.0]) * 0.1
#gammas = np.array([-1.0]) * 0.1
ducb_names = []
for kappa in kappas:
    for gamma in gammas:
        name = f"DUCB_k{kappa*(255.0/5.0):.2f}_g{gamma:.2f}"
        ducb_names.append(name)

# init agent env class
env_ducb_main = rl_gas_survey_dubins_agent_env.GasSurveyDubinsAgentEnv(bank, gp_pred_resolution=gp_pred_resolution, r_weights=[1.0, 1.0, 1.0], turn_radius=turn_radius, channels=adaptive_channels, timer=False, debug=False)

# Setup RL agent
env_device = torch.device("cpu")
action_mode = ['relative', 20, 20]
channels_rl = np.array([1, 1, 0, 0, 0])
env_rl_main = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[5.0, 1.0, 1.0], channels=channels_rl, turn_radius = turn_radius, timer=False, debug=False, device=env_device)

#load_model = '1760001506_dunder_0_12259909' # DQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_model = '1761655432_dunder_0_10329902.zip'
#load_model = '1761655432_dunder_0_14219947' # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_models = ['1761655432_dunder_0_1009963', '1761655432_dunder_0_10009907', '1761655432_dunder_1_259932', '1761655432_dunder_1_10299941', '1761655432_dunder_1_14699929'] # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#rl_names = ['1M', '10M', '20M', '30M', '35M']
load_models = []
rl_names = []
models_dir = f"models"

#agent = DQN.load(f"{models_dir}/{load_model}", env=env_rl, device=env_rl.device)

# Sample limits and sample intervals for GP testing
n_samples_lim = len(sample_coords_xy)
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
iterations = 50
rmse_lawnmower = torch.zeros(n_steps)
rmse_ducb = torch.zeros(n_steps)
rmse_rl = torch.zeros(n_steps)

cumsum_lawnmower = torch.zeros(n_samples_lim)
cumsum_ducb = torch.zeros(n_samples_lim)
cumsum_rl = torch.zeros(n_samples_lim)

rmse_lawnmower_all_list = []
rmse_ducb_all = {name: [] for name in ducb_names}
rmse_rl_all = {name: [] for name in rl_names}
#rmse_ducb_all = []
#rmse_rl_all_list = []

cumsum_lawnmower_all_list = []
cumsum_ducb_all = {name: [] for name in ducb_names}
cumsum_rl_all = {name: [] for name in rl_names}
#cumsum_ducb_all = []
#cumsum_rl_all_list = []

# %%
i = 0
print('Running..')
while i < iterations:
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
    tol_deg=10.0,                    # angular tolerance for binning
    nbins=50,
    kernel=kernel_type,                # "matern" or "rbf"
    nu=nu,
    plot=False,
    report=False
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
    obs = env_ducb_main.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
    ducb_init_loc = env_ducb_main.loc
    ducb_init_hdg = env_ducb_main.heading
    ducb_envs = []
    ducb_ags = []
    measurements_ducb_list = []
    measurement_coords_ducb_list = []

    for kappa in kappas:
        for gamma in gammas:
            env_ducb = copy.deepcopy(env_ducb_main)
            #obs = env_ducb.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
            
            ducb_ag = agents.adaptive_agents(model_type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

            ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(obs=obs)
            while env_ducb.sample_idx < n_samples_lim:
                ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(env_ducb.step(waypoints=ducb_wps, heading=ducb_hdg))

            measurements_ducb = env_ducb.sampled_vals[:n_samples_lim]
            measurement_coords_ducb = env_ducb.sampled_coords[:n_samples_lim]
            measurements_ducb_list.append(measurements_ducb)
            measurement_coords_ducb_list.append(measurement_coords_ducb)
            ducb_envs.append(env_ducb)
            ducb_ags.append(ducb_ag)

    # RL agent sampling
    obs, _ = env_rl_main.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
    env_rl_main.loc = ducb_init_loc
    env_rl_main.heading = ducb_init_hdg
    rl_envs = []
    rl_ags = []
    measurements_rl_list = []
    measurement_coords_rl_list = []

    for load_model in load_models:
        #obs, _ = env_rl.reset(random_scenario=random_scenario, env_xy=env_xy, values=values)
        env_rl = copy.deepcopy(env_rl_main)
        agent = DQN.load(f"{models_dir}/{load_model}", env=env_rl, device=env_rl.device)
    
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
        measurements_rl_list.append(measurements_rl)
        measurement_coords_rl_list.append(measurement_coords_rl)
        rl_envs.append(env_rl)
        rl_ags.append(agent)

    # Compare GP estimates with truth
    
    # Get obs_truth from DUCB env for comparison with GP prediction
    obs_truth_coords = env_ducb_main._coords_flat
    obs_truth_values = torch.as_tensor(env_ducb_main.obs_truth)

    # Update GP hyperparams according to current scenario
    base_model.set_hyperparams(ell_par, ell_perp, angle_deg=random_scenario['cur_dir'], outputscale=max(sig_par, sig_perp), mean_value=obs_truth_values.mean().item())
    
    strategy_names = ["Lawnmower"] + rl_names + ducb_names
    results_intermediate = gpt_ard32.init_strategy_results(strategy_names)
    for j in gp_iterator:
        print(f'intermediate prediction (j = {j})')
        
        # Base strategies
        strategy_samples = {
            "Lawnmower": (
                measurement_coords_lawnmower[:j],
                measurements_lawnmower[:j],),
        }

        # Add each RL model as its own strategy
        for name, coords_rl, vals_rl in zip(
            rl_names, measurement_coords_rl_list, measurements_rl_list):
            strategy_samples[name] = (coords_rl[:j], vals_rl[:j])
        
        # Add each DUCB variant as its own strategy
        for name, coords_ducb, vals_ducb in zip(
            ducb_names, measurement_coords_ducb_list, measurements_ducb_list):
            strategy_samples[name] = (coords_ducb[:j], vals_ducb[:j])

        results_intermediate = gpt_ard32.evaluate_strategies_for_field_lognorm_gridspec(
            base_model, lik,
            obs_truth_coords, obs_truth_values,
            strategy_samples,
            results_intermediate,
            threshold=threshold,
            obs_x=env_rl_main.obs_x,
            obs_y=env_rl_main.obs_y,
            make_plot=False,   # or False for batch runs
            title_prefix=f"GP predictions for depth {random_scenario['depth']}, t={random_scenario['time']*10} ({j} samples)"
        )

    # PLOT sampling strategies
    plot_coords, plot_vals, plot_labels = gpt_ard32.assemble_agent_plot_data(strategy_samples)

    gpt_ard32.plot_sampling_comparison_n_plots(
        env_xy, values,
        plot_coords, plot_vals, plot_labels,
        threshold=threshold,
        obs_x=250, obs_y=250,
        title="Sampling strategies vs true field",
    )
    
    # Comparison statistics
    # -------------------------------
    # 1. ACCUMULATE RMSE PER FIELD
    # -------------------------------
    rmse_lawn = torch.tensor(
        results_intermediate["Lawnmower"]["rmse"],
        dtype=torch.float32
    )
    rmse_lawnmower_all_list.append(rmse_lawn)

    # All RL variants
    for name in rl_names:
        rmse_rl = torch.tensor(
            results_intermediate[name]["rmse"],
            dtype=torch.float32
        )
        rmse_rl_all[name].append(rmse_rl)
    #rmse_rl_all_list.append(rmse_rl_v)

    # All DUCB variants
    for name in ducb_names:
        rmse_du = torch.tensor(
            results_intermediate[name]["rmse"],
            dtype=torch.float32
        )
        rmse_ducb_all[name].append(rmse_du)

    # -------------------------------
    # 2. ACCUMULATE GAS-DETECTION STATS
    # -------------------------------

    # Lawnmower (fixed length)
    c_lawn = torch.cumsum(
        (measurements_lawnmower > threshold).int(),
        dim=0
    )
    cumsum_lawnmower_all_list.append(c_lawn)

    # DUCB agents (assumed fixed length n_samples_lim each)
    for name, meas_ducb in zip(ducb_names, measurements_ducb_list):
        meas_ducb_t = torch.as_tensor(meas_ducb)
        c_du = torch.cumsum(
            (meas_ducb_t > threshold).int(),
            dim=0
        )
        # if you want to enforce length n_samples_lim:
        c_du = c_du[:n_samples_lim]
        cumsum_ducb_all[name].append(c_du)
    
    # RL agents (variable length)
    for name, meas_rl in zip(rl_names, measurements_rl_list):
        meas_rl_t = torch.as_tensor(meas_rl)
        L = meas_rl_t.shape[0]
        c_rl = torch.cumsum(
            (meas_rl_t > threshold).int(),
            dim=0
        )
        
        # Pad to full length n_samples_lim by holding the last value
        if L < n_samples_lim:
            pad_val = c_rl[-1].item()
            padded = torch.cat(
                [c_rl, pad_val * torch.ones(n_samples_lim - L, dtype=torch.int)]
            )
        else:
            padded = c_rl[:n_samples_lim]
        # if you want to enforce length n_samples_lim:
        #c_rl = c_rl[:n_samples_lim]
        cumsum_rl_all[name].append(padded)

    # RL (variable length)
    #meas_rl_t = torch.as_tensor(measurements_rl)
    #L = meas_rl_t.shape[0]
    #detect_rl = (meas_rl_t > threshold).int()
    #c_rl = torch.cumsum(detect_rl, dim=0)
    #cumsum_rl_all_list.append(padded)

    i += 1

# Compute final statistics
rmse_lm_all = torch.stack(rmse_lawnmower_all_list).to(dtype=torch.float)   # (N, 6)
#rmse_ducb_all      = torch.stack(rmse_ducb_all).to(dtype=torch.float)        # (N, 6)
#rmse_rl_all        = torch.stack(rmse_rl_all_list).to(dtype=torch.float)          # (N, 6)

cumsum_lm_all = torch.stack(cumsum_lawnmower_all_list).to(dtype=torch.float)  # (N, T)
#cumsum_ducb_all      = torch.stack(cumsum_ducb_all).to(dtype=torch.float)
#cumsum_rl_all        = torch.stack(cumsum_rl_all_list).to(dtype=torch.float)

# For each RL agent, stack its list of results
rmse_rl_all_stacked = {}   # name → (N_fields, 6)
cumsum_rl_all_stacked = {} # name → (N_fields, T)

for name, lst in rmse_rl_all.items():
    rmse_rl_all_stacked[name] = torch.stack(lst).float()

for name, lst in cumsum_rl_all.items():
    cumsum_rl_all_stacked[name] = torch.stack(lst).float()

# For each DUCB agent, stack its list of results
rmse_ducb_all_stacked = {}   # name → (N_fields, 6)
cumsum_ducb_all_stacked = {} # name → (N_fields, T)

for name, lst in rmse_ducb_all.items():
    rmse_ducb_all_stacked[name] = torch.stack(lst).float()

for name, lst in cumsum_ducb_all.items():
    cumsum_ducb_all_stacked[name] = torch.stack(lst).float()

# Means
rmse_mean_lm  = rmse_lm_all.mean(dim=0)
#rmse_mean_du  = rmse_ducb_all.mean(dim=0)
#rmse_mean_rl  = rmse_rl_all.mean(dim=0)
rmse_mean_rl = {}   # name → (6,)
for name, arr in rmse_rl_all_stacked.items():
    rmse_mean_rl[name] = arr.mean(dim=0)

rmse_mean_ducb = {}   # name → (6,)
for name, arr in rmse_ducb_all_stacked.items():
    rmse_mean_ducb[name] = arr.mean(dim=0)

# Variances
rmse_var_lm   = rmse_lm_all.var(dim=0)
#rmse_var_du   = rmse_ducb_all.var(dim=0)
#rmse_var_rl   = rmse_rl_all.var(dim=0)
rmse_var_rl = {}   # name → (6,)
for name, arr in rmse_rl_all_stacked.items():
    rmse_var_rl[name] = arr.var(dim=0)

rmse_var_ducb = {}   # name → (6,)
for name, arr in rmse_ducb_all_stacked.items():
    rmse_var_ducb[name] = arr.var(dim=0)

# --- CUMSUM ---
cumsum_mean_lm = cumsum_lm_all.mean(dim=0)
#cumsum_mean_rl = cumsum_rl_all.mean(dim=0)
cumsum_mean_rl = {}  # name → (T,)
for name, arr in cumsum_rl_all_stacked.items():
    cumsum_mean_rl[name] = arr.mean(dim=0)

cumsum_mean_ducb = {}  # name → (T,)
for name, arr in cumsum_ducb_all_stacked.items():
    cumsum_mean_ducb[name] = arr.mean(dim=0)

cumsum_var_lm = cumsum_lm_all.var(dim=0)
#cumsum_var_rl = cumsum_rl_all.var(dim=0)
cumsum_var_rl = {}  # name → (T,)
for name, arr in cumsum_rl_all_stacked.items():
    cumsum_var_rl[name] = arr.var(dim=0)

cumsum_var_ducb = {}  # name → (T,)
for name, arr in cumsum_ducb_all_stacked.items():
    cumsum_var_ducb[name] = arr.var(dim=0)

# %%
# Save results
results_to_save = {
    "rmse_lawnmower": rmse_lm_all,                  # (N_fields, 6)
    "rmse_rl": rmse_rl_all_stacked,                 # dict[name → tensor(N_fields, 6)]
    "rmse_ducb": rmse_ducb_all_stacked,             # dict[name → tensor(N_fields, 6)]

    "cumsum_lawnmower": cumsum_lm_all,              # (N_fields, T)
    "cumsum_rl": cumsum_rl_all_stacked,             # dict[name → tensor(N_fields, T)]
    "cumsum_ducb": cumsum_ducb_all_stacked,         # dict[name → tensor(N_fields, T)]

    # Optional: metadata so results are traceable
    "threshold": threshold,
    "n_samples_lim": n_samples_lim,
    "rl_agent_names": list(rmse_rl_all_stacked.keys()),
    "ducb_agent_names": list(rmse_ducb_all_stacked.keys())
}

fname = f"results_{i}_runs_sc1C_{time.ctime()}.pt"
torch.save(results_to_save, 'figures/' + fname)
print(f"{fname}")

# %%
# Load from file
files_to_load = [
    '/Users/ikw/code/SmartAUVs_digital_lab/figures/results_10_runs_sc1C_Wed Dec 17 15:43:45 2025.pt',
    '/Users/ikw/code/SmartAUVs_digital_lab/figures/results_10_runs_sc1C_Wed Dec 17 15:43:45 2025.pt'
    ]
loaded = gpt_ard32.load_and_merge_results(files_to_load)

# %%
# Plotting
gpt_ard32.plot_cumsum_with_variance_multi_ducb(c_lawn=cumsum_lm_all,
                          c_ducb_dict=cumsum_ducb_all_stacked,
                          c_rl_dict=cumsum_rl_all_stacked)
gpt_ard32.plot_rmse_with_confidence_multi_ducb(rmse_lawn=rmse_lm_all,            # (N_fields, K)
                                               rmse_rl_dict=rmse_rl_all_stacked,
                                                rmse_ducb_dict=rmse_ducb_all_stacked,    # dict[name → (N_fields, K)]
                                                sample_points=gp_iterator,
                                                mode='median')
# %%
# Table view
table_str = gpt_ard32.build_rmse_table_latex(
    rmse_lm_all,
    rmse_ducb_all_stacked,
    rmse_rl_all_stacked=rmse_rl_all_stacked,
    sample_points=gp_iterator,
    ci_level=1.96,
    decimals=1,
    mode='median'
)

print(table_str)

# %%
# Heatmap view
Z, kappas_sorted, gammas_sorted = gpt_ard32.build_rmse_grid_from_names(
    rmse_by_name=rmse_ducb_all_stacked,
    kappas=kappas,
    gammas=gammas,
    mode='median'
)

gpt_ard32.plot_rmse_heatmap(
    Z,
    kappas_sorted*kappa_scale_back,
    gammas_sorted,
    title="DUCB RMSE across $(\\kappa, \\gamma)$",
    savepath="figures_p3/ducb_rmse_heatmap.eps",
)

# %%
# Print summary
# for name in strategy_names:
#     rmse_arr = np.array(results_intermediate[name]["rmse"])
#     nll_arr  = np.array(results_intermediate[name]["nll"])
#     print(f"\n{name}:")
#     print(f"  RMSE: mean={rmse_arr.mean():.4f}, std={rmse_arr.std():.4f}")
#     print(f"  NLL:  mean={nll_arr.mean():.4f}, std={nll_arr.std():.4f}")

# %%
# Plotting
#env_rl.plot_env(x=env_xy[:, 0], y=env_xy[:, 1], c=values) # original env
#env_rl.plot_env(x=measurement_coords_lawnmower[:, 0], y=measurement_coords_lawnmower[:, 1], c=measurements_lawnmower)
#env_ducb.plot_env(x=env_ducb.sampled_coords[:, 0][:env_ducb.sample_idx], y=env_ducb.sampled_coords[:, 1][:env_ducb.sample_idx], c=env_ducb.sampled_vals[:env_ducb.sample_idx])
#env_rl.plot_env(x=env_rl.sampled_coords[:, 0][:env_rl.sample_idx], y=env_rl.sampled_coords[:, 1][:env_rl.sample_idx], c=env_rl.sampled_vals[:env_rl.sample_idx])

# %%

# %%
