# The purpose of this script is to support paper 3 - agents in a digital ocean laboratory
# The script samples from the database of leakage scenarios, and runs simulated surveys
# with the different agents in every scenario. The acquired sets of samples and the
# resulting GP estimates are compared across the agents.

# %%
import importlib
import random
import os
from datetime import datetime
import traceback
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
#envs_file = 'tensor_envs/2c_pCO2_112.pt'
envs_file = 'tensor_envs/1b_pco2_67_69.pt'
scenario = envs_file.split('/')[-1].split('.')[0]

threshold = 550 # gas plume threshold
#threshold = 405

bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])
bank.gas_coverage_cutoff(cutoff_concentration=threshold, cutoff_percentage=6)

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
#kappas = np.array([1]) * (5.0/255.0)
#kappas = np.array([0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 4.0]) * (5.0/255.0)
#kappas = np.array([0.5, 1, 2, 5, 10, 20, 50]) * (5.0/255.0) # This was selected for paper 3
#kappas = np.array([12, 14, 16, 18]) * (5.0/255.0)
#kappas = np.array([20, 22, 24, 50]) * (5.0/255.0)
#kappas = np.array([30]) * (5.0/255.0)
kappas = []
gammas = []
#gammas = np.array([0.0, -0.05, -0.1, -0.2, -0.5]) # This was selected for paper 3
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
rl_names = ['10M', '20M', '30M', '40M']
load_models = ['10M', '20M', '30M', '40M']
models_dir = "models"

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

# --------------------------
# Unified containers
# --------------------------
# All GP metrics (vectors over gp_iterator) are stored in gp_all[metric][strategy].
# Cumsum (vectors over sampling index) are stored separately in cumsum_all[strategy].
gp_metrics = ["rmse", "nll", "correct_es", "false_es", "iou", "f1", "iou_w", "crps_exc"]
strategy_names = ["Lawnmower"] + rl_names + ducb_names

gp_all = {m: {s: [] for s in strategy_names} for m in gp_metrics}
cumsum_all = {s: [] for s in strategy_names}

# %%
# Start loop here
iterations = 3
rmse_lawnmower = torch.zeros(n_steps)
rmse_ducb = torch.zeros(n_steps)
rmse_rl = torch.zeros(n_steps)

cumsum_lawnmower = torch.zeros(n_samples_lim)
cumsum_ducb = torch.zeros(n_samples_lim)
cumsum_rl = torch.zeros(n_samples_lim)

rmse_lawnmower_all_list = []
rmse_ducb_all = {name: [] for name in ducb_names}
rmse_rl_all = {name: [] for name in rl_names}

cumsum_lawnmower_all_list = []
cumsum_ducb_all = {name: [] for name in ducb_names}
cumsum_rl_all = {name: [] for name in rl_names}

# ---- Single checkpoint filename (constant) ----
# Put scenario count etc. in metadata inside the file; keep filename stable.
ckpt_path = os.path.join("figures", f"results_running_{scenario}.pt")

# ---- Initialize or resume ----
if os.path.exists(ckpt_path):
    print(f"[resume] Loading existing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    gp_all = ckpt["gp_all"]               # dict[metric][strategy] -> list[tensor(len(gp_iterator))]
    cumsum_all = ckpt["cumsum_all"]       # dict[strategy] -> list[tensor(n_samples_lim)]
    succeeded = ckpt.get("succeeded", [])
    failed = ckpt.get("failed", [])
    #i = ckpt.get("i", 0)
    i = len(succeeded)

    # Continue from next index (or from i as stored)
    # If you prefer to continue from len(succeeded), use: i = len(succeeded)
    print(f"[resume] Continuing from iteration {i}. "
          f"Succeeded: {len(succeeded)}, Failed: {len(failed)}")
else:
    print(f"[init] No checkpoint found. Starting new run: {ckpt_path}")
    gp_all = {m: {s: [] for s in strategy_names} for m in gp_metrics}
    cumsum_all = {s: [] for s in strategy_names}
    succeeded, failed = [], []
    i = 0

def save_checkpoint():
    """
    Save raw lists (gp_all, cumsum_all) so we can always resume.
    We also save metadata and bookkeeping. This overwrites the same file.
    """
    ckpt = {
        # Raw, append-only storage:
        # gp_all[metric][strategy] is a list over scenarios, each element is a tensor(len(gp_iterator)).
        "gp_all": gp_all,
        # cumsum_all[strategy] is a list over scenarios, each element is a tensor(n_samples_lim).
        "cumsum_all": cumsum_all,

        # Bookkeeping:
        "i": i,
        "succeeded": succeeded,
        "failed": failed,

        # Metadata for traceability:
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "n_samples_lim": n_samples_lim,
        "gp_iterator": list(gp_iterator),
        "gp_metrics": gp_metrics,
        "strategy_names": strategy_names,
        "rl_agent_names": list(rl_names),
        "ducb_agent_names": list(ducb_names),
        "grid_shape": tuple(gp_pred_resolution),
        "area_size_m": (250.0, 250.0),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[checkpoint] Saved -> {ckpt_path} (iter={i}, ok={len(succeeded)}, fail={len(failed)})")

print('Running loop...')
while i < iterations:
    print(f"\nScenario {i} of {iterations-1} ...")
    
    try:
        # ---------------------------------------------------------------------
        # Everything inside one iteration goes inside try:
        # If any step fails, we log and continue (without incrementing succeeded).
        # ---------------------------------------------------------------------

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

        # DUCB sampling (multiple kappas/gammas)
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
                ducb_ag = agents.adaptive_agents(model_type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

                ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(obs=obs)
                while env_ducb.sample_idx < n_samples_lim:
                    #fig = ducb_ag.plot_acquisition_maps(env_ducb.sampled_coords[:env_ducb.sample_idx], s=2)
                    ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(env_ducb.step(waypoints=ducb_wps, heading=ducb_hdg))

                measurements_ducb_list.append(env_ducb.sampled_vals[:n_samples_lim])
                measurement_coords_ducb_list.append(env_ducb.sampled_coords[:n_samples_lim])
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

            while env_rl.sample_idx < n_samples_lim and not truncated:
                action, _step = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_rl.step(int(action))
                #rewards = np.append(rewards, reward)
                if truncated:
                    print(f'RL truncated: {truncated}')

            measurements_rl_list.append(env_rl.sampled_vals[:min(env_rl.sample_idx, n_samples_lim)])
            measurement_coords_rl_list.append(env_rl.sampled_coords[:min(env_rl.sample_idx, n_samples_lim)])
            rl_envs.append(env_rl)
            rl_ags.append(agent)
        
        # Get obs_truth from DUCB env for comparison with GP prediction
        obs_truth_coords = env_ducb_main._coords_flat
        obs_truth_values = torch.as_tensor(env_ducb_main.obs_truth)

        # Update GP hyperparams according to current scenario
        base_model.set_hyperparams(ell_par, ell_perp, angle_deg=random_scenario['cur_dir'], outputscale=max(sig_par, sig_perp), mean_value=obs_truth_values.mean().item())
        
        results_intermediate = gpt_ard32.init_strategy_results(strategy_names, metrics=gp_metrics)
        
        for j in gp_iterator:
            #print(f'intermediate prediction (j = {j})')
            
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
        plot_sampling_strategies = False
        #ducb_names_to_plot = ['DUCB_k1.00_g0.00', 'DUCB_k10.00_g0.00', 'DUCB_k50.00_g-0.10', 'DUCB_k0.50_g-0.10', 'DUCB_k5.00_g-0.50']
        ducb_names_to_plot = []
        rl_names_to_plot = rl_names
        if plot_sampling_strategies:
            plot_coords, plot_vals, plot_labels = gpt_ard32.assemble_agent_plot_data(strategy_samples, agents=['lawnmower'] + ducb_names_to_plot + rl_names_to_plot)

            gpt_ard32.plot_sampling_comparison_n_plots(
                env_xy, values,
                plot_coords, plot_vals, plot_labels,
                threshold=threshold,
                obs_x=250, obs_y=250,
                title="Sampling strategies vs true field",
                save_str='RL'
            )

        # --------------------------
        # ACCUMULATE (unified)
        # --------------------------
        # GP metrics: append vectors (len(gp_iterator)) per scenario
        for strat in strategy_names:
            for m in gp_metrics:
                gp_all[m][strat].append(torch.tensor(results_intermediate[strat][m], dtype=torch.float32))

        # Detection cumsum curves: append vectors (len = n_samples_lim) per scenario
        cumsum_all["Lawnmower"].append(gpt_ard32.cumsum_above_threshold(measurements_lawnmower, threshold, n_samples_lim))

        for name, meas_rl in zip(rl_names, measurements_rl_list):
            cumsum_all[name].append(gpt_ard32.cumsum_above_threshold(meas_rl, threshold, n_samples_lim))

        for name, meas_du in zip(ducb_names, measurements_ducb_list):
            cumsum_all[name].append(gpt_ard32.cumsum_above_threshold(meas_du, threshold, n_samples_lim))

        succeeded.append(i)

        # Save after each successful iteration
        save_checkpoint()

    except Exception as e:
        # Log error and continue to next iteration
        print(f"[error] Iteration {i} failed: {type(e).__name__}: {e}")
        print(traceback.format_exc(limit=10))
        failed.append({"iter": i, "error": repr(e)})

        # Save checkpoint even on failure so you keep progress + error logs
        save_checkpoint()

    finally:
        # Always advance to the next iteration index
        i += 1

# --------------------------
# Stack results for analysis
# --------------------------
# gp_stacked[metric][strategy] -> (N_fields, N_j)
gp_stacked = {
    m: {s: torch.stack(lst).float() for s, lst in per_strat.items() if len(lst) > 0}
    for m, per_strat in gp_all.items()
}

# cumsum_stacked[strategy] -> (N_fields, T)
cumsum_stacked = {s: torch.stack(lst).float() for s, lst in cumsum_all.items() if len(lst) > 0}

# Convenience summaries (optional)
gp_mean = {m: {s: arr.mean(dim=0) for s, arr in per.items()} for m, per in gp_stacked.items()}
gp_var  = {m: {s: arr.var(dim=0)  for s, arr in per.items()} for m, per in gp_stacked.items()}
cumsum_mean = {s: arr.mean(dim=0) for s, arr in cumsum_stacked.items()}
cumsum_var  = {s: arr.var(dim=0)  for s, arr in cumsum_stacked.items()}

# Save results (compact + traceable)
results_to_save = {
    "gp_metrics": gp_stacked,     # gp_metrics[metric][strategy] = (N_fields, N_j)
    "cumsum": cumsum_stacked,     # cumsum[strategy] = (N_fields, T)
    "succeeded": succeeded,
    "failed": failed,

    # Optional summaries (remove if you prefer only raw)
    "gp_mean": gp_mean,
    "gp_var": gp_var,
    "cumsum_mean": cumsum_mean,
    "cumsum_var": cumsum_var,

    # Metadata
    "threshold": threshold,
    "n_samples_lim": n_samples_lim,
    "gp_iterator": list(gp_iterator),
    "strategy_names": strategy_names,
    "rl_agent_names": list(rl_names),
    "ducb_agent_names": list(ducb_names),
    "grid_shape": tuple(gp_pred_resolution),
    "area_size_m": (250.0, 250.0),
}

final_path = os.path.join("figures", f"results_final_{scenario}.pt")
torch.save(results_to_save, final_path)
print(f"[done] Final stacked results saved -> {final_path}")

# %%
# Load from finished file 
files_to_load = [
    #'figures/results_20_runs_sc2C_Sat Dec 20 04:10:11 2025.pt',
    #'figures/results_20_runs_sc2C_Sat Dec 20 17:02:44 2025.pt'
    'figures/results_final_1c_pCO2_67_69.pt'
    ]
gp_stacked, cumsum_stacked, succeeded, failed = gpt_ard32.recover_results(files_to_load[0])

# %%
# Plotting
ducb_names_to_plot = ['DUCB_k1.00_g0.00', 'DUCB_k10.00_g0.00', 'DUCB_k50.00_g-0.10', 'DUCB_k0.50_g-0.10', 'DUCB_k5.00_g-0.50']
rl_names_to_plot = []
gp_metrics_to_plot = ["rmse", "iou_w", "crps_exc"]

cumsum_lm_all = cumsum_stacked["Lawnmower"]
cumsum_ducb_all_stacked = {n: cumsum_stacked[n] for n in ducb_names_to_plot}
cumsum_rl_all_stacked = {n: cumsum_stacked[n] for n in rl_names_to_plot}

gpt_ard32.plot_cumsum_with_variance_multi_ducb(c_lawn=cumsum_lm_all,
                          c_ducb_dict=cumsum_ducb_all_stacked,
                          c_rl_dict=cumsum_rl_all_stacked,
                          ci=False, save_str='DUCB_1c')

for metric in gp_metrics_to_plot:

    metric_lm_all = gp_stacked[metric]["Lawnmower"]
    metric_ducb_all_stacked = {n: gp_stacked[metric][n] for n in ducb_names_to_plot}
    metric_rl_all_stacked = {n: gp_stacked[metric][n] for n in rl_names_to_plot}

    gpt_ard32.plot_running_metric_with_confidence(metric_lawn=metric_lm_all,            # (N_fields, K)
                                                    metric_rl_dict=metric_rl_all_stacked,
                                                    metric_ducb_dict=metric_ducb_all_stacked,    # dict[name → (N_fields, K)]
                                                    sample_points=gp_iterator,
                                                    metric=metric,
                                                    mode='median',
                                                    save_str=f'DUCB_1c')

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

for metric in gp_metrics_to_plot:
    # Build dict[name -> tensor(N_fields, N_j)] for DUCB only
    metric_ducb = {name: gp_stacked[metric][name] for name in ducb_names}

    Z, kappas_sorted, gammas_sorted = gpt_ard32.build_rmse_grid_from_names(
        rmse_by_name=metric_ducb,
        kappas=kappas,
        gammas=gammas,
        mode='median'
    )

    # Title + file naming
    if metric == 'rmse': nice_metric = 'RMSE' 
    elif metric == 'iou_w': nice_metric = r'IoU$_w$'
    elif metric == 'crps_exc': nice_metric = r'CRPS$_{\mathcal{E}(\tau)}$'
    title = f"DUCB {nice_metric} across $(\\kappa, \\gamma)$"
    savepath = f"figures_p3/ducb_{metric}_heatmap.eps"
    if metric in ['correct_es', 'iou', 'iou_w', 'f1']:
        cmap = 'viridis' # high values are good
    else:
        cmap = 'viridis_r' # low values are good
    
    gpt_ard32.plot_rmse_heatmap(
        Z,
        kappas_sorted * kappa_scale_back,
        gammas_sorted,
        title=title,
        cmap=cmap,
        cbar_label=nice_metric,
        savepath=savepath
    )




# %%
