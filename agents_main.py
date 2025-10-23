# %%
import numpy as np
import importlib
import copy

import agents
import rl_scenario_bank
import rl_gas_survey_dubins_agent_env

# %%
importlib.reload(agents)
importlib.reload(rl_gas_survey_dubins_agent_env)
importlib.reload(rl_scenario_bank)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
channels = np.array([1, 1, 0, 1, 1])
turn_radius = 25
env = rl_gas_survey_dubins_agent_env.GasSurveyDubinsAgentEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0, 1.0], turn_radius=turn_radius, channels=channels, timer=False, debug=False)

# %%
# Reset all environments
obs = env.reset()

ig_env = copy.deepcopy(env)
ucb_env = copy.deepcopy(env)
ducb_env = copy.deepcopy(env)

# %%
kappa = 255/20.0
gamma = -1.0
ig_ag = agents.adaptive_agents(model_type='IG', obs=obs, debug=False)
ucb_ag = agents.adaptive_agents(model_type='UCB', obs=obs, kappa=kappa, debug=False)
ducb_ag = agents.adaptive_agents(model_type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

# Agents must take obs and select location based on maximizing objective
# Then compute waypoints to location (exclude illegal waypoints out of bounds)
# Compute routes for all eight possible end headings.
# If new loc is inside bounds, select the shortest.
# If new loc is outside bounds, select the shortest that has a legal path.
# If no legal paths, select second best objective location.

# Send waypoints to environment instead of action left, straight, right
# Therefore, the agents need a tailored environment class...

# %%
# Sample up to at least n_samples_lim samples
n_samples_lim = 1000

# %%
ig_wps, ig_hdg = ig_ag.get_wps_to_max_objective(obs=obs)
while ig_env.sample_idx < n_samples_lim:
    ig_wps, ig_hdg = ig_ag.get_wps_to_max_objective(ig_env.step(waypoints=ig_wps, heading=ig_hdg))
ig_env.sample_idx = n_samples_lim
ig_env._estimate()

# %%
ucb_wps, ucb_hdg = ucb_ag.get_wps_to_max_objective(obs=obs)
while ucb_env.sample_idx < n_samples_lim:
    ucb_wps, ucb_hdg = ucb_ag.get_wps_to_max_objective(ucb_env.step(waypoints=ucb_wps, heading=ucb_hdg))
ucb_env.sample_idx = n_samples_lim
ucb_env._estimate()

# %%
ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(obs=obs)
while ducb_env.sample_idx < n_samples_lim:
    ducb_wps, ducb_hdg = ducb_ag.get_wps_to_max_objective(ducb_env.step(waypoints=ducb_wps, heading=ducb_hdg))
    agents.plot_n(ducb_env._coord_x, ducb_env._coord_y, [ducb_env.pred_mu, ducb_env.pred_mu_norm_clipped], titles=["pred_mu", "pred_mu_norm_clipped"], path=ducb_env.sampled_coords[:ducb_env.sample_idx])
    agents.plot_n(ducb_env._coord_x, ducb_env._coord_y, [ducb_ag.gas_scaled, ducb_ag.map], titles=["gas_scaled", "ducb map"], path=ducb_env.sampled_coords[:ducb_env.sample_idx])
ducb_env.sample_idx = n_samples_lim
ducb_env._estimate()

# %%
# Plot n_samples_lim samples
ig_env.plot_env(x=ig_env.sampled_coords[:, 0][:n_samples_lim], y=ig_env.sampled_coords[:, 1][:n_samples_lim], c=ig_env.sampled_vals[:n_samples_lim])
ucb_env.plot_env(x=ucb_env.sampled_coords[:, 0][:n_samples_lim], y=ucb_env.sampled_coords[:, 1][:n_samples_lim], c=ucb_env.sampled_vals[:n_samples_lim])
ducb_env.plot_env(x=ducb_env.sampled_coords[:, 0][:n_samples_lim], y=ducb_env.sampled_coords[:, 1][:n_samples_lim], c=ducb_env.sampled_vals[:n_samples_lim])

# %%
# Sample up to n_actions
n_actions = 25
for _ in range(n_actions):
    ig_wps = ig_ag.get_wps_to_max_objective(ig_env.step(ig_wps))
    ucb_wps = ucb_ag.get_wps_to_max_objective(ucb_env.step(ucb_wps))
    ducb_wps = ducb_ag.get_wps_to_max_objective(ducb_env.step(ducb_wps))

# %%
env.plot_env(x=env.env_x_np, y=env.env_y_np, c=env.env_vals_np)

# %%
agents.compare_envs([ig_env, ucb_env, ducb_env], env_names=['IG', 'UCB', 'DUCB'], mean_attr="pred_mu_norm_clipped", path=True)

# %%
# Plot all samples
ig_env.plot_env(x=ig_env.sampled_coords[:, 0][:ig_env.sample_idx], y=ig_env.sampled_coords[:, 1][:ig_env.sample_idx], c=ig_env.sampled_vals[:ig_env.sample_idx])
ucb_env.plot_env(x=ucb_env.sampled_coords[:, 0][:ucb_env.sample_idx], y=ucb_env.sampled_coords[:, 1][:ucb_env.sample_idx], c=ucb_env.sampled_vals[:ucb_env.sample_idx])
ducb_env.plot_env(x=ducb_env.sampled_coords[:, 0][:ducb_env.sample_idx], y=ducb_env.sampled_coords[:, 1][:ducb_env.sample_idx], c=ducb_env.sampled_vals[:ducb_env.sample_idx])
# %%
