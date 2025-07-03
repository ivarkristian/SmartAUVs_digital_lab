# %%
import numpy as np
import importlib
import copy

import agents
import rl_scenario_bank
import rl_gas_survey_env

# %%
importlib.reload(agents)
importlib.reload(rl_gas_survey_env)
importlib.reload(rl_scenario_bank)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
action_mode = ['absolute', 250, 250]
channels = np.array([1, 1, 0, 1, 1])
env = rl_gas_survey_env.GasSurveyEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], channels=channels, action_mode=action_mode, timer=False, debug=False)

# %%
# Reset all environments
obs = env.reset()

ig_env = copy.deepcopy(env)
ucb_env = copy.deepcopy(env)
ducb_env = copy.deepcopy(env)

# %%
kappa = 255/20.0
gamma = -1.0
ig_ag = agents.adaptive_agents(type='IG', obs=obs, debug=False)
ucb_ag = agents.adaptive_agents(type='UCB', obs=obs, kappa=kappa, debug=False)
ducb_ag = agents.adaptive_agents(type='DUCB', obs=obs, kappa=kappa, gamma=gamma, debug=False)

# %%
ig_action = ig_ag.get_action(obs=obs)
ucb_action = ucb_ag.get_action(obs=obs)
ducb_action = ducb_ag.get_action(obs=obs)
n_samples_lim = 1000

# %%
# Sample up to at least n_samples_lim samples
while ig_env.sample_idx < n_samples_lim:
    ig_action = ig_ag.get_action(ig_env.step(ig_action))
ig_env.sample_idx = n_samples_lim
ig_env._estimate()

while ucb_env.sample_idx < n_samples_lim:
    ucb_action = ucb_ag.get_action(ucb_env.step(ucb_action))
ucb_env.sample_idx = n_samples_lim
ucb_env._estimate()

# %%
while ducb_env.sample_idx < n_samples_lim:
    ducb_action = ducb_ag.get_action(ducb_env.step(ducb_action))
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
    ig_action = ig_ag.get_action(ig_env.step(ig_action))
    ucb_action = ucb_ag.get_action(ucb_env.step(ucb_action))
    ducb_action = ducb_ag.get_action(ducb_env.step(ducb_action))

# %%
env.plot_env(x=env.env_x_np, y=env.env_y_np, c=env.env_vals_np)

agents.compare_envs([ig_env, ucb_env, ducb_env], env_names=['IG', 'UCB', 'DUCB'], mean_attr="pred_mu_norm_clipped", path=True)

# %%
# Plot all samples
ig_env.plot_env(x=ig_env.sampled_coords[:, 0][:ig_env.sample_idx], y=ig_env.sampled_coords[:, 1][:ig_env.sample_idx], c=ig_env.sampled_vals[:ig_env.sample_idx])
ucb_env.plot_env(x=ucb_env.sampled_coords[:, 0][:ucb_env.sample_idx], y=ucb_env.sampled_coords[:, 1][:ucb_env.sample_idx], c=ucb_env.sampled_vals[:ucb_env.sample_idx])
ducb_env.plot_env(x=ducb_env.sampled_coords[:, 0][:ducb_env.sample_idx], y=ducb_env.sampled_coords[:, 1][:ducb_env.sample_idx], c=ducb_env.sampled_vals[:ducb_env.sample_idx])
# %%
