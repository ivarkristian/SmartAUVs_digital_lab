# %%
import torch
import numpy as np
from stable_baselines3 import SAC, DQN
import importlib
import matplotlib.pyplot as plt

import rl_scenario_bank
import rl_gas_survey_env
import rl_gas_survey_discrete_env
import rl_gas_survey_dubins_env

# %%
importlib.reload(rl_gas_survey_env)
importlib.reload(rl_gas_survey_discrete_env)
importlib.reload(rl_scenario_bank)
importlib.reload(rl_gas_survey_dubins_env)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])
bank.gas_coverage_cutoff(cutoff_concentration=550, cutoff_percentage=6)

# %%
env_device = torch.device("cpu")
action_mode = ['relative', 20, 20]
channels = np.array([1, 1, 0, 0, 0])
#env = rl_gas_survey_discrete_env.GasSurveyDiscEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], channels=channels, action_mode=action_mode, timer=False, debug=True, device=env_device)
turn_radius = 25
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[5.0, 1.0, 1.0], channels=channels, turn_radius = turn_radius, timer=False, debug=True, device=env_device)

# %%
#load_time = '1749488972'
#load_model = '1751822750_rudolph_0_15769965' # DQN, 01000, disc(20, 20)
#load_model = '1752402590_dunder_0_16219927' # DQN, 01000, dubins(25)
#load_model = '1754301775_cupid_0_950000' # DQN, 11000, dubins(25)
#load_model = '1757516414_cupid_0_885000'
#load_model = '1758560582_rudolph_0_3759974' # DQN, 11000, dubins(25), r_w=[3.0,1.0,1.0]
#load_model = '1760001506_dunder_0_5689993' # DQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_model = '1760001506_dunder_0_12259909' # DQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_model = '1761655432_dunder_0_10329902.zip' # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
#load_model = '1761655432_dunder_0_14219947' # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
load_model = '1761655432_dunder_1_14699929' # PERDQN, 11000, dubins(25), r_w=[5, 1, 1], 45 deg actions
models_dir = f"models"

#agent = SAC.load(f"{models_dir}/{load_model}", env=env, device=env.device)
agent = DQN.load(f"{models_dir}/{load_model}", env=env, device=env.device)

# %%
# Run an episode
obs, _ = env.reset()
done = False
rewards = np.array([])
q_values = []
while not done:
    if env.debug:
        #q_vec = rl_gas_survey_discrete_env.get_q_values(agent, obs)
        q_vec = rl_gas_survey_dubins_env.get_q_values(agent, obs)
        q_values.append(q_vec)

    action, _step = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    rewards = np.append(rewards, reward)
    done = terminated or truncated

q_values = np.vstack(q_values)

# %%
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_var_norm, path=env.sampled_coords[:env.sample_idx])
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_mu_norm_clipped, path=env.sampled_coords[:env.sample_idx])
env.plot_env(path=env.sampled_coords[:env.sample_idx])

#q_act = ['up', 'down', 'left', 'right']
q_act = ['left', 'straight', 'right']
fig, ax = plt.subplots(figsize=(4.5, 2.2), dpi=300)   # fits two-column journals
steps = np.arange(len(rewards))
ax.plot(steps, rewards, label="reward", linewidth=0.6)
for i in range(q_values.shape[1]):
    ax.plot(steps, q_values[:, i], label=f"{q_act[i]}", linewidth=0.6)

ax.set_xlabel("Step", fontsize=8)
ax.set_ylabel("Reward", fontsize=8)
#ax.set_ylim(-1, 1)
ax.tick_params(axis="both", labelsize=7)
ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=6, ncol=q_values.shape[1]+1)
fig.tight_layout()

# %%
obs, _ = env.reset()

# %%
q_vec = rl_gas_survey_discrete_env.get_q_values(agent, obs)
print("Q-values:", q_vec)
print("greedy action :", q_vec.argmax())
action, _next_state = agent.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)

# %%
rl_gas_survey_discrete_env.show_conv3_maps(agent, obs)

# %%
