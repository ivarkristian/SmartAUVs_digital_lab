# %%
from memory_profiler import profile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np
import importlib
import torch

import time
import matplotlib.pyplot as plt
import seaborn as sns

import rl_scenario_bank
import rl_gas_survey_dubins_env
from rl_DQN_PER import PERDQN

# Load environments
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_files = ['tensor_envs/environments_train.pt']
bank.load_envs(envs_files)

# Device selection supporting CUDA, MPS (Apple Silicon), or CPU
device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu") 

turn_radius = 25
channels = np.array([1, 1, 0, 0, 0])
reward_func = 'surprise'
r_weights = [5.0, 1.0, 1.0] # r_gas, r_var, r_dist
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=r_weights, channels=channels, turn_radius=turn_radius, reward_func=reward_func, timer=False, debug=False, device=device)

host = socket.gethostname().split('.')[0]
if host in ['dunder', 'cupid', 'dancer', 'rudolph', 'dasher']:
    parent_dir = "/projects/robin/users/ivarkriw"
    log_interval = 1000
else:
    parent_dir = '.'
    log_interval = 30

models_parent = parent_dir + "/models"
logs_parent = parent_dir + "/logs"

current_dir = f"/{int(time.time())}_{host}"
models_dir = models_parent + current_dir
logs_dir = logs_parent + current_dir

load_time = '1749667471_dunder' # timestamp_host
load_model = '0_1323' # zip-file without extension
save_prefix = '1_'
models_dir = f"{models_parent}/{load_time}"

agent = PERDQN.load(f"{models_dir}/{load_model}", env=env, device=env.device)
try:
    agent.load_replay_buffer(f"{models_dir}/buffer.pkl")
except:
    print(f'Could not load replay buffer from {models_dir}/buffer.pkl')

print(f'Loaded model from {models_dir}/{load_model}')

# Run an episode (with a random agent)
agent.exploration_rate = 0.9

obs, _ = env.reset()
done = False
rewards = np.array([])
q_values = []
while not done:
    if env.debug:
        #q_vec = rl_gas_survey_discrete_env.get_q_values(agent, obs)
        q_vec = rl_gas_survey_dubins_env.get_q_values(agent, obs)
        q_values.append(q_vec)

    action, _step = agent.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards = np.append(rewards, reward)
    done = terminated or truncated

q_values = np.vstack(q_values)

# Plot rewards and results from one episode
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.obs_truth, path=env.sampled_coords[:env.sample_idx], value_title='obs_truth')
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_var_norm, path=env.sampled_coords[:env.sample_idx], value_title='pred_var')
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_mu_norm, path=env.sampled_coords[:env.sample_idx], value_title='pred_mu')

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