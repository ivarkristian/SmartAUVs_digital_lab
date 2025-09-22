# %%
import gpytorch.constraints
from memory_profiler import profile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np
import gpytorch

import importlib
import torch
from stable_baselines3 import DQN

import time
import matplotlib.pyplot as plt
import seaborn as sns

import rl_scenario_bank
import rl_gas_survey_dubins_env
import chem_utils
import gpt_class_exactgpmodel
import gpt_functions

# %%
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)
importlib.reload(gpt_functions)
importlib.reload(gpt_class_exactgpmodel)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

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
    

turn_radius = 25
channels = np.array([1, 1, 0, 0, 0])
reward_func = None
r_weights = [3.0, 1.0, 1.0] # r_gas, r_var, r_dist
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=r_weights, channels=channels, turn_radius=turn_radius, reward_func=reward_func, timer=False, debug=False, device=device)

buffer_size = 400_000                      # how many transitions

replay_buffer = rl_gas_survey_dubins_env.CpuDictReplayBuffer(
    buffer_size       = buffer_size,
    observation_space = env.observation_space,
    action_space      = env.action_space,
    device            = "cpu",           # storage
    sample_device     = device,          # default target device
    optimize_memory_usage = False
)

# %%
host = socket.gethostname().split('.')[0]
if host in ['dunder', 'cupid', 'dancer', 'rudolph', 'dasher']:
    parent_dir = "/projects/robin/users/ivarkriw"
    log_interval = 100
else:
    parent_dir = '.'
    log_interval = 30

models_parent = parent_dir + "/models"
logs_parent = parent_dir + "/logs"

current_dir = f"/{int(time.time())}_{host}"
models_dir = models_parent + current_dir
logs_dir = logs_parent + current_dir

load = False
if load:
    load_time = '1749667471_dunder' # timestamp_host
    load_model = '0_1323' # zip-file without extension
    save_prefix = '1_'
    models_dir = f"{models_parent}/{load_time}"

    agent = DQN.load(f"{models_dir}/{load_model}", env=env, device=env.device)
    try:
        agent.load_replay_buffer(f"{models_dir}/buffer.pkl")
    except:
        print(f'Could not load replay buffer from {models_dir}/buffer.pkl')

    print(f'Loaded model from {models_dir}/{load_model}')
else:
    save_prefix = '0_'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

#    policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=256))
    policy_kwargs = dict(
        features_extractor_class=rl_gas_survey_dubins_env.MapPlusLocExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    agent = DQN(
        "MultiInputPolicy",
        env,                        # env returns {"map": ..., "loc": ...}
        device=env.device,
        buffer_size=buffer_size,
        batch_size=256,
        learning_rate=3e-4,
        learning_starts=256,
        tau=0.005,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logs_dir
    )

    agent.replay_buffer = replay_buffer          # overwrite in place

# %%
#agent = PPO('MlpPolicy', env, verbose=1, n_steps=4, batch_size=2, n_epochs=2)
#torch.cuda.memory._record_memory_history()
#TIMESTEPS = 2400
TIMESTEPS = 10000
a=0
while a < 1:
    agent.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        log_interval=log_interval,
        tb_log_name=f'DQN'
        )
    #torch.cuda.memory._dump_snapshot(f"{models_dir}/mem_{env.n_episodes}.pickle")
    agent.save(f"{models_dir}/{save_prefix}{env.total_steps}")
    agent.save_replay_buffer(f"{models_dir}/buffer.pkl")

# %%
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

# %%
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

# %%
# NLE detect subsets
min_cluster_dist = 14
threshold = 555
min_region_dist = 100

sample_coords = env.sampled_coords[:env.sample_idx]
sample_values = env.sampled_vals[:env.sample_idx]

ret = gpt_functions.detect_clusters_wrapper(sample_coords, sample_values, threshold, min_cluster_dist, min_region_dist, debug=False)
anomaly, no_anomaly, sample_coords_subsets, sample_values_subsets, cluster, clusters_tensors, regions, regions_tensors = ret

# --- training data set plotting
ax = plt.subplots()
plt.title('Anomaly clusters and regions for NLE (training)')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
df_nle = gpt_functions.tensors_to_df(sample_coords, sample_values)
ax = sns.scatterplot(data=df_nle, x='x', y='y', hue='value', palette='YlOrRd', s=4, hue_norm=(sample_values.min(), sample_values.max()))

showlegend=True

for i, cluster in enumerate(clusters_tensors):
    df_cluster = gpt_functions.tensors_to_df(cluster)
    ax = sns.lineplot(data=df_cluster, x='x', y='y', color='green', sort=False, estimator=None)
    showlegend=False

showlegend=True
for i, region in enumerate(regions_tensors):
    df_region = gpt_functions.tensors_to_df(region)
    ax = sns.lineplot(data=df_region, x='x', y='y', color='red', sort=False, estimator=None)
    showlegend=False

plt.show()

# %%
# NLE models init
num_nle_models = 2
type = 'scale_rbf'
noise_interval=gpytorch.constraints.Interval(0.0001, 1)
#length_constraint=gpytorch.constraints.GreaterThan([25, 2.5])

coords_subsets = [sample_coords[anomaly], sample_coords[no_anomaly]]
values_subsets = [sample_values[anomaly], sample_values[no_anomaly]]

nle_likelihoods = []
nle_models = []
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i in range(num_nle_models):
    gpt_functions.plot_2d_NLE(ax=axs[i], sample_coords=coords_subsets[i], sample_values=values_subsets[i], cmap='YlOrRd')
    nle_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
    nle_likelihoods.append(nle_likelihood)
    nle_model = gpt_class_exactgpmodel.ExactGPModel(coords_subsets[i], values_subsets[i], nle_likelihoods[i], type, constant_constraint=gpytorch.constraints.GreaterThan(values_subsets[i].min()))
    nle_models.append(nle_model)

# %%
# NLE train on subsets
iter = 80
e_delta = [False, False]

for i, model in enumerate(nle_models):
    st = time.process_time()
    gpt_functions.train_model(coords_subsets[i], values_subsets[i], model, iter=iter, early_delta=(e_delta[i], 'mll', sample_coords_subsets[i], sample_values_subsets[i], 0), debug=False)
    x_l = []
    s = []
    for n in model.lengthscales[:model.curr_trained]:
        x_l.append(n[0])

    model.print_named_parameters()

fig_mix_train, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
m = ['Anomaly', 'Background']
showlegend = True

linestyles = ['-', '--']

for i, model in enumerate(nle_models):
    x_l = [n[0] for n in model.lengthscales[:model.curr_trained]]

    axs.plot(x_l, label='Lx %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)

axs.set_xlabel('Iterations', fontsize=12)
axs.set_ylabel('Lengthscale [m]', fontsize=12)

# %%
# NLE predictions for each model
preds_NLE = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    #model.set_train_data(coords_region_subsets_test[i], values_region_subsets_test[i], strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        st = time.process_time()
        pred = model.likelihood(model(env._coords_flat)) # both models predict entire domain

    preds_NLE.append(pred)

# Then pick prediction model based on within anomaly region or not
mask = torch.zeros(len(env._coords_flat))
preds_mean_NLE = torch.zeros(len(env._coords_flat))
sigmas_lower_NLE = torch.zeros(len(env._coords_flat))
sigmas_upper_NLE = torch.zeros(len(env._coords_flat))

st = time.process_time()
for polygon in regions:
    mask = mask + polygon.contains_points(env._coords_flat.numpy()).astype(int)
NLE_contains_points_time = time.process_time() - st

mask = (mask > 0)
mask_int = mask.int()
mask_inv = mask.__invert__().int()
preds_mean_NLE = preds_mean_NLE + preds_NLE[0].mean*mask_int
preds_mean_NLE = preds_mean_NLE + preds_NLE[1].mean*mask_inv
sigmas_lower_NLE = sigmas_lower_NLE + preds_NLE[0].confidence_region()[0]*mask_int
sigmas_lower_NLE = sigmas_lower_NLE + preds_NLE[1].confidence_region()[0]*mask_inv
sigmas_upper_NLE = sigmas_upper_NLE + preds_NLE[0].confidence_region()[1]*mask_int
sigmas_upper_NLE = sigmas_upper_NLE + preds_NLE[1].confidence_region()[1]*mask_inv

figs, axs = plt.subplots(2, 2, figsize=(10, 10))

title='Anomaly mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 0], sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=preds_NLE[0].mean, title=title)
title='Anomaly stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 1], sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=preds_NLE[0].stddev, title=title)
title='Background mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 0], sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=preds_NLE[1].mean, title=title)
title='Background stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 1], sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=preds_NLE[1].stddev, title=title)
figs.show()

# %%
# NLE-MA prediction
window_value = 10
distances = torch.norm(env._coords_flat.unsqueeze(1) - sample_coords.unsqueeze(0), dim=2)
nearest = distances.topk(window_value, largest=False, dim=1) # The window_value smallest elements

# Weighted MLL
preds_NLE_weight = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    model.set_train_data(sample_coords, sample_values, strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(sample_coords)) # both models predict observations
    preds_NLE_weight.append(pred)

# Get the K's and u's to be weighted
K0 = preds_NLE_weight[0].covariance_matrix
u0 = preds_NLE_weight[0].mean
K1 = preds_NLE_weight[1].covariance_matrix
u1 = preds_NLE_weight[1].mean
log_pr = torch.full((len(distances), 2), 0.0)

for i, nearest_indices in enumerate(nearest.indices):
    weights = torch.tensor([0.0]*len(sample_coords))
    weights[nearest_indices] = (torch.tensor(1.0).div(nearest.values[i])).clamp_max(1.0)
    log_pr[i, 0] = gpt_functions.compute_weighted_mll(sample_values, K0, weights, u0, jitter=1e-6)
    log_pr[i, 1] = gpt_functions.compute_weighted_mll(sample_values, K1, weights, u1, jitter=1e-6)
    if i%1000 == 0:
        print(f'{i}...')

max_log_probs = torch.max(log_pr, dim=1)[0].unsqueeze(1)
log_probs_n = log_pr - max_log_probs
factor = log_probs_n[:, 0].min()/log_probs_n[:, 1].min()
normed_log_pr = torch.full((len(distances), 2), 0.0)
normed_log_pr[:, 0] = log_probs_n[:, 0]
normed_log_pr[:, 1] = log_probs_n[:, 1]*factor

resps = torch.exp(gpt_functions.compute_responsibilities(normed_log_pr, normalize=False))
mean_bma, sigmas_lower_bma, sigmas_upper_bma, nle_bma_pofz = gpt_functions.predict_mixture(env._coords_flat, nle_models, (resps[:, 0], resps[:, 1]))

# %%
# Plot NLE-MA
figs, axs = plt.subplots(1, 2, figsize=(10, 4.5))
title = 'Mix mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0], cluster_tensors=clusters_tensors, region_tensors=regions_tensors, sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=mean_bma, title=title, cmap='YlOrRd')
fig.legend(loc='upper right')

title = 'Mix stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1], cluster_tensors=clusters_tensors, region_tensors=regions_tensors, sample_coords=sample_coords, smooth_coords=env._coords_flat, smooth_values=(sigmas_upper_bma-mean_bma)/2.0, title=title, cmap='YlOrRd')
fig.legend(loc='upper right')

# %%
