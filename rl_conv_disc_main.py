# %%
# 1. Train CNN for area coverage using only variance channel - implement
# the prediction inside the environment
# 2. Include ScenarioBank class to help generalize learning
# 3. Test reward based on correctness of prediction mean vs. rewards for
# exploration and exploitation

# %%
from memory_profiler import profile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np

import importlib
import torch
from stable_baselines3 import DQN
#from stable_baselines3.common.buffers import DictReplayBuffer

import time
import rl_scenario_bank
import rl_gas_survey_discrete_env
import rl_classes
import chem_utils

# %%
importlib.reload(rl_gas_survey_discrete_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])
#bank.environments = bank.environments[0:5]
#bank.print_envs_info()

# %%
# Device selection supporting CUDA, MPS (Apple Silicon), or CPU
#if torch.backends.mps.is_available():
#    self.device = torch.device("mps")
device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

#action_mode = {'absolute', 250, 250}
action_mode = ['relative', 20, 20]
channels = np.array([0, 1, 0, 0, 0])
env = rl_gas_survey_discrete_env.GasSurveyDiscEnv(bank, gp_pred_resolution=[100, 100], r_weights=[2.0, 3.0], channels=channels, action_mode=action_mode, timer=False, debug=False, device=device)

buffer_size = 800_000                      # how many transitions

replay_buffer = rl_gas_survey_discrete_env.CpuDictReplayBuffer(
    buffer_size       = buffer_size,
    observation_space = env.observation_space,
    action_space      = env.action_space,
    device            = "cpu",           # storage
    sample_device     = device,          # default target device
    optimize_memory_usage = False
)

# %%
host = socket.gethostname().split('.')[0]
if host in ['dunder', 'cupid', 'dancer']:
    parent_dir = "/projects/robin/users/ivarkriw"
else:
    parent_dir = '.'

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
        features_extractor_class=rl_gas_survey_discrete_env.MapPlusLocExtractor,
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
TIMESTEPS = 2400

while True:
    agent.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        log_interval=30,
        tb_log_name=f'DQN'
        )
    #torch.cuda.memory._dump_snapshot(f"{models_dir}/mem_{env.n_episodes}.pickle")
    agent.save(f"{models_dir}/{save_prefix}{env.total_steps}")
    agent.save_replay_buffer(f"{models_dir}/buffer.pkl")

# %%
from stable_baselines3.common.env_checker import check_env
check_env(env)

# %%
# Move around a bit
obs, reward, done, info = env.step(torch.tensor([130, 220]))
obs, reward, done, info = env.step(torch.tensor([140, 100]))
obs, reward, done, info = env.step(torch.tensor([150, 220]))
obs, reward, done, info = env.step(torch.tensor([160, 120]))


# %%
def train_conv_agent(env, agent, episodes, max_steps):
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        rewards, log_probs, actions, states, positions = [], [], [], [], []

        # init samples and prediction

        while not done:
            gp_mean = torch.FloatTensor(state["gp_mean"]).unsqueeze(0)  # Add batch dimension
            gp_variance = torch.FloatTensor(state["gp_variance"]).unsqueeze(0)
            position = torch.FloatTensor(state["position"]).unsqueeze(0)

            # Get action from policy
            action, log_prob = agent.get_action(gp_mean, gp_variance, position)

            # Compute new_loc from dx, dy in action
            new_loc = agent.location + torch.tensor([action[0], action[1]])

            # Interact with environment
            n_new_samples = env.step(agent.location, new_loc, agent.speed, agent.sampling_freq)
            if n_new_samples:
                next_prediction = agent.estimate_env(env.env_xy, env.sampled_coords, env.sampled_vals)
            
            next_state, reward, done

            # Store experience
            rewards.append(reward)
            log_probs.append(log_prob)
            actions.append(action)
            states.append((gp_mean, gp_variance))
            positions.append(position)

            # Update state
            state = next_state
            steps += 1
            if steps == max_steps:
                done = 1

        # Compute advantages
        rewards = torch.FloatTensor(rewards)
        advantages = rewards - agent.value_net(gp_mean, gp_variance).detach()

        # Update policy
        agent.optimizer.zero_grad()
        loss = agent.compute_loss(
            gp_mean, gp_variance, torch.stack(positions), torch.stack(actions),
            rewards, torch.stack(log_probs), advantages
        )
        loss.backward()
        agent.optimizer.step()


# %%
# Doing stuff
data_dir = '../scenario_1c_medium/'
data_file = 'SMART-AUVs_OF-June-1c-0003.nc'
param = 'pCO2'
depth = 66
time = 3
env = rl_classes.EnvironmentWrapper(data_dir=data_dir)
env.load_dataset(data_file)
env.set_env(parameter=param, depth=depth, time=time)
#fig = env.plot_env()

# %%
agent_speed = 1.0
sampling_freq = 1.0
agent_small_grid_bins = 5
state_n = agent_small_grid_bins**2 + 2
action_n = 4
agent = rl_gas_survey_env.PPOAgent(state_n, action_n, env.env_xy, agent_speed, sampling_freq, agent_small_grid_bins, nn_filename='my_nn.nn')

# %%
total_rewards = train_conv_agent(env, agent, episodes=1000, max_steps=100, epsilon_decay=0.99, train_mode=True)
