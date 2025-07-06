# %%
from memory_profiler import profile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np

import importlib
import torch
from stable_baselines3 import DQN

import time
import rl_scenario_bank
import rl_gas_survey_dubins_env
import chem_utils

# %%
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

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

turn_radius = 25
channels = np.array([0, 1, 0, 0, 0])
env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], channels=channels, turn_radius=turn_radius, timer=False, debug=False, device=device)

buffer_size = 800_000                      # how many transitions

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

while True:
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
