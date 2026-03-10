# %%
from memory_profiler import profile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import numpy as np
#import gpytorch
#import gpytorch.constraints
import importlib
import torch
#from stable_baselines3 import DQN
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv

import time
import matplotlib.pyplot as plt
import seaborn as sns

import rl_scenario_bank
import rl_gas_survey_dubins_env
import rl_DQN_PER
from rl_DQN_PER import PERDQN
import chem_utils
import gpt_class_exactgpmodel
import gpt_functions

# %%
importlib.reload(rl_gas_survey_dubins_env)
importlib.reload(rl_scenario_bank)
importlib.reload(chem_utils)
importlib.reload(gpt_functions)
importlib.reload(gpt_class_exactgpmodel)
importlib.reload(rl_DQN_PER)

def make_env(rank, bank, device):
    def _init():
        turn_radius = 25
        channels = np.array([1, 1, 0, 0, 0])
        reward_func = 'surprise'
        r_weights = [5.0, 1.0, 1.0] # r_gas, r_var, r_dist
        
        env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], r_weights=r_weights, channels=channels, turn_radius=turn_radius, reward_func=reward_func, timer=False, debug=False, device=device)
        return env
    return _init


def main():
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

    # dummy env
    channels = np.array([1, 1, 0, 0, 0])
    env = rl_gas_survey_dubins_env.GasSurveyDubinsEnv(bank, gp_pred_resolution=[100, 100], channels=channels)
    obs, info = env.reset()   # single env, not vec env

    n_envs = 2
    buffer_size = 400_000                      # how many transitions

    replay_buffer = rl_DQN_PER.PrioritizedCpuDictReplayBuffer(
        buffer_size       = buffer_size,
        observation_space = env.observation_space,
        action_space      = env.action_space,
        device            = "cpu",           # storage
        sample_device     = device,          # default target device
        n_envs            = n_envs,
        optimize_memory_usage = False,
        alpha=0.6, beta0=0.4, beta_steps=1_000_000, eps=1e-6
    )

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

    load = False
    if load:
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

        vec_env = SubprocVecEnv([make_env(i, bank, device) for i in range(n_envs)])

        agent = PERDQN(
            "MultiInputPolicy",
            vec_env,                        # env returns {"map": ..., "loc": ...}
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


    obs = vec_env.reset()

    #TIMESTEPS = 2400
    TIMESTEPS = 10000
    a=0
    while a < 1:
        agent.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            log_interval=log_interval,
            tb_log_name=f'PERDQN'
            )
        #torch.cuda.memory._dump_snapshot(f"{models_dir}/mem_{env.n_episodes}.pickle")
        agent.save(f"{models_dir}/{save_prefix}{env.total_steps}")
        agent.save_replay_buffer(f"{models_dir}/buffer.pkl")
    
    return

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

