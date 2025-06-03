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

import importlib
import torch
from stable_baselines3 import PPO, SAC

import time
import rl_scenario_bank
import rl_gas_survey_env
import rl_classes
import chem_utils

# %%
importlib.reload(rl_gas_survey_env)
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
env = rl_gas_survey_env.GasSurveyEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 0.0], timer=False, debug=True)

# %%
env = rl_gas_survey_env.GasSurveyEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 0.0], timer=False, debug=False)

load = False
if load:
    load_time = '1747301502'
    load_model = '0_1349.zip'
    save_prefix = '1_'
    models_dir = f"models/{load_time}/"
    logdir = f"logs/{load_time}/"

    agent = SAC.load(f"{models_dir}/{load_model}", env=env, device=env.device)
    try:
        agent.load_replay_buffer(f"{models_dir}/buffer.pkl")
    except:
        print(f'Could not load replay buffer from {models_dir}/buffer.pkl')

    print(f'Loaded model from {models_dir}/{load_model}')
else:
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"
    save_prefix = '0_'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #agent = PPO('MlpPolicy', env, device=env.device, verbose=1, tensorboard_log=logdir)
    policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=256))

    agent = SAC(
        "CnnPolicy",
        env,
        device=env.device,          # 'cuda', 'mps', or 'cpu'
        buffer_size=40000,        # fewer GP calls than PPO
        batch_size=256,
        learning_rate=3e-4,
        learning_starts=265,
        tau=0.005,                  # target-network smoothing
        train_freq=1,
        gradient_steps=1,           # one GD step per env.step()
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logdir
    )

# %%
#agent = PPO('MlpPolicy', env, verbose=1, n_steps=4, batch_size=2, n_epochs=2)
TIMESTEPS = 940

while True:
    agent.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        log_interval=30,
        tb_log_name=f'SAC'
        )
    
    agent.save(f"{models_dir}/{save_prefix}{env.n_episodes}")
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
