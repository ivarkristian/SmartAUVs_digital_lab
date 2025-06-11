# %%
import torch
from stable_baselines3 import SAC
import importlib

import rl_scenario_bank
import rl_gas_survey_env

# %%
importlib.reload(rl_gas_survey_env)
importlib.reload(rl_scenario_bank)

# %%
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# %%
env_device = torch.device("cpu")
env = rl_gas_survey_env.GasSurveyEnv(bank, gp_pred_resolution=[100, 100], r_weights=[1.0, 1.0], timer=False, debug=True, device=env_device)

# %%
load_time = '1749488972'
load_model = '0_1992.zip'
models_dir = f"models/{load_time}/"

agent = SAC.load(f"{models_dir}/{load_model}", env=env, device=env.device)

# %%
obs, _ = env.reset()
done = False
while not done:
    action, _step = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# %%
env.plot_env(x=env._coord_x, y=env._coord_y, c=env.pred_var)