# %%
import torch
import gpytorch
import numpy as np
import importlib

import dubins

import rl_gas_survey_discrete_env
import agents
import rl_scenario_bank

# %%
importlib.reload(agents)

# %%
# Import scenarios. Have to use the same scenarios for lm, adap, rl.
bank = rl_scenario_bank.ScenarioBank(data_dir='.')

envs_file = 'tensor_envs/1c_pCO2_67_69.pt'
bank.load_envs(envs_file)
sensor_range = [0, 2000]
bank.clip_sensor_range(parameter='pCO2', min=sensor_range[0], max=sensor_range[1])

# Every call to env.reset() samples a new scenario. Lm can run on the env.env_xy and env.vals

# %%
# Testing Dubins path to achieve more realistic paths

# Initialize the planner with the turn radius and the desired distance between consecutive points
local_planner = dubins.Dubins(radius=49, point_separation=1.0)

# Define start and end points with their headings
east = 0
north = 3.1415/2
west = 3.1415
south = -north

start = (50, 50, north)  # heading east
end = (100, 100, east)  # heading west

# Compute the path between them
path = local_planner.dubins_path(start, end)
agents.plot_n(x=path[:, 0], y=path[:, 1], data_list=[np.ones_like(path[:, 0])], path=path)


# %%
