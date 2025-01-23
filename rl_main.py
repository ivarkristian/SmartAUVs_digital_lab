# %%
import torch
import random
import importlib
import rl_classes
import path

# %%
importlib.reload(rl_classes)

# %% The Training Loop
def train_agent(env, agent, episodes=1000, max_steps=200, epsilon_decay=0.995, min_epsilon=0.1):
    agent.location = torch.tensor([0.0, 0.0, env.depth])
    epsilon = 1.0
    max_x = env.env_xy[:, 0].max()
    max_y = env.env_xy[:, 1].max()
    bin_x = max_x/agent_small_grid_bins
    bin_y = max_y/agent_small_grid_bins
    half_bin_x = int(bin_x/2)
    half_bin_y = int(bin_y/2)
    
    for episode in range(episodes):
        total_reward = 0
        done = 0
        state = torch.cat((torch.zeros(agent.small_grid_bins**2), torch.tensor((0, 0))))
        for step in range(max_steps):
            # Map from 5 by 5 to 4 actions, append agent loc to input vector
            # Later map from 250 by 250 to four actions, append agent loc to input vector in
            # linear layers.
            print(f'Small grid loc: {agent.small_grid_location}')
            action = agent.select_action(state, epsilon)
            print(f'Action: ', end='')
            agent.print_action(action)

            new_small_grid_loc = agent.new_small_grid_location_from_action(action)
            if new_small_grid_loc.max() > agent_small_grid_bins or new_small_grid_loc.min() < 0:
                new_small_grid_loc = agent.small_grid_location
                new_loc = agent.location
                print(f'Tried illegal action. Staying put at {new_small_grid_loc} ({new_loc})')
            else:
                new_loc_x = new_small_grid_loc[0]*bin_x + bin_x/2
                new_loc_y = new_small_grid_loc[1]*bin_y + bin_y/2
                new_loc_x += random.randint(-half_bin_x, half_bin_x)
                new_loc_y += random.randint(-half_bin_y, half_bin_y)
                new_loc = torch.tensor((new_loc_x, new_loc_y, env.depth))

            # The agents must have separate memories.
            # Move agent to new loc, while sampling.
            n_new_samples = env.step(agent.location, new_loc, agent.speed, agent.sampling_freq)
            if n_new_samples:
                next_prediction = agent.estimate_env(env.env_xy, env.sampled_coords, env.sampled_vals)
                small_grid_mean = agent.make_small_grid(env.env_xy, next_prediction.variance)
            else:
                next_prediction = agent.current_pred

            if agent.current_pred is None:
                reward = agent.compute_reward(torch.ones_like(next_prediction.variance)*5, next_prediction.variance)
            else:
                reward = agent.compute_reward(agent.current_pred.variance, next_prediction.variance)
            
            if step == max_steps - 1:
                done = 1
            
            next_state = torch.cat((small_grid_mean, new_small_grid_loc))
            agent.store_transition(state, action, reward, next_state, done)
            
            agent.location = new_loc
            agent.small_grid_location = new_small_grid_loc
            agent.current_pred = next_prediction

            state = next_state
            total_reward += reward
            if done:
                break

        agent.train()
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        
        env.reset() # delete sample memory
        agent.reset() # 5*5 of random numbers

# %%
importlib.reload(rl_classes)
importlib.reload(path)

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
agent = rl_classes.GPAgent(agent_small_grid_bins, 4, env.env_xy, agent_speed, sampling_freq, agent_small_grid_bins)

train_agent(env, agent, episodes=2, max_steps=10)

# %%
