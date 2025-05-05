# %%
import gpytorch.constraints
import torch
import random
import matplotlib.pyplot as plt
import importlib
import timeit
import gpytorch
import rl_classes
import path
#torch.set_default_device('cpu')

# %%
importlib.reload(rl_classes)

# %% The Training Loop
def train_agent(env, agent, episodes=1000, max_steps=200, epsilon_decay=0.98, min_epsilon=0.1, train_mode=False):
    
    if train_mode:
        epsilon = agent.epsilon
    else:
        epsilon = 0.0

    max_x = env.env_xy[:, 0].max()
    max_y = env.env_xy[:, 1].max()
    bin_x = max_x/agent_small_grid_bins
    bin_y = max_y/agent_small_grid_bins
    half_bin_x = int(bin_x/2)
    half_bin_y = int(bin_y/2)
    total_rewards = []
    bin_tensor = torch.tensor([bin_x, bin_y])
    grid_offset = torch.tensor([2, 2], device=agent.device)
    new_loc_init_offset = torch.tensor([half_bin_x-1, half_bin_y-1])

    for episode in range(episodes):
        
        start_time = timeit.default_timer()
        total_reward = 0
        done = torch.tensor([0], device=agent.device)
        illegal_action_n = 0
        
        # Init by sampling along a random line
        agent.location = torch.tensor([random.randint(0+half_bin_x, max_x-half_bin_y), random.randint(0+half_bin_y, max_y-half_bin_y)])
        new_loc = agent.location + new_loc_init_offset
        n_new_samples = env.step(agent.location, new_loc, agent.speed, agent.sampling_freq)
        agent.location = new_loc
        agent.small_grid_location = torch.div(agent.location, bin_tensor, rounding_mode='floor').to(agent.device) - grid_offset
        
        # Init predictions
        current_prediction = agent.estimate_env(env.env_xy, env.sampled_coords, env.sampled_vals)
        agent.current_pred_mean = current_prediction.mean
        agent.current_pred_variance = current_prediction.variance
        agent.small_grid_mean = agent.make_small_grid(env.env_xy, current_prediction.mean)
        agent.small_grid_variance = agent.make_small_grid(env.env_xy, current_prediction.variance)
        state = torch.cat((agent.small_grid_variance, agent.small_grid_location))
        
        for step in range(max_steps):
            action = agent.select_action(state, epsilon, train_mode) # if train_mode is False, then no random actions
            if train_mode is False:
                print('')
                agent.print_grid(agent.small_grid_variance)
                print(f'Loc: {agent.small_grid_location}', end=' ')
                print(f'Action: ', end='')
                agent.print_action(action)

            new_small_grid_loc = agent.new_small_grid_location_from_action(action)
            
            if new_small_grid_loc.max() > 2 or new_small_grid_loc.min() < -2:
                new_small_grid_loc = agent.small_grid_location
                new_loc = agent.location
                #print(f'Tried illegal action. Staying put at {new_small_grid_loc} ({new_loc})')
                illegal_action = 1
                n_new_samples = 0
            else:
                new_loc_x = (new_small_grid_loc[0] + 2)*bin_x + bin_x/2
                new_loc_y = (new_small_grid_loc[1] + 2)*bin_y + bin_y/2
                new_loc_x += random.randint(-int(half_bin_x/2), int(half_bin_x/2))
                new_loc_y += random.randint(-int(half_bin_y/2), int(half_bin_y/2))
                new_loc = torch.tensor((new_loc_x, new_loc_y), device='cpu') # keep new_loc on cpu
                # path.path crashes if new_loc == old_loc
                if (new_loc == agent.location).all():
                    new_loc[0] += 1

                illegal_action = 0
                n_new_samples = env.step(agent.location, new_loc, agent.speed, agent.sampling_freq)
            
            # The agents must have separate memories.
            # Move agent to new loc, while sampling.
            if n_new_samples:
                next_prediction = agent.estimate_env(env.env_xy, env.sampled_coords, env.sampled_vals)
                next_prediction_mean = next_prediction.mean
                next_prediction_variance = next_prediction.variance
            else:
                next_prediction_mean = agent.current_pred_mean
                next_prediction_variance = agent.current_pred_variance

            # New small_grids
            next_small_grid_mean = agent.make_small_grid(env.env_xy, next_prediction_mean)
            next_small_grid_variance = agent.make_small_grid(env.env_xy, next_prediction_variance)

            if illegal_action:
                reward = -10
            else:
                #reward = agent.compute_reward(agent.current_pred_variance, next_prediction_variance)
                reward = (agent.compute_reward(agent.small_grid_variance, next_small_grid_variance)*10)**2 # Encourage rewards above 1.0

            if train_mode is False:
                print(f'Reward: {reward}')
            
            if step == max_steps - 1:
                done[0] = 1
            
            # state is small grid variance, could be small grid mean
            next_state = torch.cat((agent.small_grid_variance, new_small_grid_loc))
            agent.store_transition(state, action, reward, next_state, done)

            agent.location = new_loc
            agent.small_grid_location = new_small_grid_loc
            agent.current_pred_mean = next_prediction_mean
            agent.current_pred_variance = next_prediction_variance
            agent.small_grid_mean = next_small_grid_mean
            agent.small_grid_variance = next_small_grid_variance

            state = next_state
            total_reward += reward
            illegal_action_n += illegal_action
            if done:
                break

        if train_mode:
            agent.train(training_per_episode=min(episode, 5))
            epsilon = max(epsilon * epsilon_decay, min_epsilon)
        
        # Make plots
        if (episode == 0) or ((episode + 1) % 1 == 0):
            fig, ax = env.plot_env(title_postfix=f'ep {episode + 1}', path=True)
        
        if train_mode is False:
            fig, ax = agent.plot_estimate(env.env_xy, next_prediction_variance, title=f'Variance ep: {episode} step {step}')

        end_time = timeit.default_timer()
        episode_time = end_time - start_time
        # Print info
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:2f}, Epsilon: {epsilon:.2f}, Time: {episode_time:.0f} s.")
        print(f'Sampled {len(env.sampled_coords)} locations, did {illegal_action_n} illegal_actions')
        total_rewards.append((total_reward, illegal_action_n, len(env.sampled_coords)))

        # Save model
        if (episode + 1) % 100 == 0:
            agent.epsilon = epsilon
            if train_mode:
                agent.save_model(agent.nn_filename)

        # Get ready for next episode
        env.reset() # delete sample memory
        agent.reset() # set small_grid_mean, small_grid_variance to 0
    
    agent.epsilon = epsilon
    if train_mode:
        agent.save_model(agent.nn_filename)

    return total_rewards

# %%
importlib.reload(rl_classes)
importlib.reload(path)

# %%
# Environment init
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
# Agent init
agent_speed = 1.0
sampling_freq = 1.0
agent_small_grid_bins = 5
state_n = agent_small_grid_bins**2 + 2
action_n = 4
device = 'cpu'
lengthscale_constraint = gpytorch.constraints.Interval(9, 11)
nn_file = 'my_nn_1.nn'
agent = rl_classes.GPAgent(state_n, action_n, env.env_xy, agent_speed, sampling_freq, agent_small_grid_bins, gp_lengthscale_constraint=lengthscale_constraint, nn_filename=nn_file, device=device)

# %%
# Training
episodes = 3
steps = 25
eps_decay = 0.995
train_mode = False
total_rewards = train_agent(env, agent, episodes=episodes, max_steps=steps, epsilon_decay=eps_decay, train_mode=train_mode)

# %%
# Extract values
episodes = torch.arange(len(total_rewards))
total_reward_values = [r[0] for r in total_rewards]
illegal_actions_values = [r[1] for r in total_rewards]
sampled_coords_values = [r[2] for r in total_rewards]

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Total reward plot
axes[0].plot(episodes, total_reward_values, label="Total Reward", color='blue', linestyle='-', marker='.')
axes[0].set_ylabel("Total Reward")
axes[0].set_title(f"Total Reward per Episode ({steps} steps)")
axes[0].legend()
axes[0].grid(True)

# Illegal actions plot
axes[1].plot(episodes, illegal_actions_values, label="Illegal Actions", color='red', linestyle='-', marker='.')
axes[1].set_ylabel("Illegal Actions")
axes[1].set_title("Illegal Actions per Episode")
axes[1].legend()
axes[1].grid(True)

# Sampled coordinates plot
axes[2].plot(episodes, sampled_coords_values, label="Sampled Coords", color='green', linestyle='-', marker='.')
axes[2].set_xlabel("Episodes")
axes[2].set_ylabel("Number of Sampled Coordinates")
axes[2].set_title("Number of Sampled Coordinates per Episode")
axes[2].legend()
axes[2].grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %%
