# %%
# 1. Train CNN for area coverage using only variance channel - implement
# the prediction inside the environment
# 2. Include ScenarioBank class to help generalize learning
# 3. Test reward based on correctness of prediction mean vs. rewards for
# exploration and exploitation


# %%
import importlib
import torch
import rl_scenario_bank
import rl_conv_classes
import rl_classes

# %%
importlib.reload(rl_conv_classes)
importlib.reload(rl_scenario_bank)

# %%
bank = rl_scenario_bank.ScenarioBank()

data_dir = '../scenario_1c_medium/'
data_file = 'SMART-AUVs_OF-June-1c-0003.nc'
param = 'pCO2'
depth = 66
time = 3

bank.load_dataset(data_file)
bank.add_env(param, depth, 5)

# %%
bank.print_envs_info()

# %%
env = rl_conv_classes.GasSurveyEnv(bank)

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
agent = rl_conv_classes.PPOAgent(state_n, action_n, env.env_xy, agent_speed, sampling_freq, agent_small_grid_bins, nn_filename='my_nn.nn')

# %%
total_rewards = train_conv_agent(env, agent, episodes=1000, max_steps=100, epsilon_decay=0.99, train_mode=True)
