# %%
#import torch
import importlib
import rl_classes

# %%
importlib.reload(rl_classes)

# %% The Training Loop
def train_agent(env, agent, episodes=1000, max_steps=200, epsilon_decay=0.995, min_epsilon=0.1):
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, epsilon)
            # Perhaps env.step should return nothing, just store new samples, and
            # then an agent method should predict and compute rewards?
            # The agents must have separate memories.
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.train()
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")


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
fig = env.plot_env()

agent = rl_classes.Agent(env.observation_space, env.action_space)
#train_agent(env, agent)
