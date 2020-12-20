"""!@brief Lab 2, Problem 1 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 1 main file.
@author Martin Schuck, Damian Valle
@date 10.12.2020
"""

# Load packages
import time
import numpy as np
from pathlib import Path
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from Agent import DeepAgent, AdvantageAgent
from ReplayBuffer import ExperienceReplayBuffer

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 600                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# Training parameters
alpha = 1e-3
epsilon_max = 1.
epsilon_min = 0.01
Z = 0.9*N_episodes
clipping_value = 1.3
L = int(3e4)                                # Buffer size
N = 64                                      # Training batch size

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random agent, target, buffer and optimizer initialization
l1_size = 128
Q1 = AdvantageAgent(dim_state, l1_size, n_actions).to(device=dev)
Q2 = AdvantageAgent(dim_state, l1_size, n_actions).to(device=dev)
buffer = ExperienceReplayBuffer(maximum_length=L, combine=True)
optimizer1 = optim.Adam(Q1.parameters(), lr=alpha)
optimizer2 = optim.Adam(Q2.parameters(), lr=alpha)

### Training process
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Choose action epsilon greedy with epsilon decay.
        epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**(i/Z))
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            actions1 = Q1.forward(torch.tensor([state], requires_grad=False).to(device=dev))
            actions2 = Q2.forward(torch.tensor([state], requires_grad=False).to(device=dev))
            action = torch.argmax(actions1 + actions2).item()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        exp = (state, action, reward, next_state, done)
        buffer.append(exp)
        # Train network.
        if len(buffer) >= N:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            states, actions, rewards, next_states, dones = buffer.sample_batch(N)
            # Double-Q learning.
            coin = np.random.rand() <= 0.5
            [Q_theta, Q_target] = [Q1, Q2] if coin else [Q2, Q1]

            state_values = Q_theta.forward(torch.tensor(states,dtype=torch.float32).to(device=dev))
            q_value = state_values[range(N), actions]
            with torch.no_grad():
                opt_actions = torch.max(Q_theta.forward(torch.tensor(next_states, requires_grad=False).to(device=dev)),1).indices
                target_max = Q_target.forward(torch.tensor(next_states, requires_grad=False).to(device=dev))[range(N),opt_actions]
                rewards = torch.tensor(rewards,requires_grad=False, dtype=torch.float32).to(device=dev)
                dones = 1 - torch.tensor(dones,requires_grad=False, dtype=torch.float32).to(device=dev)
                target_values = rewards + discount_factor*target_max*dones
            loss = nn.functional.mse_loss(q_value, target_values)
            loss.backward()
            if coin:
                nn.utils.clip_grad_norm_(Q1.parameters(), clipping_value)
                optimizer1.step()
            else:
                nn.utils.clip_grad_norm_(Q2.parameters(), clipping_value)
                optimizer2.step()

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    # Break in case training goal has been achieved.
    if i > 50 and running_average(episode_reward_list, n_ep_running_average)[-1] > 200:
        print('Training goal reached!')
        break
agent = Q1
# Show solution:
visualize = True
if visualize:
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:
        env.render()
        # Choose action on policy.
        action = torch.argmax(agent.forward(torch.tensor([state], requires_grad=False).to(device=dev))).item()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
    print(f'Total reward of optimal policy was {total_episode_reward}.')

# Save the agent
path = Path(__file__).resolve().parent
torch.save(agent.to('cpu'), path.joinpath('neural-network-1.pth'))

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

