"""!@brief Lab 2, Problem 2 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 2 main file.
@author Martin Schuck, Damian Valle
@date 14.12.2020
"""

from pathlib import Path
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent
from DDPG_soft_updates import soft_updates
from Agent import Actor, Critic
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


def fill_buffer():
    agent = RandomAgent(m)
    # Reset enviroment data
    while len(buffer) < L:
        done = False
        state = env.reset()
        while not done:
            action = agent.forward(state)
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done)) 
            state = next_state
    print('Buffer filled!')


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                                # Number of episodes to run for training
discount_factor = 0.99                          # Value of gamma
n_ep_running_average = 50                       # Running average of 50 episodes
m = len(env.action_space.high)                  # dimensionality of the action
dim_state = len(env.observation_space.high)     # State dimensionality
L = int(3e4)                                    # Length experience buffer
batch_size = 64                                 # Training batch size
alpha_actor = 5e-5                              # Learning rate actor
alpha_critic = 5e-4                             # Learning rate critic
tau = 1e-3                                      # Target networks update constant
policy_update_hz = 2                            # Target networks update frequency
n = np.zeros((m))                               # Running Ornstein-Uhlenbeck process vector
mu = 0.15                                       # Ornstein-Uhlenbeck process mean
sigma = 0.2                                     # Ornstein-Uhlenbeck process variance


# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(dim_state, m).to(device=dev)
critic = Critic(dim_state).to(device=dev)
target_actor = Actor(dim_state, m).to(device=dev)
target_critic = Critic(dim_state).to(device=dev)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=alpha_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=alpha_critic)


buffer = ExperienceReplayBuffer(maximum_length=L, combine=True)
fill_buffer()


# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    n = np.zeros((m))
    t = 0
    while not done:
        # Noise process step
        n = -mu*n + np.random.multivariate_normal(np.zeros((m)), sigma**2*np.diag(np.ones((m))))
        # Take a random action
        action = np.clip(actor.forward(torch.tensor([state]).to(device=dev))[0].cpu().detach().numpy() + n,-1,1)

        next_state, reward, done, _ = env.step(action)
        
        # Add experience to buffers
        buffer.append((state, action, reward, next_state, done)) 

        # Training
        critic_optimizer.zero_grad()
        states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
        # Compute target values
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device=dev)
        with torch.no_grad():
            next_actions = target_actor.forward(next_states_tensor)
            next_q_values = target_critic.forward(next_states_tensor, next_actions).squeeze()
            rewards = torch.tensor(rewards,requires_grad=False, dtype=torch.float32).to(device=dev)
            dones = 1 - torch.tensor(dones,requires_grad=False, dtype=torch.float32).to(device=dev)
            target_values = rewards + discount_factor*next_q_values*dones
        
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device=dev)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device=dev)
        action_values = critic.forward(states_tensor, actions_tensor)

        loss = torch.nn.functional.mse_loss(action_values.squeeze(), target_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
        critic_optimizer.step()
        if t%policy_update_hz == 0:
            actor_optimizer.zero_grad()
            actions = actor.forward(states_tensor)
            action_values = critic.forward(states_tensor, actions)
            loss = -torch.mean(action_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            actor_optimizer.step()
            target_actor = soft_updates(actor, target_actor, tau)
            target_critic = soft_updates(critic, target_critic, tau)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
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

    if running_average(episode_reward_list, n_ep_running_average)[-1] > 200:
        break

path = Path(__file__).resolve().parent
#torch.save(actor.to('cpu'), path.joinpath('neural-network-2-actor.pth'))
#torch.save(critic.to('cpu'), path.joinpath('neural-network-2-critic.pth'))

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
