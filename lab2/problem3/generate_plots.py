"""!@brief Lab 2, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 plots file.
@author Martin Schuck, Damian Valle
@date 10.12.2020
"""


from pathlib import Path
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from PPO_agent import RandomAgent


def plot_value_function(critic):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Value Function')
    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    Y, W = np.meshgrid(y, w)
    M = np.zeros_like(Y)
    for i in range(M.shape[0]): 
        for j in range(M.shape[1]):
            state = torch.tensor([np.array([0, y[i], 0, 0, w[j], 0, 0, 0])], dtype=torch.float32)
            action_value = critic.forward(state)[0].cpu().detach().numpy()
            M[j,i] = action_value
    ax.plot_surface(Y, W, M, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Height')
    ax.set_ylabel('Lander angle')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

def plot_policy(actor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Engine direction. Right=1, Nothing=0, Left=-1')
    y = np.linspace(0, 1.5, 300)
    w = np.linspace(-np.pi, np.pi, 300)
    Y, W = np.meshgrid(y, w)
    M = np.zeros_like(Y)
    for i in range(M.shape[0]): 
        for j in range(M.shape[1]):
            state = torch.tensor([np.array([0, y[i], 0, 0, w[j], 0, 0, 0])], dtype=torch.float32)
            mean, _ = actor.forward(state)
            mean = mean[0].cpu().detach().numpy()[1]
            direction = 1 if mean > 0 else -1
            M[j,i] = direction
    ax.plot_surface(Y, W, M, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Height')
    ax.set_ylabel('Lander angle')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

def running_average(x, N):
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def plot_agent_rewards(actor):

    episode_reward_list = run_sim()

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, 51)], episode_reward_list, label='Random Agent episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_ylim(-500,400)
    ax[0].set_title('Random Agent episode reward')
    ax[0].legend()
    ax[0].grid(alpha=0.3)


    episode_reward_list = run_sim(actor)

    ax[1].plot([i for i in range(1, 51)], episode_reward_list, label='PPO Agent episode reward')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_ylim(-500,400)
    ax[1].set_title('PPO Agent episode reward')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()

def run_sim(agent=None):
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Parameters
    N_episodes = 50                              # Number of episodes
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = len(env.action_space.high)       # Action dimension
    dim_state = len(env.observation_space.high)  # State dimensionality

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode

    if agent is None:
        actor = RandomAgent(n_actions)
    else:
        actor = agent
    for i in range(50):
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Choose action at random.
            if agent is None:
                action = actor.forward(state)
            else:
                mean, variance = actor.forward(torch.tensor([state], dtype=torch.float32))
                mean = mean[0].cpu().detach().numpy()
                variance = variance[0].cpu().detach().numpy()
                action = np.clip(np.random.multivariate_normal(mean, np.diag(variance)), -1, 1)

            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()
    return episode_reward_list


def main():
    path = Path(__file__).resolve().parent
    actor = torch.load(path.joinpath('neural-network-3-actor.pth'))
    critic = torch.load(path.joinpath('neural-network-3-critic.pth'))
    plot_value_function(critic)
    plot_policy(actor)
    plot_agent_rewards(actor)

if __name__ == '__main__':
    main()
