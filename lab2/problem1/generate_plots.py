"""!@brief Lab 2, Problem 1 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 1 plots file.
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


def plot_value_function(q_net):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Value Function')
    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    Y, W = np.meshgrid(y, w)
    M = np.zeros_like(Y)
    for i in range(M.shape[0]): 
        for j in range(M.shape[1]):
            state = np.array([0, y[i], 0, 0, w[j], 0, 0, 0])
            action_values = q_net.forward(torch.tensor([state], dtype=torch.float32))
            M[j,i] = torch.max(action_values).item()
    ax.plot_surface(Y, W, M, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Height')
    ax.set_ylabel('Lander angle')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

def plot_policy(q_net):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Policy. nothing=0, left=1, main=2, right=3')
    y = np.linspace(0, 1.5, 300)
    w = np.linspace(-np.pi, np.pi, 300)
    Y, W = np.meshgrid(y, w)
    M = np.zeros_like(Y)
    for i in range(M.shape[0]): 
        for j in range(M.shape[1]):
            state = np.array([0, y[i], 0, 0, w[j], 0, 0, 0])
            M[j,i] = torch.argmax(q_net.forward(torch.tensor([state], dtype=torch.float32))).item()
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

def plot_agent_rewards(q_net):

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


    episode_reward_list = run_sim(q_net)

    ax[1].plot([i for i in range(1, 51)], episode_reward_list, label='DQN Agent episode reward')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_ylim(-500,400)
    ax[1].set_title('DQN Agent episode reward')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()

def run_sim(agent=None):
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    N_episodes = 50                             # Number of episodes
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode

    for i in range(50):
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Choose action at random.
            if agent is None:
                action = np.random.randint(0, n_actions)
            else:
                action = torch.argmax(agent.forward(torch.tensor([state], requires_grad=False))).item()

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
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
    path = Path(__file__).resolve().parent.joinpath('neural-network-1.pth')
    q_net = torch.load(path)
    #plot_value_function(q_net)
    plot_policy(q_net)
    #plot_agent_rewards(q_net)

if __name__ == '__main__':
    main()
