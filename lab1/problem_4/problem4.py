# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
from pathlib import Path
import gym
from numpy.core.fromnumeric import argmax
import torch
import matplotlib.pyplot as plt
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 1000        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma

# Reward
episode_reward_list = []  # Used to save episodes reward

# Basis and weights.
nu = np.array([[0,0], [0,1], [1,0], [1,1], [1,2], [2,0], [2,2]]).T  # [2,1], [0,2] not included.
weights = np.random.randn(nu.shape[1],3)/10  # Initialize weights randomly with small values and variance.

z = np.zeros_like(weights)
v = np.zeros_like(weights)
alpha_reduce = 1

alpha_base = 0.01/np.maximum(np.linalg.norm(nu, axis=0), 1).reshape(-1,1)  # Normalized learning rate.
epsilon_base = 0.3
gamma = 0.99
lambda_z = 0.75  # Initial guess.
m = 0.5

def q_fct(state):
    return weights.T@np.cos(np.pi*nu.T@state)

def grad_Q(s,a):
    grad = np.zeros_like(z)
    grad[:,a] = np.cos(np.pi*nu.T@s)  # Grad_wa Q(s,a) = phi(s) (at column a).
    return grad

def sgd_nesterov(delta,v, z):
    v = m*v + alpha*delta*z
    return m*v + alpha*delta*z, v

def sgt_momentum(delta, z, v):
    v = m*v + alpha*delta*z
    return 

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def running_average_window(x, N):
    return sum(x[-N:])/N

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

# Weights history
w_hist = list()
w_hist.append(weights.copy())
max_state = 0
# Training process
for i in range(N_episodes):
    # Reset enviroment data
    done = False
    z[:,:] = 0
    v[:,:] = 0
    epsilon = epsilon_base*(1 - (i//(N_episodes/10))/10)
    state = scale_state_variables(env.reset())
    total_episode_reward = 0.
    # Employ epsilon greedy strategy
    action = np.random.randint(0, k) if np.random.rand() < epsilon else np.argmax(q_fct(state))
    idx = 1
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        idx += 1
        if done and idx < 200:
            print('######')
            print(total_episode_reward)
        if episode_reward_list:
            if episode_reward_list[-1] > -100:
                alpha_reduce = min(0.1, alpha_reduce)
            elif episode_reward_list[-1] > -140:
                alpha_reduce = min(0.5, alpha_reduce)
            elif episode_reward_list[-1] > -150:
                alpha_reduce = min(0.7, alpha_reduce)
            else:
                alpha_reduce = min(1, alpha_reduce)
        alpha = alpha_base*alpha_reduce
        # Employ epsilon greedy strategy
        next_action = np.random.randint(0, k) if np.random.rand() < epsilon else np.argmax(q_fct(next_state))
        
        delta = reward + gamma*q_fct(next_state)[next_action] - q_fct(state)[action]
        z = np.clip(gamma*lambda_z*z + grad_Q(state,action), -5,5)  # clip eligibility traces.

        # w_update, v = sgd_nesterov(delta, v, z)
        w_update = alpha * delta * z
        # w_update, v = sgt_momentum(delta, z, v)
        weights += w_update
        
        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state
        action = next_action
        
    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    w_hist.append(weights.copy())
    # Close environment
    env.close()

analyze = False

def plot_q_function(plot_policy=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Combined Q function')
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((*X.shape, 3))
    M = np.zeros_like(X)
    for i in range(Z.shape[0]): 
        for j in range(Z.shape[1]):
            state = np.array([x[i], y[j]])
            q_values = q_fct(state)
            for k in range(3):
                Z[j,i,k] = q_values[k]
            M[j,i] = np.argmax(q_values)
    color_dict = {0: 'b', 1: 'gray', 2: 'r'}
    for k in range(3):
        ax.plot_surface(X, Y, Z[:,:,k], color=color_dict[k], linewidth=0, antialiased=False)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    if plot_policy:
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection='3d')
        ax2.set_title('policy')
        surf = ax2.plot_surface(X,Y,M,cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.set_xlabel('position')
        ax2.set_ylabel('velocity')
        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

if analyze:
    plot_q_function(plot_policy=True)

    state = scale_state_variables(env.reset())
    done=False
    idx = 0
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        idx += 1
        action = np.argmax(q_fct(state))
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        env.render()
        time.sleep(0.002)
        state = next_state
    print(idx)


# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

data = {'W': weights.T, 'N': nu.T}

path = Path(__file__).resolve().parent.joinpath('weights.pkl')
pickle.dump(data, open(path, "wb"))
