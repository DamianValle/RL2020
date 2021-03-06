"""!@brief Lab 2, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 multiprocessing training file.
@author Martin Schuck, Damian Valle
@date 18.12.2020
"""

from multiprocessing import Process, TimeoutError, Queue, Manager
import queue
import time
import os
import sys
import numpy as np
from pathlib import Path
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from Agent import Actor, Critic
from ReplayBuffer import ExperienceReplayBuffer
import json


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

def fill_buffer(buffer,m,L,env):
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
    return buffer

def clipping_function(x, epsilon):
    return torch.maximum(torch.ones_like(x)-epsilon, torch.min(x,torch.ones_like(x) + epsilon))

def gaussian_pdf(actions, mean, variance):
    pi_1 = (1/torch.sqrt(2*np.pi*variance[:,0]))*torch.exp(-(actions[:,0]-mean[:,0])**2/(2*variance[:,0]))
    pi_2 = (1/torch.sqrt(2*np.pi*variance[:,1]))*torch.exp(-(actions[:,1]-mean[:,1])**2/(2*variance[:,1]))
    return pi_1*pi_2


def run_training(result_queue):
    # sys.stdout = open(str(os.getpid()) + ".out", "w")
    # sys.stderr = open(str(os.getpid()) + ".err_out", "w")  # Use for debugging.
    idx = 0
    while result_queue.empty():
        print(f"Thread {os.getpid()} working on run {idx}", flush=True)
        idx += 1
        save = False
        # Import and initialize Mountain Car Environment
        env = gym.make('LunarLanderContinuous-v2')
        env.reset()

        # Parameters
        N_episodes = 4000                               # Number of episodes to run for training
        discount_factor = 0.99                          # Value of gamma
        n_ep_running_average = 50                       # Running average of 20 episodes
        m = len(env.action_space.high)                  # dimensionality of the action
        dim_state = len(env.observation_space.high)     # State dimensionality
        training_epochs = 10                            # Number of successive training epochs
        epsilon = 0.2                                   # Limiting parameter of the objective function (1-epsilon, 1+epsilon)
        alpha_actor = 1e-5                              # Learning rate actor
        alpha_critic = 1e-3                             # Learning rate critic

        # Reward
        episode_reward_list = []                        # Used to save episodes reward
        episode_number_of_steps = []

        # Agent initialization
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actor = Actor(dim_state, m).to(device=dev)
        critic = Critic(dim_state).to(device=dev)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=alpha_actor)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=alpha_critic)

        buffer = ExperienceReplayBuffer()

        # Training process
        t_episodes = 0
        for i in range(N_episodes):
            # Reset enviroment data
            done = False
            state = env.reset()
            total_episode_reward = 0.
            t = 0
            while not done:
                mean, variance = actor.forward(torch.tensor([state]).to(device=dev))
                mean = mean[0].cpu().detach().numpy()
                variance = variance[0].cpu().detach().numpy()
                # Sample actions from means, variances
                action = np.clip(np.random.multivariate_normal(mean, np.diag(variance)),-1,1)
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                # Update episode reward
                total_episode_reward += reward
                # Update state for next iteration
                state = next_state
                t+= 1

            # Training procedure
            states, actions, rewards, next_states, dones = buffer.buffer
            np_gamma = np.flip(np.array([discount_factor**t for t in range(len(rewards))]))
            g_t = torch.tensor(np.convolve(np_gamma,rewards)[len(rewards)-1:], dtype=torch.float32).to(device=dev)
            states_tensor = torch.tensor(states).to(device=dev)
            actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device=dev)
            with torch.no_grad():
                mean, variance = actor.forward(states_tensor)
                pi_theta_old = gaussian_pdf(actions_tensor, mean, variance)
                advantage = g_t - critic.forward(torch.tensor(states).to(device=dev)).squeeze()
            for n in range(training_epochs):
                critic_optimizer.zero_grad()
                state_values = critic.forward(torch.tensor(states).to(device=dev)).squeeze()
                loss = torch.nn.functional.mse_loss(state_values, g_t)
                loss.backward()
                critic_optimizer.step()
                
                actor_optimizer.zero_grad()
                mean, variance = actor.forward(states_tensor)
                pi_theta = gaussian_pdf(actions_tensor, mean, variance)
                loss = -torch.mean(torch.minimum((pi_theta/pi_theta_old)*advantage,clipping_function(pi_theta/pi_theta_old, epsilon)*advantage))
                loss.backward()
                actor_optimizer.step()
            buffer.clear()
            # Append episode reward
            episode_reward_list.append(total_episode_reward)
            episode_number_of_steps.append(t)

            # Close environment
            env.close()
            avg = running_average(episode_reward_list, n_ep_running_average)
            t_episodes += 1
            if i%40 == 0:
                print("Thread {} at {:.0f}%. Avg. Reward: {}".format(os.getpid(), i*100/N_episodes, avg[-1]), flush=True)

            if avg[-1] > 200:
                print('Training success! Saving and aborting other threads.', flush=True)
                save = True
                result_queue.put(True)
            if avg[-1] > 230:
                break
        if save:
            path = Path(__file__).resolve().parent.joinpath(str(os.getpid())+'_plot.png')
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
            ax[0].plot([i for i in range(1, t_episodes+1)], episode_reward_list, label='Episode reward')
            ax[0].plot([i for i in range(1, t_episodes+1)], running_average(
                episode_reward_list, n_ep_running_average), label='Avg. episode reward')
            ax[0].set_xlabel('Episodes')
            ax[0].set_ylabel('Total reward')
            ax[0].set_title('Total Reward vs Episodes')
            ax[0].legend()
            ax[0].grid(alpha=0.3)

            ax[1].plot([i for i in range(1, t_episodes+1)], episode_number_of_steps, label='Steps per episode')
            ax[1].plot([i for i in range(1, t_episodes+1)], running_average(
                episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
            ax[1].set_xlabel('Episodes')
            ax[1].set_ylabel('Total number of steps')
            ax[1].set_title('Total number of steps vs Episodes')
            ax[1].legend()
            ax[1].grid(alpha=0.3)
            plt.savefig(path)

            #torch.save(actor.to('cpu'),path.joinpath(str(os.getpid())+'_'+str(idx)+'_actor.pth'))
            #torch.save(critic.to('cpu'),path.joinpath(str(os.getpid())+'_'+str(idx)+'_critic.pth'))

    print('Successful training detected! Returning from all tasks.', flush=True)
    return

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

def main():
    t1 = time.time()
    mananger = Manager()
    work_queue = Queue()
    result_queue = mananger.Queue()  # Avoid deadlocking on joining processes due to open Pipes to result_queue.
    processes = []

    for _ in range(6):
        p = Process(target=run_training, args=(result_queue,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    t2 = time.time()
    print("Total run time: {:.0f}h {:.0f}m {:.0f}s".format((t2-t1)//3600, ((t2-t1)%3600)//60,(t2-t1)%60))

if __name__ == '__main__':
    main()