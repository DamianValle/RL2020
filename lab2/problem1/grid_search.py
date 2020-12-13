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
from Agent import DeepAgent, AdvantageAgent
from ReplayBuffer import ExperienceReplayBuffer
import json


def run_training(work_queue, result_queue):
    # sys.stdout = open(str(os.getpid()) + ".out", "w")
    # sys.stderr = open(str(os.getpid()) + ".err_out", "w")  # Use for debugging.
    idx = 0
    while not work_queue.empty():
        print(f"Thread {os.getpid()} working on task {idx}", flush=True)
        idx += 1
        params = None
        try:
            params = work_queue.get(True, 1)
        except queue.Empty:
            print(f"Thread {os.getpid()} returning from all tasks.", flush=True)
            return
        except Exception as e:
            print(e, flush=True)
            return
        env = gym.make('LunarLander-v2')
        env.reset()

        # Parameters
        N_episodes = 200                             # Number of episodes
        n_actions = env.action_space.n               # Number of available actions
        dim_state = len(env.observation_space.high)  # State dimensionality

        alpha, epsilon_max, epsilon_min, clipping_value, L, N, discount_factor, N_episodes, l1_size, l2_size = params

        # Training parameters
        Z = 0.9*N_episodes

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Random agent, target, buffer and optimizer initialization
        Q1 = AdvantageAgent(dim_state, l1_size, n_actions).to(device=dev)
        Q2 = AdvantageAgent(dim_state, l1_size, n_actions).to(device=dev)
        buffer = ExperienceReplayBuffer(maximum_length=L, combine=True)
        optimizer1 = optim.Adam(Q1.parameters(), lr=alpha)
        optimizer2 = optim.Adam(Q2.parameters(), lr=alpha)

        ### Training process
        # trange is an alternative to range in python, from the tqdm library
        # It shows a nice progression bar that you can update with useful information
        for i in range(N_episodes):
            if not i%25:
                print("Thread {} task {} at {:.2f}%".format(os.getpid(), idx, (i)*100/(N_episodes+50)), flush=True)
            # Reset enviroment data and initialize variables
            done = False
            state = env.reset()
            while not done:
                # Choose action epsilon greedy with epsilon decay.
                epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**((i)/(Z)))
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

                # Update state for next iteration
                state = next_state

            # Close environment
            env.close()
        agent = Q1
        env.reset()
        # Parameters
        N_EPISODES = 50            # Number of episodes to run for trainings
        CONFIDENCE_PASS = 50
        model = agent.to('cpu')
        # Reward
        episode_reward_list = []  # Used to store episodes reward

        # Simulate episodes
        for i in range(N_EPISODES):
            if not i%25:
                print("Thread {} task {} at {:.2f}%".format(os.getpid(), idx, (i+N_episodes)*100/(N_episodes+50)), flush=True)

            # Reset enviroment data
            done = False
            state = env.reset()
            total_episode_reward = 0.
            while not done:
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                q_values = model(torch.tensor([state]))
                _, action = torch.max(q_values, axis=1)
                next_state, reward, done, _ = env.step(action.item())

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state

            # Append episode reward
            episode_reward_list.append(total_episode_reward)

            # Close environment
            env.close()

        avg_reward = np.mean(episode_reward_list)
        confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

        result_queue.put((avg_reward, confidence, params))
    print(f"Thread {os.getpid()} returning from all tasks.", flush=True)
    return

def load_tasks(work_queue):
    x = 0
    for alpha in [1e-3]:
        for epsilon_max in [0.6]:
            for epsilon_min in [0.2]:
                for clipping_value in [1.3]:
                    L = int(2e4)  # 2e4
                    N = 100  # 64
                    discount_factor = 0.7
                    for N_episodes in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
                        for l1_size in [16]:
                            for l2_size in [8]:
                                work_queue.put([alpha, epsilon_max, epsilon_min, clipping_value, L, N, discount_factor, N_episodes, l1_size, l2_size])
                                x += 1
    print(f'Loaded {x} tasks.')

def main():

    t1 = time.time()
    mananger = Manager()
    work_queue = Queue()
    result_queue = mananger.Queue()  # Avoid deadlocking on joining processes due to open Pipes to result_queue.
    processes = []
    load_tasks(work_queue)
    try:
        with open(Path(__file__).resolve().parent.joinpath('best_param.json'), 'r') as f:
            old_rslt = json.load(f)
        best_err = old_rslt['best_err']
        best_param = old_rslt['best_param']
        print(f"Best old result: {old_rslt}", flush=True)
    except FileNotFoundError:
        best_err = -np.Inf
        best_param = None

    # params = [alpha, epsilon_max, epsilon_min, clipping_value, L, N, discount_factor]
    for _ in range(8):
        p = Process(target=run_training, args=(work_queue, result_queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    x = 0
    while not result_queue.empty():
        result = result_queue.get()
        print(result)
        if result[0] - result[1] > best_err:
            best_err = result[0] - result[1]
            best_param = result[2]
        x += 1
    print('\n')
    t2 = time.time()
    print("Total run time: {:.0f}h {:.0f}m {:.0f}s".format((t2-t1)//3600, ((t2-t1)%3600)//60,(t2-t1)%60))
    print(f"Total searched parameters: {x}")
    print(f"Best error: {best_err}, Best parameters: {best_param}")
    best_rslt = {'best_err': best_err, 'best_param': best_param}
    with open(Path(__file__).resolve().parent.joinpath('best_param.json'), 'w') as f:
        json.dump(best_rslt, f)



if __name__ == '__main__':
    main()