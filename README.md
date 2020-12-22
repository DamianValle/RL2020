# Reinforcement Learning

This repository provides a collection of both theoretical problems as homeworks and python implementations as labs.

## Lab 1
### **The minotaur maze**

The agent tries to find the exit of the maze while escaping a minotaur following a random walk. We provide the problem formulation as an MDP and its [solution](https://github.com/DamianValle/RL2020/blob/main/lab1/problem_1/problem_1.ipynb).

<p align="center"><img src="/img/minotaur.png" width="500"/></p>

### **Robbing banks**

The agent tries to maximize its reward by staying as much as possible inside a bank but avoiding being captured by the police. We provide the problem formulation as an MDP and its [solution](https://github.com/DamianValle/RL2020/blob/main/lab1/problem_2/problem_2.ipynb).

<p align="center"><img src="/img/banks.png" width="500"/></p>

### **Mountain Car with linear function approximators**

We solve the [OpenAI Mountain Car problem](https://gym.openai.com/envs/MountainCar-v0/) using linear function approximators. We provide the [solution](https://github.com/DamianValle/RL2020/blob/main/lab1/problem_4/problem4.py) and a set of trained weights that solve the problem.

<p align="center"><img src="/img/hill.png" width="500"/></p>

## Lab 2

This lab focuses on the [OpenAI Lunar Lander problem](https://gym.openai.com/envs/LunarLander-v2/) for both the discrete and continuous action spaces.

<p align="center"><img src="/img/lander.png" width="500"/></p>

### **Deep Q-Networks (DQN)**

We implement the DQN algorithm with some modifications (Dueling DQN and combined experience replay buffer) and train it to [solve the problem](https://github.com/DamianValle/RL2020/blob/main/lab2/problem1/problem_1.py).

### **Deep Deterministic Policy Gradient (DDPG)**

We implement the DDPG algorithm, train our model and [solve the problem](https://github.com/DamianValle/RL2020/blob/main/lab2/problem2/problem_2.py).

### **Proximal Policy Optimization (PPO)**

We implement the PPO algorithm, train our model and [solve the problem](https://github.com/DamianValle/RL2020/blob/main/lab2/problem3/problem_3.py).
