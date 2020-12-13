import numpy as np
import gym
import torch
from pathlib import Path

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = torch.load(Path(__file__).resolve().parent.joinpath("neural-network-1.pth")).to(device=dev)

env = gym.make('LunarLander-v2')
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
