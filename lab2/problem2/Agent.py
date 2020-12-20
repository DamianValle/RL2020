"""!@brief Lab 2, Problem 2 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 2 agent file.
@author Martin Schuck, Damian Valle
@date 14.12.2020
"""

import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    """Actor network"""
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(input_size,400)
        self.linear2 = nn.Linear(400,200)
        self.output = nn.Linear(200,n_actions)
        
    def forward(self, state):
        l1out = torch.relu(self.linear1(state))
        l2out = torch.relu(self.linear2(l1out))
        return torch.tanh(self.output(l2out))

class Critic(nn.Module):
    """Critic network"""
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,400)
        self.linear2 = nn.Linear(402,200)  # We concat 2 actions to l1 output.
        self.output = nn.Linear(200,1)
        
    def forward(self, states, actions):
        l1out = torch.relu(self.linear1(states))
        concat_stage = torch.cat([l1out, actions], dim=1)
        l2out = torch.relu(self.linear2(concat_stage))
        return self.output(l2out)
