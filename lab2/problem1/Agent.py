"""!@brief Lab 2, Problem 1 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 2 agent file.
@author Martin Schuck, Damian Valle
@date 10.12.2020
"""

import torch.nn as nn
import torch.optim as optim
import torch

class DeepAgent(nn.Module):
    
    def __init__(self, input_size, l1_size, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(input_size,l1_size)
        self.linear2 = nn.Linear(l1_size,l2_size)
        self.output = nn.Linear(l2_size,n_actions)
        
    def forward(self, state):
        l1out = torch.relu(self.linear1(state))
        l2out = torch.relu(self.linear2(l1out))
        return self.output(l2out)  # No activation function for the output.

class AdvantageAgent(nn.Module):

    def __init__(self, input_size, l1_size, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(input_size,l1_size)
        self.v_layer = nn.Linear(l1_size,1)
        self.a_layer = nn.Linear(l1_size,n_actions)
        
    def forward(self, state):
        l1out = nn.functional.relu(self.linear1(state))
        v_out = self.v_layer(l1out)
        a_out = self.a_layer(l1out)
        return a_out + v_out - torch.mean(a_out, dim=1, keepdim=True)  # Enforce advantage function structure in the network.

