"""!@brief Lab 2, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 agent file.
@author Martin Schuck, Damian Valle
@date 17.12.2020
"""

import torch.nn as nn
import torch


class Actor(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 400)
        self.mean_layer1 = nn.Linear(400, 200)
        self.mean_layer2 = nn.Linear(200,n_actions)
        self.var_layer1 = nn.Linear(400, 200)
        self.var_layer2 = nn.Linear(200,n_actions)

    def forward(self, state):
        out1 = torch.relu(self.layer1(state))
        mean1 = torch.relu(self.mean_layer1(out1))
        mean2 = torch.tanh(self.mean_layer2(mean1))
        sigma1 = torch.relu(self.var_layer1(out1))
        sigma2 = torch.sigmoid(self.var_layer2(sigma1))
        return mean2, sigma2


class Critic(nn.Module):

    def __init__(self, n_states):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 400)
        self.layer2 = nn.Linear(400, 200)
        self.layer3 = nn.Linear(200,1)

    def forward(self, state):
        out1 = torch.relu(self.layer1(state))
        out2 = torch.relu(self.layer2(out1))
        output = self.layer3(out2)
        return output