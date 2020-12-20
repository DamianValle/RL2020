"""!@brief Lab 2, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 experience replay buffer file.
@author Martin Schuck, Damian Valle
@date 17.12.2020
"""

import numpy as np
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, maximum_length=1000):
        self._buffer = deque(maxlen=maximum_length)

    @property
    def buffer(self):
        states, actions, rewards, next_states, dones = zip(*list(self._buffer))
        return states, actions, rewards, next_states, dones

    def append(self, experience):
        self._buffer.append(experience)

    def __len__(self):
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
