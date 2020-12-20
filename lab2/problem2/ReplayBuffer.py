"""!@brief Lab 2, Problem 2 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 2 experience replay buffer file.
@author Martin Schuck, Damian Valle
@date 14.12.2020
"""

import numpy as np
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, maximum_length=1000, combine=False):
        self.buffer = deque(maxlen=maximum_length)
        self.last_xp = None
        self.combine = combine

    def append(self, experience):
        self.buffer.append(experience)
        self.last_xp = experience

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            print('Error! Asked to retrieve too many elements from the buffer')

        indices = np.random.choice(len(self.buffer), n, replace=False)

        batch = [self.buffer[i] for i in indices]
        if self.combine and batch and self.last_xp:
            batch[0] = self.last_xp

        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
