#pylint: disable = wildcard-import, method-hidden
#pylint: disable = too-many-lines

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Replay memory:
'''
- This stores the transitions that the agent observes, therefore allowing us to 
reuse the data later.
- Sampling also happens randomly --> allows transition to be de-correlated
- 2 classes
    - transition
    - replaymemory
'''
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class D_QN(nn.Module):
    def __init__(self, n_obs, n_act):
        super(D_QN, self).__init__()
        '''
        - 3 layers mapping from the observations to the actions to be taken
        - Basically input layers will have dimensions of observations
        - Output layer willhave dimensions of actions
        '''
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_act)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
