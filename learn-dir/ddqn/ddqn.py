import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

#Replay buffer:
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    def store(self, state, action, reward, next_state, done):
        if (len(self.buffer) < self.capacity):
            self.buffer.append(None) #basically don't append anything if limit is crossed
        self.buffer[self.index] = tuple(state, action, reward, next_state, done)
        self.index = (self.index + 1)%self.capacity
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace = False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(states)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            torch.tensor(np.array(states).float()),
            torch.tensor(np.array(actions).float()),
            torch.tensor(np.array(rewards).float()),
            torch.tensor(np.array(next_states).float()),
            torch.tensor(np.array(dones).float())
        )
    def __len__(self):
        return len(self.buffer)
    
#DDQN agent:
class DDQN:
    def __init__(self, state_size, action_size, seed, l_r = 1e-3, capacity =1e6, d_f = 0.99, tau = 1e-3, update_every = 4, batch_size = 64):
        

