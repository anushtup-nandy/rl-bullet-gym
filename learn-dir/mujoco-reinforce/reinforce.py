import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.distributions.normal as normal

import gymnasium as gym

plt.rcParams["figure.figsize"] = (10, 5)

class PolicyNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNet, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.fc1 = nn.Linear(obs_space, 16)
        self.fc2 = nn.Linear(16, 32)
        self.p_means = nn.Linear(32, action_space)
        self.p_stddevs = nn.Linear(32, action_space)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.fc1(x.float()))
        x = torch.tanh(self.fc2(x))

        action_means = self.p_means(x)
        action_stddevs = torch.log(1 + torch.exp(self.p_stddevs(x)))
        
        return action_means, action_stddevs

class Reinforce:
    def __init__(self, obs_space, action_space):
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        self.probs = [] #probability of sampled action
        self.rewards = [] #corresponding rewards

        self.net = PolicyNet(obs_space, action_space)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = self.learning_rate)
    
    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs =self.net(state) 


