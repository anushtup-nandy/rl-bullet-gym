import torch
import torch.nn as nn
import torch.optim as optim

'''
Actor:
- outputs actions given states
'''
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_max):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(s_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, a_dim)
        self.max_action = a_max
    
    def forward(self, x):
        a = torch.relu(self.layer1(x))
        a = torch.relu(self.layer2(a))
        
        return (self.max_action*torch.tanh(self.layer3(a)))

'''
Critic:
- Estimates Q-values given state-action pairs
- state dimension + action dimension : we use state_dim + action_dim as the input size for the 
                                       first layer. This is because the Critic in DDPG is designed 
                                       to estimate the Q-value (expected cumulative reward) for a given 
                                       state-action pair
- torch.cat : state and action tensors along dimension 1 (the feature dimension), 
              creating a single input tensor that contains both state and action information.
'''
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(s_dim + a_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], 1) 
        q = torch.relu(self.layer1(q))
        q = torch.relu(self.layer2(q))
        return self.layer3(q)