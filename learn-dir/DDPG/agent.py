import torch
import torch.nn as nn
import torch.optim as optim
from ActorCritic import Actor, Critic
from memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DDPG:
    def __init__(self, s_dim, a_dim, a_max):
        self.actor = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_target = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = 1e-3)

        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = 1e-3)
       
        self.replay_buffer = ReplayBuffer(1_000_000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001


    '''
    - method uses the actor network to select an action given a state
    - state is converted to a PyTorch tensor and reshaped
    - actor network outputs an action, which is then converted back to a numpy array
    '''
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self):
        if (len(self.replay_buffer) < self.batch_size):
            return
        
        #sample from memory:
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

        #compute Q
        target_Q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

        #get Q estimate
        current_Q = self.critic(state_batch, action_batch)

        #compute loss
        criticLoss = nn.MSELoss()(current_Q, target_Q.detach())

        #optimize the critic
        self.critic_optim.zero_grad()
        criticLoss.backward()
        self.critic_optim.step()

        #actor loss
        actorLoss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor
        self.actor_optim.zero_grad()
        actorLoss.backward()
        self.actor_optim.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
