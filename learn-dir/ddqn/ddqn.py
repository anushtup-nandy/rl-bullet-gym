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
        # print("STORE - INDEX - ", self.index)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = int((self.index + 1) % self.capacity)
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace = False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            # torch.tensor(np.array(states), dtype=torch.float32),
            # torch.tensor(np.array(actions), dtype=torch.int64),
            # torch.tensor(np.array(rewards), dtype=torch.float32),
            # torch.tensor(np.array(next_states), dtype=torch.float32),
            # torch.tensor(np.array(dones), dtype=torch.float32)
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).long(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int()
        )
    def __len__(self):
        return len(self.buffer)
    
#QNet:
class Q(nn.Module):
    def __init__(self, nS, nA):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(nS, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, nA)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#DDQN agent:
class DDQN:
    def __init__(self, state_size, action_size, seed, l_r = 1e-3, capacity =1e6, d_f = 0.99, tau = 1e-3, update_every = 4, batch_size = 64):
        self.state_size = state_size 
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = l_r
        self.discount_factor = d_f
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.q_1 = Q(state_size, action_size).to(self.device)
        self.q_2 = Q(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_1.parameters(), lr = l_r)
        self.replay_buffer = ReplayBuffer(capacity)
        self.update_target_net()

    def step(self, s, a, r, ns, d):
        self.replay_buffer.store(s, a, r, ns, d)
        # print("STEP - ", self.steps, "(s, a, r, ns, d) : ", s, a, r, ns, d)
        self.steps += 1
        if (self.steps % self.update_every == 0):
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def interact(self, state, eps = 0.0):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # print("INTERACT - STATE - ", state)
        self.q_1.eval() 
        with torch.no_grad():
            action_values = self.q_1(state)
            # print("INTERACT - ACTION_VAL - ", action_values)
        self.q_1.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        s, a, r, ns, d = experiences

        # Move experiences to the correct device
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)

        # print("LEARN - NEXT_STATE - ", ns)
        #max predicted Q values from q2
        q_targets_next = self.q_2(ns).detach().max(1)[0].unsqueeze(1)
        # print("LEARN - Q_TARGETS_NEXT - ", q_targets_next)

        #q-targets for q1
        q_targets = r + self.discount_factor*(q_targets_next*(1-d))
        # print("LEARN - Q_TARGETS - ", q_targets)
        #expected q from q1
        q_expected = self.q_1(s).gather(1, a.view(-1, 1))
        # print("LEARN - Q_EXP - ", q_expected)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.q_1, self.q_2)

    def update_target_net(self):
        for q2_param, q1_param in zip(self.q_2.parameters(), self.q_1.parameters()):
            # print("HARD UPDATE - Q1 - ", q1_param)
            # print("HARD UPDATE - Q2 - ", q2_param)
            q1_param.data.copy_(self.tau*q1_param.data + (1 - self.tau)*q2_param.data)

    def soft_update(self, q1, q2):
        for q2_param, q1_param in zip(q2.parameters(), q1.parameters()):
            q2_param.data.copy_(self.tau*q1_param + (1-self.tau)*q2_param)