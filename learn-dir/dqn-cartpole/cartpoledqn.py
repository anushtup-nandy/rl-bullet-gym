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

from DQN import ReplayMemory, D_QN, Transition

# Build Environment
env = gym.make("CartPole-v1")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128 #number of transitions 
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

#policy, target and optimizer and memory
policyNet = D_QN(n_observations, n_actions).to(device)
targetNet = D_QN(n_observations, n_actions).to(device)
targetNet.load_state_dict(policyNet.state_dict())
optimizer = optim.AdamW(policyNet.parameters(), lr = LR, amsgrad = True)
memory = ReplayMemory(10000)

#steps
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1*steps_done/EPS_DECAY)
    steps_done+=1
    if (sample > eps_threshold):
        with torch.no_grad():
            return policyNet(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Training 
def optimize_model():
    if (len(memory) < BATCH_SIZE):
        return
    transitions = memory.sample(BATCH_SIZE)
    #convert batch-array of transitions into transition of batch-arrays
    batch = Transition(*zip(*transitions))
    
    #compute masks:
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype = torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    '''
    - Computing V_s{t+1}
    - selecting best reward with max(1)
    '''
    state_action_values = policyNet(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = targetNet(non_final_next_state).max(1).values
    #compute E[Q]
    expected_state_action_values = (next_state_values)*GAMMA + reward_batch
    #HUBER loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = targetNet.state_dict()
        policy_net_state_dict = policyNet.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        targetNet.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.tight_layout()
plt.show()
plt.tight_layout()
plt.savefig("training_statistics.png")
plt.show()
