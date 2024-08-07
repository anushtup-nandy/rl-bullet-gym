from collections import deque
import random

'''
Replay buffer (similar to DQN)
'''
class ReplayBuffer:
    def __init__(self, memory):
        self.buffer = deque(maxlen=memory)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
