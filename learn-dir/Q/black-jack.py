from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as Patch

import gymnasium as gym

# creating the env:
env = gym.make("Blackjack-v1", sab = True) #following rules from Sutton and Barto

# reading first observation from env
done = False
observation, info = env.reset() #something like (21, 2, 1)-->tuple[int, int, bool]

action = env.action_space.sample() # 0 or 1 

observation, reward, terminated, truncated, info = env.step(action)

'''
Building an agent
'''
class BlackJackAgent:
    def __init__(self, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.final_epsilon = final_epsilon
        self.training_error = []

    def action(self, obs: tuple[int, int, bool]) -> int:
        # explore with e-greedy
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        # exploit with probability (1-epsilon)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs:tuple[int, int, bool], action:int, reward: float, terminated: bool,  
            next_obs: tuple[int, int, bool]):
        future_q = (not terminated)*np.max(self.q_values[next_obs])
        td = (reward + self.discount_factor * future_q - self.q_values[obs][action])
        self.q_values[obs][action] = ((self.q_values[obs][action]) + self.lr*(td))
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# hyperparameters:
learning_rate = 0.01
n_ep = 100000
s_epsilon = 1.0
ep_decay = s_epsilon/(n_ep/2) # decaying
f_epsilon = 0.1

agent = BlackJackAgent(learning_rate, s_epsilon, ep_decay, f_epsilon)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = n_ep)

# Q-learning
for episode in tqdm(range(n_ep)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # updating
        agent.update(obs, action, reward, terminated, next_obs)
        # update if env is done
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()
plt.tight_layout()
plt.savefig("training_statistics.png")
plt.show()
