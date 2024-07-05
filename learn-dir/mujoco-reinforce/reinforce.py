import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
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
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = self.learning_rate) #weight decay using some default 1e-2 
                                                                                            # instead of L2 loss
    
    def sample_action(self, state: np.ndarray) -> float:
        '''
        Returns an action to be performed
        1. take the stateas the input
        2. convert it to a tensor
        3. draw a distribution of action probabilities
        4. sample an action from that.

        '''
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs =self.net(state)
        
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps) # mean and standard deviation
        action = distrib.sample()
        prob = distrib.log_prob(action)
        action = action.numpy() #convert it to a numpy array
        self.probs.append(prob)

        return action
    
    def update(self): #update "theta" or the weights
        running_g = 0
        gs = []
        #discounted returns:
        for r in self.rewards[::-1]: #returns the array in reverse
            running_g = r + self.gamma*running_g
            gs.insert(0, running_g)
            
        deltas = torch.tensor(gs)
        loss = 0
        for logprob, delta in zip(self.probs, deltas):
            loss += logprob.mean()*delta*(-1) # this is where we are doing gradient ascent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #reset probabilities and rewards for next run
        self.probs = []
        self.rewards = []


#training
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50) #record every 50 episodes

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = Reinforce(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

#plot:
rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
plt.savefig("./training-reinforce.png")