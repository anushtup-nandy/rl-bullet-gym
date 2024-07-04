import gymnasium as gym
import numpy as np
from ddqn import DDQN
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

seed = 0
agent = DDQN(state_size, action_size, seed)

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1000
max_steps = 1000

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Set the number of episodes and the maximum number of steps per episode
num_episodes = 1000
max_steps = 1000

# Set the exploration rate
eps = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Set the rewards and scores lists
rewards = []
scores = []

# Run the training loop
for i_episode in range(num_episodes):
    print(f'Episode: {i_episode}')
    # Initialize the environment and the state
    state = env.reset()[0]
    score = 0
    # eps = eps_end + (eps_start - eps_end) * np.exp(-i_episode / eps_decay)
    # Update the exploration rate
    eps = max(eps_end, eps_decay * eps)
    
    # Run the episode
    for t in range(max_steps):
        # Select an action and take a step in the environment
        action = agent.interact(state, eps)
        next_state, reward, done, trunc, _ = env.step(action)
        print("STATE, ACTION, REWARD, NEXT_STATE, DONE - ", state, action, reward, next_state, done)
        # Store the experience in the replay buffer and learn from it
        agent.step(state, action, reward, next_state, done)
        # Update the state and the score
        state = next_state
        score += reward
        # Break the loop if the episode is done or truncated
        if done or trunc:
            break
     
    print(f"\tScore: {score}, Epsilon: {eps}")
    # Save the rewards and scores
    rewards.append(score)
    scores.append(np.mean(rewards[-100:]))

# Close the environment
env.close()

plt.ylabel("Score")
plt.xlabel("Episode")
plt.plot(range(len(rewards)), rewards)
plt.plot(range(len(rewards)), scores)
plt.legend(['Reward', "Score"])
plt.show()
plt.savefig("./ddqn/score.png")
