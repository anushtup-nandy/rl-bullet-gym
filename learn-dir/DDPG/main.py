import gymnasium as gym
from agent import DDPG
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)
episodes = 1000
max_steps = 500

episode_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        action = np.clip(action + np.random.normal(0, 0.1), -max_action, max_action)

        next_state, reward, done, _, _ = env.step(action)

        agent.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.train()

        if done:
            break

    episode_rewards.append(episode_reward)

    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title('DDPG Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('ddpg_rewards.png')
print("Reward plot saved as ddpg_rewards.png")

plt.show()

