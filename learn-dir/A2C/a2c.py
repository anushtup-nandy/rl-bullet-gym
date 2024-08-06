import os
import gymnasium as gym
import panda_gym
from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

env_id = "PandaReachDense-v3"

env = gym.make(env_id)

#state and action space sizes
s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) 

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

'''
It is standard and good practice to normalize action space (the input features)/
Usually done when the action space is continuous.

Vectorized Environments are a method for stacking multiple independent environments 
into a single environment. Instead of training an RL agent on 1 environment per step, 
it allows us to train it on n environments per step
'''
env = make_vec_env(env_id, n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
model = A2C(policy = "MultiInputPolicy",
            env = env,
            verbose = 1)

model.learn(1_000_000) #1000000 timesteps

model.save("a2c-PandaReachDense-v3")
env.save("vec-normalize.pkl")

#evaluation:
eval_env = DummyVecEnv([lambda : gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
# eval_env.render_mode = "rgb_array"
eval_env.render_mode = "human"

eval_env.training = False
eval_env.norm_reward = False #no need to normalize testing rewards

model = A2C.load("a2c-PandaReachDense-v3")

mean_rew, std_rew = evaluate_policy(model, eval_env)
print(f"Mean reward = {mean_rew:.2f} +/- {std_rew:.2f}")




