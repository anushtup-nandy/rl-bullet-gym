import os
import tempfile
import random
import time
import gc
import glob
import gc #garbage collector
from IPython.display import HTML

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
from collections import namedtuple, deque
from itertools import cycle, count
from textwrap import wrap

from utils import get_gif_html, get_make_env_fn, get_videos_html
from strategies import FCQ, GreedyStrategy, EGreedyExpStrategy
import numpy as np
# import gymnasium as gym
from gym import wrappers

LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

# class ReplayBuffer():
#     def __init__(self, max_size = 10000, batch_size = 64):
#         #hold the state, action, reward, nextstate, and done flag
#         self.ss_mem = np.empty(shape = max_size, dtype = np.ndarray)
#         self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
#         self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

#         #storage and sampling
#         self.max_size = max_size
#         self.batch_size = batch_size
#         self._idx = 0
#         self.size = 0

#     def store(self, sample):
#         s, a, r, p, d = sample
#         self.ss_mem[self._idx] = s
#         self.as_mem[self._idx] = a
#         self.rs_mem[self._idx] = r
#         self.ps_mem[self._idx] = p
#         self.ds_mem[self._idx] = d
        
#         self._idx += 1
#         self._idx = self._idx % self.max_size

#         self.size += 1
#         self.size = min(self.size, self.max_size)

#     def sample(self, batch_size=None):
#         if batch_size == None:
#             batch_size = self.batch_size

#         idxs = np.random.choice(
#             self.size, batch_size, replace=False)
#         experiences = np.vstack(self.ss_mem[idxs]), \
#                       np.vstack(self.as_mem[idxs]), \
#                       np.vstack(self.rs_mem[idxs]), \
#                       np.vstack(self.ps_mem[idxs]), \
#                       np.vstack(self.ds_mem[idxs])
#         return experiences

#     def __len__(self):
#         return self.size

class ReplayBuffer():
    def __init__(self, max_size=10000, batch_size=64, state_dim=None):
        self.state_dim = state_dim
        self.ss_mem = np.empty((max_size, state_dim), dtype=np.float32)
        self.as_mem = np.empty((max_size), dtype=np.int32)
        self.rs_mem = np.empty((max_size), dtype=np.float32)
        self.ps_mem = np.empty((max_size, state_dim), dtype=np.float32)
        self.ds_mem = np.empty((max_size), dtype=np.float32)

        # Storage and sampling
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx = (self._idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = (
            self.ss_mem[idxs],
            self.as_mem[idxs],
            self.rs_mem[idxs],
            self.ps_mem[idxs],
            self.ds_mem[idxs]
        )
        return experiences
    def __len__(self):
        return self.size
    
class DDQN():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 max_gradient_norm,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.max_gradient_norm = max_gradient_norm
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        
        # argmax_a_q_sp = self.target_model(next_states).max(1)[1]
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[
            np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.max_gradient_norm)
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, truncated, info = env.step(action)
        # is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        # is_failure = is_terminal and not is_truncated
        is_terminal = is_terminal or truncated
        is_failure = is_terminal and not truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)
    def train(self, make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)

        # Initialize the replay buffer with state_dim
        self.replay_buffer = self.replay_buffer_fn(state_dim=nS)
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, is_terminal = env.reset(), False
            print(f"Initial state shape: {state.shape}")
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                if is_terminal:
                    gc.collect()
                    break

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.save_checkpoint(episode - 1, self.online_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, ar 10 {:05.1f}\u00B1{:05.1f}, 100 {:05.1f}\u00B1{:05.1f}, ex 100 {:02.1f}\u00B1{:02.1f}, ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward, mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat, mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes:
                    print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes:
                    print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward:
                    print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))
        env.close()
        del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time 

    # def train(self, make_env_fn, make_env_kargs, seed, gamma, 
    #           max_minutes, max_episodes, goal_mean_100_reward):
    #     training_start, last_debug_time = time.time(), float('-inf')

    #     self.checkpoint_dir = tempfile.mkdtemp()
    #     self.make_env_fn = make_env_fn
    #     self.make_env_kargs = make_env_kargs
    #     self.seed = seed
    #     self.gamma = gamma
        
    #     env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
    #     torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
    #     nS, nA = env.observation_space.shape[0], env.action_space.n
    #     self.episode_timestep = []
    #     self.episode_reward = []
    #     self.episode_seconds = []
    #     self.evaluation_scores = []        
    #     self.episode_exploration = []
        
    #     self.target_model = self.value_model_fn(nS, nA)
    #     self.online_model = self.value_model_fn(nS, nA)
    #     self.update_network()

    #     self.value_optimizer = self.value_optimizer_fn(self.online_model, 
    #                                                    self.value_optimizer_lr)

    #     self.replay_buffer = self.replay_buffer_fn()
    #     self.training_strategy = training_strategy_fn()
    #     # self.training_strategy = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
    #                                                 #   min_epsilon=0.3, 
    #                                                 #   decay_steps=20000) 
    #     self.evaluation_strategy = evaluation_strategy_fn()  
                    
    #     result = np.empty((max_episodes, 5))
    #     result[:] = np.nan
    #     training_time = 0
    #     for episode in range(1, max_episodes + 1):
    #         episode_start = time.time()
            
    #         state, is_terminal = env.reset(), False
    #         # print(f"Initial state shape: {state.shape}")
    #         self.episode_reward.append(0.0)
    #         self.episode_timestep.append(0.0)
    #         self.episode_exploration.append(0.0)

    #         for step in count():
    #             state, is_terminal = self.interaction_step(state, env)
                
    #             min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
    #             if len(self.replay_buffer) > min_samples:
    #                 experiences = self.replay_buffer.sample()
    #                 experiences = self.online_model.load(experiences)
    #                 self.optimize_model(experiences)
                
    #             if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
    #                 self.update_network()

    #             if is_terminal:
    #                 gc.collect()
    #                 break
            
    #         # stats
    #         episode_elapsed = time.time() - episode_start
    #         self.episode_seconds.append(episode_elapsed)
    #         training_time += episode_elapsed
    #         evaluation_score, _ = self.evaluate(self.online_model, env)
    #         self.save_checkpoint(episode-1, self.online_model)
            
    #         total_step = int(np.sum(self.episode_timestep))
    #         self.evaluation_scores.append(evaluation_score)
            
    #         mean_10_reward = np.mean(self.episode_reward[-10:])
    #         std_10_reward = np.std(self.episode_reward[-10:])
    #         mean_100_reward = np.mean(self.episode_reward[-100:])
    #         std_100_reward = np.std(self.episode_reward[-100:])
    #         mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
    #         std_100_eval_score = np.std(self.evaluation_scores[-100:])
    #         lst_100_exp_rat = np.array(
    #             self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
    #         mean_100_exp_rat = np.mean(lst_100_exp_rat)
    #         std_100_exp_rat = np.std(lst_100_exp_rat)
            
    #         wallclock_elapsed = time.time() - training_start
    #         result[episode-1] = total_step, mean_100_reward, \
    #             mean_100_eval_score, training_time, wallclock_elapsed
            
    #         reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
    #         reached_max_minutes = wallclock_elapsed >= max_minutes * 60
    #         reached_max_episodes = episode >= max_episodes
    #         reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
    #         training_is_over = reached_max_minutes or \
    #                            reached_max_episodes or \
    #                            reached_goal_mean_reward

    #         elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
    #         debug_message = 'el {}, ep {:04}, ts {:06}, '
    #         debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
    #         debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
    #         debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
    #         debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
    #         debug_message = debug_message.format(
    #             elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
    #             mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
    #             mean_100_eval_score, std_100_eval_score)
    #         print(debug_message, end='\r', flush=True)
    #         if reached_debug_time or training_is_over:
    #             print(ERASE_LINE + debug_message, flush=True)
    #             last_debug_time = time.time()
    #         if training_is_over:
    #             if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
    #             if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
    #             if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
    #             break
                
    #     final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
    #     wallclock_time = time.time() - training_start
    #     print('Training complete.')
    #     print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
    #           ' {:.2f}s wall-clock time.\n'.format(
    #               final_eval_score, score_std, training_time, wallclock_time))
    #     env.close() ; del env
    #     self.get_cleaned_checkpoints()
    #     return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=5):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.online_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))


ddqn_results = []
ddqn_agents, best_ddqn_agent_key, best_eval_score = {}, None, float('-inf')
for seed in SEEDS:
    environment_settings = {
        'env_name': 'CartPole-v1',
        'gamma': 1.00,
        'max_minutes': 20,
        'max_episodes': 10000,
        'goal_mean_100_reward': 475
    }

    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128)) #issue is here--> 
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    max_gradient_norm = float('inf')

    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=20000)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
    n_warmup_batches = 5
    update_target_every_steps = 10
    
    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = DDQN(replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 max_gradient_norm,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    ddqn_results.append(result)
    ddqn_agents[seed] = agent
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_ddqn_agent_key = seed
ddqn_results = np.array(ddqn_results)
_ = BEEP()

ddqn_agents[best_ddqn_agent_key].demo_last() 