import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from abc import ABC, abstractmethod
from CheatingEnv import LessSlipperyFrozenLakeEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class Agent(ABC):
    """所有 Agent 的基類"""
    
    def __init__(self, agent_type="QLearning", map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi", is_cheating=False):
        # -------- env generation parameters --------
        self.agent_type = agent_type
        self.map_name = map_name
        self.is_training = is_training
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.is_cheating = is_cheating
        self.render = False


        self.rng = np.random.default_rng()   # random number generator
        
        # -------- training parameters --------
        self.episodes = episodes
        self.is_training = is_training
        self.q = None   # Q-table

        self.learning_rate_a = 0.5 # alpha or learning rate
        self.learning_rate_decay = 0.9995 # Adjusted for smooth decay
        self.min_learning_rate = 0.0001

        self.discount_factor_g = 0.99 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        
        self.exploration_rate = 1         # 1 = 100% random actions
        self.exploration_decay_rate = 0.00015    # epsilon decay rate. 1/0.0001 = 10,000
        self.min_exploration_rate = 0.001
        
        self.file_dir = "Result/" + self.agent_type + "/" + self.map_name
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
    
    def print_success_rate(self, rewards_per_episode):
        """Calculate and print the success rate of the agent."""
        total_episodes = len(rewards_per_episode)
        success_count = np.sum(rewards_per_episode)
        success_rate = (success_count / total_episodes) * 100
        print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
        return success_rate
    
    def plot_rewards(self, rewards_per_episode):
        """Plot the sum of rewards over episodes."""
        plt.figure()  # 建立一個新的圖表，避免與上一張圖重疊
        
        sum_rewards = np.zeros(self.episodes)
        for t in range(self.episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        

        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards (Last 100 Episodes)') 
        plt.title('Frozen Lake Rewards over Episodes')
        if not os.path.exists(self.map_name):
                os.makedirs(self.map_name)
        if self.is_training == True :
            plt.savefig(f'{self.file_dir}/frozen_lake_Training{self.map_name}.png')
        else:
            plt.savefig(f'{self.file_dir}/frozen_lake_Evaluation{self.map_name}.png')
            
        plt.close() # 存檔後關閉圖表，釋放記憶體

    @abstractmethod
    def run(self):
        pass


class QLearningAgent(Agent):
    def init_q_table(self, env):
        """Initialize the Q-table."""
        if(self.is_training):
            # Optimistic Initialization: Initialize with small positive values to encourage exploration
            # self.q = np.zeros((env.observation_space.n, env.action_space.n)) 
            self.q = np.random.uniform(low=0.0, high=0.001, size=(env.observation_space.n, env.action_space.n))
        else:
            f = open(f'{self.file_dir}/frozen_lake{self.map_name}.pkl', 'rb')
            self.q = pickle.load(f)
            f.close()

    def run(self):
        if self.is_cheating:
            env = LessSlipperyFrozenLakeEnv(map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
            env = TimeLimit(env, max_episode_steps=100) 
        else:
            env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)

        self.init_q_table(env)

        rewards_per_episode = np.zeros(self.episodes)
        for i in range(self.episodes):
            state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
            terminated = False      # True when fall in hole or reached goal
            truncated = False       # True when actions > 200

            while(not terminated and not truncated):
                if self.is_training and self.rng.random() < self.exploration_rate:
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    action = np.argmax(self.q[state,:])

                new_state, reward, terminated, truncated,_ = env.step(action)

                if self.is_training:
                    # 利用轉移機率計算總期望值 ( 算小作弊? )
                    expected_target = 0
                    transitions = env.unwrapped.P[state][action]
                    
                    for prob, next_s, r, term in transitions:
                        target = r
                        if not term:
                            target += self.discount_factor_g * np.max(self.q[next_s, :])
                        expected_target += prob * target

                    self.q[state,action] = self.q[state,action] + self.learning_rate_a * (
                        expected_target - self.q[state,action]
                    )

                state = new_state

            self.exploration_rate = max(self.exploration_rate - self.exploration_decay_rate, self.min_exploration_rate)
            
            # Smooth learning rate decay
            self.learning_rate_a = max(self.learning_rate_a * self.learning_rate_decay, self.min_learning_rate)

            if reward == 1:
                rewards_per_episode[i] = 1

        env.close()
        
        self.plot_rewards(rewards_per_episode)
        
        success_rate = 0.0
        if self.is_training == False:
            success_rate = self.print_success_rate(rewards_per_episode)

        if self.is_training:
            f = open(f"{self.file_dir}/frozen_lake{self.map_name}.pkl","wb")
            pickle.dump(self.q, f)
            f.close()
        
        return success_rate


# class DPAgent(Agent):
#     def __init__(self, map_name="8x8", is_slippery=True, render_mode="ansi"):
#         self.map_name = map_name
#         self.is_slippery = is_slippery
#         self.render_mode = render_mode
#         self.gamma = 0.99  # Discount factor
#         self.theta = 1e-9  # Convergence threshold
#         self.policy = None
#         self.V = None

#     def value_iteration(self, env):
#         """
#         Perform Value Iteration to find the optimal policy.
#         """
#         n_states = env.observation_space.n
#         n_actions = env.action_space.n
#         self.V = np.zeros(n_states)
        
#         print("Starting Value Iteration...")
#         iteration = 0
#         while True:
#             delta = 0
#             for s in range(n_states):
#                 v = self.V[s]
#                 # Calculate the value for each action
#                 action_values = np.zeros(n_actions)
#                 for a in range(n_actions):
#                     for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
#                         target = reward
#                         if not terminated:
#                             target += self.gamma * self.V[next_s]
#                         action_values[a] += prob * target
                
#                 # Update the value of the state to the maximum action value
#                 self.V[s] = np.max(action_values)
#                 delta = max(delta, abs(v - self.V[s]))
            
#             iteration += 1
#             if delta < self.theta:
#                 print(f"Value Iteration converged in {iteration} iterations.")
#                 break
        
#         # Extract the optimal policy
#         self.policy = np.zeros(n_states, dtype=int)
#         for s in range(n_states):
#             action_values = np.zeros(n_actions)
#             for a in range(n_actions):
#                 for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
#                     target = reward
#                     if not terminated:
#                         target += self.gamma * self.V[next_s]
#                     action_values[a] += prob * target
#             self.policy[s] = np.argmax(action_values)

#     def print_success_rate(self, rewards_per_episode):
#         """Calculate and print the success rate of the agent."""
#         total_episodes = len(rewards_per_episode)
#         success_count = np.sum(rewards_per_episode)
#         success_rate = (success_count / total_episodes) * 100
#         print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
#         return success_rate

#     def run_evaluation(self, episodes=1000):
#         env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
        
#         # Solve the MDP first
#         self.value_iteration(env)
        
#         print(f"\nEvaluating DP Policy for {episodes} episodes...")
#         rewards_per_episode = np.zeros(episodes)
        
#         for i in range(episodes):
#             state = env.reset()[0]
#             terminated = False
#             truncated = False
            
#             while not terminated and not truncated:
#                 action = self.policy[state]
#                 new_state, reward, terminated, truncated, _ = env.step(action)
#                 state = new_state
                
#                 if reward == 1:
#                     rewards_per_episode[i] = 1
        
#         env.close()
#         self.print_success_rate(rewards_per_episode)


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.rewards.append(info['episode']['r'])
        return True


def make_env_helper(map_name, is_slippery, render_mode, is_cheating, rank, seed=0):
    def _init():
        if is_cheating:
            env = LessSlipperyFrozenLakeEnv(map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
            env = TimeLimit(env, max_episode_steps=100) 
        else:
            env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    return _init

class PPOAgent(Agent):
    def run(self):
        model_path = f"{self.file_dir}/ppo_frozen_lake_{self.map_name}"

        if self.is_training:
            n_envs = 5
            # Create the vectorized environment
            # Use SubprocVecEnv for parallel execution
            env = SubprocVecEnv([make_env_helper(self.map_name, self.is_slippery, self.render_mode, self.is_cheating, i) for i in range(n_envs)])
            env = VecMonitor(env)

            # Initialize PPO model
            model = PPO("MlpPolicy", env, verbose=1)
            
            # Create the callback to collect rewards
            callback = RewardCallback()

            # Train the model
            total_timesteps = self.episodes * 35
            model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Save the model
            model.save(model_path)
            
            # Get rewards from Callback
            rewards_per_episode = callback.rewards
            
            # Plot rewards
            self.plot_rewards(rewards_per_episode)
            
            # Calculate success rate
            success_rate = self.print_success_rate(rewards_per_episode)
            
            env.close()

        else:
            # Evaluation (Single environment is usually enough and simpler)
            if self.is_cheating:
                env = LessSlipperyFrozenLakeEnv(map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
                env = TimeLimit(env, max_episode_steps=100) 
            else:
                env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
            
            # Load the model
            if os.path.exists(model_path + ".zip"):
                model = PPO.load(model_path)
            else:
                print(f"Model not found at {model_path}")
                return 0.0

            # Evaluate
            rewards_per_episode = []
            for _ in range(self.episodes):
                obs, _ = env.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                while not terminated and not truncated:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    episode_reward += reward
                rewards_per_episode.append(episode_reward)
            
            self.plot_rewards(rewards_per_episode)
            success_rate = self.print_success_rate(rewards_per_episode)
            env.close()
        
        return success_rate