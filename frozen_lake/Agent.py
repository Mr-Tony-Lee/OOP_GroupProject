import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from CheatingEnv import LessSlipperyFrozenLakeEnv
from gymnasium.wrappers import TimeLimit


class Agent(ABC):
    """所有 Agent 的基類"""
    
    def __init__(self, agent_type="QLearning", map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi", is_cheating=False
                , step_penalty=0.00035, closer_to_goal_reward=0.005):
        
        # -------- env generation parameters --------
        self.agent_type = agent_type
        self.map_name = map_name
        self.episodes = episodes
        self.is_training = is_training
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.is_cheating = is_cheating
        self.render = False
        
        
        # -------- reward parameters --------
        self.step_penalty = step_penalty
        self.closer_to_goal_reward = closer_to_goal_reward


        self.rng = np.random.default_rng()   # random number generator
        
        
        self.file_dir = "Result/" + self.agent_type + "/" + self.map_name
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
    
    def print_success_rate(self, rewards_per_episode):
        """計算並列印代理的成功率。"""
        total_episodes = len(rewards_per_episode)
        success_count = np.sum(rewards_per_episode)
        success_rate = (success_count / total_episodes) * 100
        print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
        return success_rate
    
    def plot_rewards(self, rewards_per_episode):
        """畫出過去100個episodes的獎勵總和。"""
        plt.figure()  # 建立一個新的圖表，避免與上一張圖重疊
        
        sum_rewards = np.zeros(self.episodes)
        for t in range(self.episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        

        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards (Last 100 Episodes)') 
        plt.title('Frozen Lake Rewards over Episodes')
        if self.is_training == True :
            plt.savefig(f'{self.file_dir}/frozen_lake_Training{self.map_name}.png')
        else:
            plt.savefig(f'{self.file_dir}/frozen_lake_Evaluation{self.map_name}.png')
            
        plt.close() # 存檔後關閉圖表，釋放記憶體

    def close_to_goal(self, state, env):
        """distacnce to goal - 保持不變"""
        if(self.map_name == "4x4"):
            goal_state = 15 # 4x4 地圖的目標狀態是 15
        elif self.map_name == "8x8":
            goal_state = 63 # 8x8 地圖的目標狀態是 63
        state_row, state_col = divmod(state, env.unwrapped.ncol)
        goal_row, goal_col = divmod(goal_state, env.unwrapped.ncol)
        distance = abs(state_row - goal_row) + abs(state_col - goal_col)
        return distance 
    
    def check_danger_action(self, state, action, env):
        """檢查特定動作是否會導致高機率掉入洞中 - 保持不變"""
        transitions = env.unwrapped.P[state][action]
        hole_fall_prob = 0
        
        for prob, next_s, r, term in transitions:
            if term and r == 0:
                hole_fall_prob += prob
                
        return hole_fall_prob > 0.2
    
    def step_penalty_helper(self, reward, terminated):
        """Helper function 去計算步數懲罰"""
        if terminated and reward == 1:
            # 終點獎勵保持不變
            return reward
        else:
            # 否則，套用步數懲罰 (包含掉入洞中 terminated and reward == 0)
            return reward - self.step_penalty
    
    def init_env(self):
        """初始化環境"""
        if self.is_cheating:
            env = LessSlipperyFrozenLakeEnv(map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
            env = TimeLimit(env, max_episode_steps=100) 
        else:
            env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
        return env
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    def run(self):
        if self.is_training:
            return self.train()
        else:
            return self.test()


class QLearningAgent(Agent):
    def __init__(self, agent_type="QLearning", map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi", is_cheating=False
                , step_penalty=0.0001, closer_to_goal_reward=0.005
                , learning_rate = 0.5,learning_rate_decay = 0.9995, gamma=0.99, exploration_decay_rate=0.00015):
        
        # ------- inherit base Agent class --------
        super().__init__(agent_type=agent_type, map_name=map_name, episodes=episodes, is_slippery=is_slippery, render_mode=render_mode, is_training=is_training, is_cheating=is_cheating
                , step_penalty=step_penalty, closer_to_goal_reward=closer_to_goal_reward)
        
        # -------- training parameters --------

        self.q = None   # Q-table

        self.learning_rate = learning_rate # alpha or learning rate
        self.learning_rate_decay = learning_rate_decay # Adjusted for smooth decay
        self.min_learning_rate = 0.0001

        self.gamma = gamma # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        
        self.exploration_rate = 1         # 1 = 100% random actions
        self.exploration_decay_rate = exploration_decay_rate    # epsilon decay rate. 1/0.0001 = 10,000
        self.min_exploration_rate = 0.001
        
        self.step_penalty = 0.0001
        
    def init_q_table(self, env):
        """Initialize the Q-table."""
        if(self.is_training):
            self.q = np.random.uniform(low=0.0, high=0.001, size=(env.observation_space.n, env.action_space.n))
        else:
            f = open(f'{self.file_dir}/frozen_lake{self.map_name}.pkl', 'rb')
            self.q = pickle.load(f)
            f.close()
    
    def train(self):
        env = self.init_env()
        self.init_q_table(env)
        
        rewards_per_episode = np.zeros(self.episodes)
        
        for i in tqdm(range(self.episodes)):
            state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
            terminated = False      # True when fall in hole or reached goal
            truncated = False       # True when actions > 200

            while(not terminated and not truncated):
                if self.rng.random() < self.exploration_rate:
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    action = np.argmax(self.q[state,:])


                new_state, reward, terminated, truncated,_ = env.step(action)
                reward = self.step_penalty_helper(reward, terminated)
                
                # 如果新狀態更接近目標，給予獎勵
                if(self.close_to_goal(new_state, env) < self.close_to_goal(state, env)):
                    reward += self.closer_to_goal_reward
                
                reward = self.step_penalty_helper(reward, terminated)
                
                
                
                # 利用轉移機率計算總期望值 ( 算小作弊? )
                expected_target = 0
                transitions = env.unwrapped.P[state][action]
                
                for prob, next_s, r, term in transitions:
                    target = r
                    if not term :
                        target += self.gamma * np.max(self.q[next_s, :])
                    expected_target += prob * target

                self.q[state,action] = self.q[state,action] + self.learning_rate * (
                    expected_target - self.q[state,action]
                )

                state = new_state

            self.exploration_rate = max(self.exploration_rate - self.exploration_decay_rate, self.min_exploration_rate)
            
            # Smooth learning rate decay
            self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)
            
            if reward == 1:
                rewards_per_episode[i] = 1

        env.close()
        
        self.plot_rewards(rewards_per_episode)
        
        # 儲存 Q-table
        f = open(f"{self.file_dir}/frozen_lake{self.map_name}.pkl","wb")
        pickle.dump(self.q, f)
        f.close()

    def test(self):
        env = self.init_env()
        self.init_q_table(env)
        rewards_per_episode = np.zeros(self.episodes)
        
        # External-interference thresholds (same idea as DPAgent.test)
        RISK_THRESHOLD = 0.45
        BACKTRACK_THRESHOLD = 0.35

        for i in tqdm(range(self.episodes)):
            state = env.reset()[0]
            terminated = False
            truncated = False
            prev_state = -1

            while(not terminated and not truncated):
                n_actions = env.action_space.n

                # Base greedy action from learned Q-table
                base_action = np.argmax(self.q[state, :])

                # 1) compute expected q-values under environment stochasticity
                q_values = np.zeros(n_actions)
                action_hole_probs = np.zeros(n_actions)

                for a in range(n_actions):
                    for prob, next_s, reward, terminated_ in env.unwrapped.P[state][a]:
                        target = self.step_penalty_helper(reward, terminated_)
                        if not terminated_:
                            target += self.gamma * np.max(self.q[next_s, :])
                        q_values[a] += prob * target

                        # accumulate hole risk
                        if terminated_ and reward == 0:
                            action_hole_probs[a] += prob

                # Start with the base greedy action
                action = base_action

                # 2) backtrack avoidance: if best action tends to go back to prev_state, avoid it
                if prev_state != -1:
                    backtrack_prob = 0.0
                    for prob, next_s, _, _ in env.unwrapped.P[state][base_action]:
                        if next_s == prev_state:
                            backtrack_prob += prob

                    if backtrack_prob > BACKTRACK_THRESHOLD:
                        temp_q_values = np.copy(q_values)
                        temp_q_values[base_action] = -np.inf
                        action = np.argmax(temp_q_values)

                # 3) risk filter: if chosen action has high hole risk, pick a safer high-q action
                current_action_hole_prob = action_hole_probs[action]
                if current_action_hole_prob > RISK_THRESHOLD:
                    action_metrics = []
                    for a in range(n_actions):
                        action_metrics.append((q_values[a], action_hole_probs[a], a))

                    # sort by q desc then hole prob asc
                    action_metrics.sort(key=lambda x: (x[0], -x[1]), reverse=True)

                    for qv, prob, a_candidate in action_metrics:
                        if prob <= RISK_THRESHOLD:
                            action = a_candidate
                            break

                # 4) execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                prev_state = state
                state = new_state

                if reward == 1:
                    rewards_per_episode[i] = 1

        env.close()
        self.plot_rewards(rewards_per_episode)
        success_rate = self.print_success_rate(rewards_per_episode)
        
        return success_rate

class DPAgent(Agent):
    def __init__(self, agent_type="DPAgent", map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi", is_cheating=False
                , step_penalty=0.00035, closer_to_goal_reward=0.005
                , gamma=0.99, theta=1e-6):
        
        # ------- inherit base Agent class --------
        super().__init__(agent_type=agent_type, map_name=map_name, episodes=episodes, is_slippery=is_slippery, render_mode=render_mode, is_training=is_training, is_cheating=is_cheating
                         , step_penalty=step_penalty, closer_to_goal_reward=closer_to_goal_reward)
        
        
        # -------- DP specific parameters --------
        self.gamma = 0.984652024790762
        self.theta = 1.1855031515737096e-07
        self.step_penalty = 0.0011852435317933866
        
        
        self.policy = None
        self.V = None
        self.is_cheating = False

    def save_policy_and_value(self):
        """Save the learned policy and value function to files."""
        f = open(f"{self.file_dir}/dp_policy{self.map_name}.pkl","wb")
        pickle.dump(self.policy, f)
        f.close()
        
        f = open(f"{self.file_dir}/dp_value{self.map_name}.pkl","wb")
        pickle.dump(self.V, f)
        f.close()
    
    def load_policy_and_value(self):
        """Load the learned policy and value function from files."""
        f = open(f"{self.file_dir}/dp_policy{self.map_name}.pkl","rb")
        self.policy = pickle.load(f)
        f.close()
        
        f = open(f"{self.file_dir}/dp_value{self.map_name}.pkl","rb")
        self.V = pickle.load(f)
        f.close()
    
    def value_iteration(self, env):
        """
        Perform Value Iteration to find the optimal policy.
        """
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.V = np.zeros(n_states)
        
        print("Starting Value Iteration...")
        iteration = 0

        while True:
            delta = 0
            for s in range(n_states):
                v = self.V[s]
                # Calculate the value for each action
                action_values = np.zeros(n_actions)
                for a in range(n_actions):
                    for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                        
                        target = self.step_penalty_helper(reward, terminated)
                        
                        if not terminated:
                            target += self.gamma * self.V[next_s]
                        action_values[a] += prob * target
                
                # Update the value of the state to the maximum action value
                self.V[s] = np.max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            iteration += 1
            if delta < self.theta:
                print(f"Value Iteration converged in {iteration} iterations.")
                break
        
        # 讓政策提取時也能考慮步數懲罰
        self.policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                    
                    target = self.step_penalty_helper(reward, terminated)
                    
                    if not terminated:
                        target += self.gamma * self.V[next_s]
                    action_values[a] += prob * target
            self.policy[s] = np.argmax(action_values)
            
        self.save_policy_and_value()
        

    def train(self):
        env = self.init_env()
        self.value_iteration(env)
        env.close()

    def test(self):
        env = self.init_env()
        self.load_policy_and_value()        
        rewards_per_episode = np.zeros(self.episodes)

        # --- 外部干擾參數 ---
        RISK_THRESHOLD = 0.45 
        BACKTRACK_THRESHOLD = 0.35 # 針對濕滑性 (1/3 滑到其他方向)
        
        for i in tqdm(range(self.episodes)):
            state = env.reset()[0]
            terminated = False
            truncated = False
            prev_state = -1 # 初始化上一個狀態
            
            while not terminated and not truncated:
                
                action = self.policy[state] # 來自訓練好的確定性策略
                
                # ----------------------------------------------------
                # I. 外部干擾：強制「不後退」干擾
                # ----------------------------------------------------
                n_actions = env.action_space.n
                
                # 1. 計算 Q-values for all actions (使用包含懲罰的 Q 值)
                q_values = np.zeros(n_actions)
                action_hole_probs = np.zeros(n_actions)

                for a in range(n_actions):
                    for prob, next_s, reward, terminated_ in env.unwrapped.P[state][a]:
                        
                        # 套用步數懲罰
                        target = self.step_penalty_helper(reward, terminated_)
                        
                        if not terminated_:
                            target += self.gamma * self.V[next_s]
                        q_values[a] += prob * target
                        
                        # 累積掉洞風險
                        if terminated_ and reward == 0:
                            action_hole_probs[a] += prob
                
                # 2. 檢查最佳動作是否高機率導致後退
                best_action = self.policy[state]
                
                if prev_state != -1:
                    backtrack_prob = 0.0
                    for prob, next_s, _, _ in env.unwrapped.P[state][best_action]:
                        if next_s == prev_state:
                            backtrack_prob += prob
                            
                    if backtrack_prob > BACKTRACK_THRESHOLD:
                        
                        # 最佳動作高機率後退，將其 Q 值設定為負無窮，選擇次優
                        temp_q_values = np.copy(q_values)
                        temp_q_values[best_action] = -np.inf
                        
                        # 選擇 Q 值最高的「非後退」動作作為基礎動作
                        action = np.argmax(temp_q_values) 
                
                # ----------------------------------------------------
                # II. 外部干擾：動作安全過濾器 (作用於當前 action)
                # ----------------------------------------------------
                
                # 檢查當前選定動作 (可能是修正後的新動作) 的掉洞風險
                current_action_hole_prob = action_hole_probs[action]

                if current_action_hole_prob > RISK_THRESHOLD:
                    
                    # 尋找次優且安全的動作：Q 值最高且風險低於閾值的動作
                    action_metrics = []
                    for a in range(n_actions):
                        action_metrics.append((q_values[a], action_hole_probs[a], a))

                    # 排序：Q值降序，Hole_Prob升序
                    action_metrics.sort(key=lambda x: (x[0], -x[1]), reverse=True) 

                    # 選擇第一個 Q 值高且風險低於閾值的動作
                    for q, prob, a_candidate in action_metrics:
                        if prob <= RISK_THRESHOLD:
                            action = a_candidate
                            break # 找到安全動作，跳出
                
                # ----------------------------------------------------
                # III. 執行動作與更新狀態
                # ----------------------------------------------------
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # 更新歷史狀態
                prev_state = state 
                state = new_state
                
                if reward == 1:
                    rewards_per_episode[i] = 1
        env.close()
        self.plot_rewards(rewards_per_episode)
        success_rate = self.print_success_rate(rewards_per_episode)
        
        return success_rate

# class SARSAAgent(Agent):
    
#     def __init__(self, agent_type="SARSA", map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi", is_cheating=False
#             , step_penalty=0.0001, closer_to_goal_reward=0.005
#             , learning_rate = 0.5,learning_rate_decay = 0.9995, gamma=0.99, exploration_decay_rate=0.00015):
    
#         # ------- inherit base Agent class --------
#         super().__init__(agent_type=agent_type, map_name=map_name, episodes=episodes, is_slippery=is_slippery, render_mode=render_mode, is_training=is_training, is_cheating=is_cheating
#                 , step_penalty=step_penalty, closer_to_goal_reward=closer_to_goal_reward)
        
        
#         # -------- SARSA specific parameters --------
#         self.q = None   # Q-table

#         self.learning_rate = learning_rate # alpha or learning rate
#         self.learning_rate_decay = learning_rate_decay # Adjusted for smooth decay
#         self.min_learning_rate = 0.0001

#         self.gamma = gamma # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        
#         self.exploration_rate = 1         # 1 = 100% random actions
#         self.exploration_decay_rate = exploration_decay_rate    # epsilon decay rate. 1/0.0001 = 10,000
#         self.min_exploration_rate = 0.001
        
        
#     def init_q_table(self, env):
#         """Initialize the Q-table."""
#         if(self.is_training):
#             # Optimistic Initialization: Initialize with small positive values to encourage exploration
#             # self.q = np.zeros((env.observation_space.n, env.action_space.n)) 
#             self.q = np.random.uniform(low=0.0, high=0.001, size=(env.observation_space.n, env.action_space.n))
#         else:
#             f = open(f'{self.file_dir}/frozen_lake{self.map_name}.pkl', 'rb')
#             self.q = pickle.load(f)
#             f.close()
            
#     def train(self):
#         env = self.init_env()
#         self.init_q_table(env)
        
#         rewards_per_episode = np.zeros(self.episodes)
        
#         for i in tqdm(range(self.episodes)):
#             state = env.reset()[0]
#             terminated = False
#             truncated = False
            
#             # --- SARSA Step 1: 選擇第一個動作 A (使用 ε-greedy) ---
#             if self.rng.random() < self.exploration_rate:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(self.q[state,:])
            
#             while(not terminated and not truncated):
                
#                 # --- SARSA Step 2: 執行動作 A，得到 R 和 S' ---
#                 new_state, reward, terminated, truncated, _ = env.step(action)
#                 reward = self.step_penalty_helper(reward, terminated)
                
#                 # 如果新狀態更接近目標，給予獎勵
#                 if(self.close_to_goal(new_state, env) < self.close_to_goal(state, env)):
#                     reward += self.closer_to_goal_reward
                
#                 reward = self.step_penalty_helper(reward, terminated)
                
#                 # --- SARSA Step 3: 選擇下一個動作 A' (使用 ε-greedy) ---
#                 if self.rng.random() < self.exploration_rate:
#                     new_action = env.action_space.sample() 
#                 else:
#                     new_action = np.argmax(self.q[new_state,:]) 
                
#                 # --- SARSA Step 4: Q 值更新 ---
                
#                 # 終點狀態的 Q 值為 0
#                 Q_next = self.q[new_state, new_action] if not terminated else 0
                
#                 # SARSA TD Target
#                 td_target = reward + self.gamma * Q_next
                
#                 self.q[state,action] = self.q[state,action] + self.learning_rate * (
#                     td_target - self.q[state,action]
#                 )
                
#                 # --- SARSA Step 5: 更新狀態和動作 ---
#                 state = new_state
#                 action = new_action
                

#             self.exploration_rate = max(self.exploration_rate - self.exploration_decay_rate, self.min_exploration_rate)
#             self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)
            
#             if reward == 1:
#                 rewards_per_episode[i] = 1

#         env.close()
        
#         self.plot_rewards(rewards_per_episode)
#         f = open(f"{self.file_dir}/frozen_lake{self.map_name}.pkl","wb")
#         pickle.dump(self.q, f)
#         f.close()
            
#     def test(self):
#         """測試/評估邏輯 - 保持不變 (使用貪婪策略)"""
#         env = self.init_env()
#         self.init_q_table(env)
#         rewards_per_episode = np.zeros(self.episodes)
        
#         for i in tqdm(range(self.episodes)):
#             state = env.reset()[0]
#             terminated = False
#             truncated = False
            
#             while(not terminated and not truncated):
#                 # 評估模式使用貪婪策略
#                 action = np.argmax(self.q[state,:])
#                 new_state, reward, terminated, truncated, _ = env.step(action)
#                 state = new_state
            
#             if reward == 1:
#                  rewards_per_episode[i] = 1

#         env.close()
        
#         self.plot_rewards(rewards_per_episode)
#         success_rate = self.print_success_rate(rewards_per_episode)
        
#         return success_rate