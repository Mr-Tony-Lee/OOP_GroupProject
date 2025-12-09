import numpy as np
import random
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle

# 設定裝置 (有顯卡用顯卡，沒顯卡用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 抽象基類：Agent
# ============================================================================
class Agent(ABC):
    """所有 Agent 的基類"""
    
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, state):
        """根據狀態獲取動作"""
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """根據經驗進行學習"""
        pass

    @abstractmethod
    def save(self, filename):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, filename):
        """載入模型"""
        pass


# ============================================================================
# Q-Learning Agent (表格型，適合簡單狀態空間)
# ============================================================================
class QLearningAgent(Agent):
    """
    傳統 Q-Learning Agent
    
    適用於：
    - 狀態空間較小的環境
    - 需要快速收斂的場景
    
    狀態表示：(row, col, has_key) 三元組
    """
    
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(action_space)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}  # 使用 Dictionary 儲存 Q-Table: key=state_tuple, value=[q_values]

    def get_q_values(self, state):
        """取得狀態對應的 Q 值，不存在則初始化"""
        state_key = tuple(state)
        if state_key not in self.q_table:
            # 如果這個狀態沒遇過，初始化為全 0
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return self.q_table[state_key]

    def get_action(self, state):
        """使用 Epsilon-Greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return self.action_space.sample()  # 探索 (Exploration)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)  # 利用 (Exploitation)

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning 更新"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        
        # Q-Learning 更新公式
        # Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        target = reward + (0 if done else self.gamma * np.max(next_q_values))
        q_values[action] += self.lr * (target - q_values[action])

        # 衰減 Epsilon
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """保存 Q-Table 為 pickle 文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-Table saved to {filename}")

    def load(self, filename):
        """載入 Q-Table"""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-Table loaded from {filename}")


# ============================================================================
# DQN 相關類別
# ============================================================================
class QNetwork(nn.Module):
    """
    CNN + Scalar 混合網絡
    
    設計用於處理：
    - 圖像輸入 (spatial information)
    - 標量輸入 (global information like has_key)
    
    架構：
    1. CNN 分支：處理圖像 (7 channels)
    2. 全連接分支：處理標量 (1 value)
    3. 合併層：整合兩種信息
    4. 輸出層：生成 Q 值
    """
    
    def __init__(self, input_shape, scalar_dim, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        
        # 1. CNN 部分 (處理地圖畫面)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 自動計算 Flatten 後的大小
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_size = self.cnn(dummy).shape[1]
            
        # 2. 全連接部分 (結合 CNN 特徵 + 數值特徵)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, image, scalar):
        # image shape: (batch, 7, h, w)
        # scalar shape: (batch, 1)
        img_feat = self.cnn(image)
        combined = torch.cat((img_feat, scalar), dim=1)
        return self.fc(combined)


class BaseDQNAgent(Agent):
    def __init__(self, observation_space, action_space, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.05, batch_size=512, memory_size=50000,
                 target_update_freq=1000):
        super().__init__(action_space)
        
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Double DQN Agent using device: {self.device}")
        
        # 取得輸入形狀
        self.img_shape = observation_space['image'].shape
        self.scalar_dim = observation_space['scalars'].shape[0]
        self.action_dim = action_space.n

        # 初始化兩個網路：Policy Net (訓練用) 和 Target Net (計算目標用)
        self.policy_net = QNetwork(self.img_shape, self.scalar_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.img_shape, self.scalar_dim, self.action_dim).to(self.device)

        # 一開始先同步權重
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # 改用 SmoothL1Loss
        
        # 記憶體和計數器
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

    def get_action(self, state):
        """使用 Epsilon-Greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # 處理輸入
        img_tensor = torch.FloatTensor(state['image']).unsqueeze(0).to(self.device)
        scalar_tensor = torch.FloatTensor(state['scalars']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(img_tensor, scalar_tensor)

        return torch.argmax(q_values).item()
    def save(self, filename):
        """保存 Double DQN 模型"""
        torch.save(self.policy_net.state_dict(), filename)
        print(f"Double DQN Model saved to {filename}")

    def load(self, filename):
        """載入 Double DQN 模型"""
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        print(f"Double DQN Model loaded from {filename}")

class DoubleDQNAgent(BaseDQNAgent):
    """
    Double Deep Q-Network Agent
    
    改進點：
    1. 使用 Double DQN 減少 Q 值高估
    2. 支援多模態輸入 (Image + Scalar)
    3. 使用 SmoothL1Loss
    4. 增加梯度裁剪
    5. 調整超參數以匹配 CNNAgent
    """
    
    def __init__(self, observation_space, action_space, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.05, batch_size=512, memory_size=50000,
                 target_update_freq=1000):
        
        super().__init__(observation_space, action_space, learning_rate, 
                         gamma, epsilon, epsilon_decay, min_epsilon, batch_size, 
                         memory_size, target_update_freq)
    
    def learn(self, state, action, reward, next_state, done):
        """Double DQN 學習步驟"""
        self.memory.add(state, action, reward, next_state, done)
        
        # 衰減 Epsilon
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # 2. 如果樣本數不足，先不訓練
        if self.memory.size < self.batch_size:
            return
            
        # 3. 隨機抽樣 (Batch Training)
        batch = random.sample(self.memory, self.batch_size)
        
        # 整理 Batch 資料
        batch_imgs = torch.FloatTensor(np.array([x[0]['image'] for x in batch])).to(self.device)
        batch_scalars = torch.FloatTensor(np.array([x[0]['scalars'] for x in batch])).to(self.device)
        batch_actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(self.device)
        
        batch_next_imgs = torch.FloatTensor(np.array([x[3]['image'] for x in batch])).to(self.device)
        batch_next_scalars = torch.FloatTensor(np.array([x[3]['scalars'] for x in batch])).to(self.device)
        batch_dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(self.device)

        # 計算 Q 值
        # Current Q: 由 Policy Net 計算
        curr_q = self.policy_net(batch_imgs, batch_scalars).gather(1, batch_actions)
 
        # Double DQN Logic
        """
        先問 Current Net「你覺得該做哪個動作？」，然後再去問 Target Net「你覺得這個動作值多少？」。
        """
        # 1. 使用 Online Network 選擇動作
        with torch.no_grad():  
            next_actions = self.policy_net(batch_next_imgs, batch_next_scalars).argmax(1).unsqueeze(1)
            # 2. 使用 Target Network 評估動作
            next_q = self.target_net(batch_next_imgs, batch_next_scalars).gather(1, next_actions)
            
        target_q = batch_rewards + (self.gamma * next_q * (1 - batch_dones))
           
        loss = self.loss_fn(curr_q, target_q)
        
        # 更新網路
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新 Target Net 
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
# ============================================================================
# DQN Agent (CNN + 標量輸入的混合架構)
# ============================================================================

class DQNAgent(BaseDQNAgent):
    """
    Standard Deep Q-Network Agent (formerly CNNAgent)
    
    適用於：
    - 複雜的視覺環境
    - 需要同時處理空間和全局信息
    
    觀察格式：
    - image: (7, H, W) - 7 個通道的地圖
    - scalars: (1,) - has_key 信息
    """
    
    def __init__(self, observation_space, action_space, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.05, batch_size=512, memory_size=50000,
                 target_update_freq=1000):
                 
        super().__init__(observation_space, action_space, learning_rate, 
                         gamma, epsilon, epsilon_decay, min_epsilon, batch_size, 
                         memory_size, target_update_freq)
    
    def learn(self, state, action, reward, next_state, done):
        """DQN Agent 的學習步驟"""
        # 1. 儲存經驗
        self.memory.append((state, action, reward, next_state, done))
        
        # 衰減 Epsilon
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
        # 2. 如果樣本數不足，先不訓練
        if len(self.memory) < self.batch_size:
            return

        # 3. 隨機抽樣 (Batch Training)
        batch = random.sample(self.memory, self.batch_size)
        
        # 整理 Batch 資料
        batch_imgs = torch.FloatTensor(np.array([x[0]['image'] for x in batch])).to(self.device)
        batch_scalars = torch.FloatTensor(np.array([x[0]['scalars'] for x in batch])).to(self.device)
        batch_actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(self.device)
        
        batch_next_imgs = torch.FloatTensor(np.array([x[3]['image'] for x in batch])).to(self.device)
        batch_next_scalars = torch.FloatTensor(np.array([x[3]['scalars'] for x in batch])).to(self.device)
        batch_dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(self.device)

        # 計算 Q 值
        # Current Q: 由 Policy Net 計算
        curr_q = self.policy_net(batch_imgs, batch_scalars).gather(1, batch_actions)
        
        # DQN Logic 
        """
        直接問 Target Net 說「你覺得哪個最好？值是多少？」
        """
        # Target Q: 由 Target Net 計算
        with torch.no_grad():
            next_q = self.target_net(batch_next_imgs, batch_next_scalars).max(1)[0].unsqueeze(1)
            target_q = batch_rewards + (1 - batch_dones) * self.gamma * next_q
            
        loss = self.loss_fn(curr_q, target_q)
        
        # 更新網路
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 定期更新 Target Net (例如每學習 1000 次同步一次)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
