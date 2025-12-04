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
class DQN(nn.Module):
    """
    Deep Q-Network 架構 (修改版：支援圖像+純量輸入)
    
    用途：處理圖像和純量混合輸入
    輸入：
        - image: (batch_size, C, H, W)
        - scalar: (batch_size, scalar_dim)
    輸出：(batch_size, action_dim) 的 Q 值
    """
    
    def __init__(self, input_shape, scalar_dim, output_dim):
        super(DQN, self).__init__()
        c, h, w = input_shape
        
        # CNN 分支 (參考 CNNAgent 的 QNetwork)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 計算展平後的大小
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_size = self.cnn(dummy).shape[1]
            
        # 全連接分支 (結合 CNN 特徵 + 數值特徵)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, image, scalar):
        img_feat = self.cnn(image)
        combined = torch.cat((img_feat, scalar), dim=1)
        return self.fc(combined)


class ReplayBuffer:
    """經驗重放緩衝區，用於存儲和採樣訓練數據 (支援圖像+純量)"""
    
    def __init__(self, capacity, state_shape, scalar_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.scalars = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_scalars = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """添加一條經驗"""
        self.states[self.ptr] = state['image']
        self.scalars[self.ptr] = state['scalars']
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state['image']
        self.next_scalars[self.ptr] = next_state['scalars']
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """隨機採樣一個 batch"""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.scalars[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.next_scalars[ind],
            self.dones[ind]
        )


class DQNAgent(Agent):
    """
    Deep Q-Network Agent (改進版)
    
    改進點：
    1. 支援多模態輸入 (Image + Scalar)
    2. 使用 SmoothL1Loss
    3. 增加梯度裁剪
    4. 調整超參數以匹配 CNNAgent
    """
    
    def __init__(self, observation_space, action_space, learning_rate=0.001, 
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.01, batch_size=512, memory_size=50000):
        super().__init__(action_space)
        
        self.state_shape = observation_space['image'].shape
        self.scalar_dim = observation_space['scalars'].shape[0]
        self.action_dim = action_space.n
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        
        # 使用優化後的 Replay Buffer
        self.memory = ReplayBuffer(memory_size, self.state_shape, self.scalar_dim, self.action_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        
        self.q_network = DQN(self.state_shape, self.scalar_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_shape, self.scalar_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # 改用 SmoothL1Loss
        
        self.steps = 0
        self.target_update_freq = 1000

    def get_action(self, state):
        """使用 Epsilon-Greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # 處理輸入
        img_tensor = torch.FloatTensor(state['image']).unsqueeze(0).to(self.device)
        scalar_tensor = torch.FloatTensor(state['scalars']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(img_tensor, scalar_tensor)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        """DQN 學習步驟"""
        self.memory.add(state, action, reward, next_state, done)
        self.steps += 1
        
        if self.memory.size < self.batch_size:
            return
            
        states, scalars, actions, rewards, next_states, next_scalars, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        scalars = torch.FloatTensor(scalars).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_scalars = torch.FloatTensor(next_scalars).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN Logic
        # 1. 使用 Online Network 選擇動作
        with torch.no_grad():
            next_actions = self.q_network(next_states, next_scalars).argmax(1).unsqueeze(1)
            # 2. 使用 Target Network 評估動作
            next_q_values = self.target_network(next_states, next_scalars).gather(1, next_actions)
            
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # 當前 Q 值
        current_q_values = self.q_network(states, scalars).gather(1, actions)
            
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 增加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
             
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        """保存 Q-Network 權重"""
        torch.save(self.q_network.state_dict(), filename)
        print(f"DQN Model saved to {filename}")

    def load(self, filename):
        """載入 Q-Network 權重"""
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()
        print(f"DQN Model loaded from {filename}")


# ============================================================================
# CNN Agent (CNN + 標量輸入的混合架構)
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


class CNNAgent(Agent):
    """
    CNN-based Agent (CNN + 標量混合架構)
    
    適用於：
    - 複雜的視覺環境
    - 需要同時處理空間和全局信息
    
    觀察格式：
    - image: (7, H, W) - 7 個通道的地圖
    - scalars: (1,) - has_key 信息
    """
    
    def __init__(self, observation_space, action_space, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        super().__init__(action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # 取得輸入形狀
        img_shape = observation_space['image'].shape
        scalar_dim = observation_space['scalars'].shape[0]
        
        # 初始化兩個網路：Policy Net (訓練用) 和 Target Net (計算目標用)
        self.policy_net = QNetwork(img_shape, scalar_dim, action_space.n).to(device)
        self.target_net = QNetwork(img_shape, scalar_dim, action_space.n).to(device)
        
        # 一開始先同步權重
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target Net 不需要計算梯度，設為評估模式
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss (SmoothL1) 比 MSE 更穩定
        
        # 記憶體和計數器
        self.memory = deque(maxlen=50000)
        self.batch_size = 512
        self.learn_step_counter = 0  # 計數器，用來決定何時更新 Target Net

    def get_action(self, state):
        """使用 Epsilon-Greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # 處理輸入資料
        img_tensor = torch.FloatTensor(state['image']).unsqueeze(0).to(device)     # (1, 7, H, W)
        scalar_tensor = torch.FloatTensor(state['scalars']).unsqueeze(0).to(device)  # (1, 1)
        
        with torch.no_grad():
            q_values = self.policy_net(img_tensor, scalar_tensor)
            
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        """CNN Agent 的學習步驟"""
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
        batch_imgs = torch.FloatTensor(np.array([x[0]['image'] for x in batch])).to(device)
        batch_scalars = torch.FloatTensor(np.array([x[0]['scalars'] for x in batch])).to(device)
        batch_actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(device)
        batch_rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(device)
        
        batch_next_imgs = torch.FloatTensor(np.array([x[3]['image'] for x in batch])).to(device)
        batch_next_scalars = torch.FloatTensor(np.array([x[3]['scalars'] for x in batch])).to(device)
        batch_dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(device)

        # 計算 Q 值
        # Current Q: 由 Policy Net 計算
        curr_q = self.policy_net(batch_imgs, batch_scalars).gather(1, batch_actions)
        
        # Target Q: 由 Target Net 計算 (關鍵！)
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
        if self.learn_step_counter % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Target Network Updated")

    def save(self, filename):
        """保存 CNN 模型"""
        torch.save(self.policy_net.state_dict(), filename)
        print(f"CNN Model saved to {filename}")

    def load(self, filename):
        """載入 CNN 模型"""
        self.policy_net.load_state_dict(torch.load(filename, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 載入時也要同步
        self.policy_net.eval()
        print(f"CNN Model loaded from {filename}")
