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

class Agent(ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

class QLearningAgent(Agent):
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(action_space)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {} # 使用 Dictionary 來儲存 Q-Table: key=state_tuple, value=[q_values]

    def get_q_values(self, state):
        # 將 numpy array 轉換為 tuple 以作為 dictionary 的 key
        state_key = tuple(state)
        if state_key not in self.q_table:
            # 如果這個狀態沒遇過，初始化為全 0
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return self.q_table[state_key]

    def get_action(self, state):
        # Epsilon-Greedy 策略
        if random.random() < self.epsilon:
            return self.action_space.sample() # 探索 (Exploration)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values) # 利用 (Exploitation)

    def learn(self, state, action, reward, next_state, done):
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
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-Table saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-Table loaded from {filename}")

# 定義神經網路結構
class QNetwork(nn.Module):
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
    def __init__(self, observation_space, action_space, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        super().__init__(action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # 取得輸入形狀
        img_shape = observation_space['image'].shape
        scalar_dim = observation_space['scalars'].shape[0]
        
        # [修改 1] 初始化兩個網路：Policy Net (訓練用) 和 Target Net (計算目標用)
        self.policy_net = QNetwork(img_shape, scalar_dim, action_space.n).to(device)
        self.target_net = QNetwork(img_shape, scalar_dim, action_space.n).to(device)
        
        # 一開始先同步權重
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target Net 不需要計算梯度，設為評估模式
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss() # [建議] 改用 Huber Loss (SmoothL1) 比 MSE 更穩定
        
        # [修改 2] 加大記憶體
        self.memory = deque(maxlen=50000) 
        self.batch_size = 512
        self.learn_step_counter = 0 # 計數器，用來決定何時更新 Target Net

    def get_action(self, state):
        # Epsilon-Greedy 策略
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # 處理輸入資料
        img_tensor = torch.FloatTensor(state['image']).unsqueeze(0).to(device)     # (1, 7, H, W)
        scalar_tensor = torch.FloatTensor(state['scalars']).unsqueeze(0).to(device) # (1, 1)
        
        with torch.no_grad():
            q_values = self.policy_net(img_tensor, scalar_tensor)
            
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
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

        # [修改 3] 計算 Q 值
        # Current Q: 由 Policy Net 計算
        curr_q = self.policy_net(batch_imgs, batch_scalars).gather(1, batch_actions)
        
        # Target Q: 由 Target Net 計算 (關鍵！)
        with torch.no_grad():
            next_q = self.target_net(batch_next_imgs, batch_next_scalars).max(1)[0].unsqueeze(1)
            target_q = batch_rewards + (1 - batch_dones) * self.gamma * next_q
            
        loss = self.loss_fn(curr_q, target_q)
        
        # 5. 更新網路
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # [修改 4] 定期更新 Target Net (例如每學習 1000 次同步一次)
        self.learn_step_counter += 1
        if self.learn_step_counter % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Target Network Updated")

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
        print(f"CNN Model saved to {filename}")

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 載入時也要同步
        self.policy_net.eval()
        print(f"CNN Model loaded from {filename}")