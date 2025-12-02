import numpy as np
import random
from abc import ABC, abstractmethod
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Agent(ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
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

class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        # input_shape: (C, H, W) -> (8, 11, 12)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size
        # 11x12 -> 11x12 (padding=1, stride=1 keeps size same)
        flatten_size = 64 * input_shape[1] * input_shape[2]
        
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )

class DQNAgent(Agent):
    def __init__(self, state_shape, action_space, learning_rate=0.0001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01, batch_size=64, memory_size=50000):
        super().__init__(action_space)
        self.state_shape = state_shape # (C, H, W)
        self.action_dim = action_space.n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        
        # 使用優化後的 Replay Buffer
        self.memory = ReplayBuffer(memory_size, state_shape, self.action_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")
        
        self.q_network = DQN(self.state_shape, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_shape, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.steps = 0
        self.target_update_freq = 1000 

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # state is (C, H, W), need (1, C, H, W)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.steps += 1
        
        if self.memory.size < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN Logic
        # 1. Select action using Online Network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # 2. Evaluate action using Target Network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
            
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if done:
             self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
             
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        print(f"DQN Model saved to {filename}")

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()
        print(f"DQN Model loaded from {filename}")
