import gymnasium as gym
from agent import QLearningAgent, DQNAgent, CNNAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np

# ============================================================================
# 狀態預處理函數
# ============================================================================
def preprocess_state(state, agent_type):
    """
    根據 agent 類型預處理狀態
    
    - QLearning: 需要 (row, col, has_key) 三元組
    - DQN: 需要 (8, H, W) 的 numpy array
    - CNN: 需要 Dict {"image": ..., "scalars": ...}
    """
    if agent_type == "QLearning":
        # DQN 環境返回 (8, rows, cols)
        # Find player position (Channel 1)
        player_pos = np.where(state[1] == 1)
        if len(player_pos[0]) > 0:
            row, col = player_pos[0][0], player_pos[1][0]
        else:
            row, col = 0, 0  # Should not happen
            
        # Check has_key (Channel 7)
        has_key = 1 if state[7, 0, 0] == 1 else 0
        
        return (row, col, has_key)
    
    elif agent_type == "CNN":
        # CNN 環境返回 Dict，直接返回
        return state
    
    else:  # DQN
        # DQN 環境返回 (8, H, W)，直接返回
        return state


# ============================================================================
# Agent 工廠函數
# ============================================================================
def get_agent(agent_type, env):
    """根據 agent 類型創建相應的 agent"""
    if agent_type == "QLearning":
        return QLearningAgent(
            env.action_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.999
        )
    
    elif agent_type == "CNN":
        # CNN 使用 DQN 環境的 CNN 版本
        return CNNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=0.00025,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.998,  # CNN 需要多一點時間探索
            min_epsilon=0.05
        )
    
    elif agent_type == "DQN":
        state_shape = env.observation_space.shape  # (C, H, W)
        return DQNAgent(
            state_shape=state_shape,
            action_space=env.action_space,
            learning_rate=0.0001,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.9995,
            min_epsilon=0.01,
            batch_size=64,
            memory_size=50000
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================================
# 環境選擇函數
# ============================================================================
def get_env_id(agent_type):
    """根據 agent 類型選擇合適的環境"""
    if agent_type == "CNN":
        return 'dungeon-crawler-cnn-v0'
    else:  # QLearning 或 DQN
        return 'dungeon-crawler-dqn-v0'


# ============================================================================
# 訓練函數
# ============================================================================
def train(agent_type="DQN"):
    """
    訓練指定類型的 Agent
    
    支援的 agent_type:
    - "QLearning": 傳統 Q-Learning
    - "DQN": Deep Q-Network
    - "CNN": CNN-based DQN
    """
    print(f"Training {agent_type} Agent...")
    
    # 根據 agent 類型選擇環境
    env_id = get_env_id(agent_type)
    env = gym.make(env_id, render_mode=None)
    
    # 建立 Agent
    agent = get_agent(agent_type, env)

    # 設定路徑
    base_dir = f"{agent_type}Agent"
    log_dir = os.path.join(base_dir, "log")
    result_dir = os.path.join(base_dir, "result")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    episodes = 2000
    rewards_history = []
    best_reward = -float('inf')

    log_path = os.path.join(log_dir, "training_log.txt")
    event_log_path = os.path.join(log_dir, "event_log.txt")

    with open(log_path, "w") as log_file, open(event_log_path, "w") as event_file:
        log_file.write(f"Start Training with {agent_type}...\n")
        event_file.write("Episode, Event\n")
        print(f"Start Training... (Logging to {log_path})")
        
        for episode in range(episodes):
            state, info = env.reset()
            state = preprocess_state(state, agent_type)  # Preprocess
            total_reward = 0
            done = False
            truncated = False
            step_count = 0

            while not (done or truncated):
                # 1. 獲取動作
                action = agent.get_action(state)
                
                # 2. 執行動作
                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_state(next_state, agent_type)  # Preprocess
                
                # --- 記錄事件 ---
                if "events" in info:
                    for event in info["events"]:
                        event_file.write(f"{episode+1}, {event}\n")
                # ----------------

                # 3. 學習
                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # 強制終止過長的回合 (避免卡死)
                if step_count > 300:
                    truncated = True

            rewards_history.append(total_reward)
            event_file.flush()  # 確保即時寫入
            
            # 保存最佳模型
            if total_reward > best_reward:
                best_reward = total_reward
                model_filename = "best_model.pkl" if agent_type == "QLearning" else "best_model.pth"
                model_path = os.path.join(result_dir, model_filename)
                agent.save(model_path)
            
            # 定期記錄訓練狀況
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {agent.epsilon:.4f}\n"
                log_file.write(log_msg)
                log_file.flush()
                print(log_msg.strip())

        log_file.write("Training Finished!\n")
        print(f"Training Finished! Best Reward: {best_reward:.2f}")
    
    # 儲存最終模型
    model_filename = "final_model.pkl" if agent_type == "QLearning" else "final_model.pth"
    model_path = os.path.join(result_dir, model_filename)
    agent.save(model_path)
    
    # 繪製訓練曲線
    plt.figure()
    plt.plot(rewards_history)
    plt.title(f"Training Progress ({agent_type})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(result_dir, "training_curve.png"))
    print(f"Training curve saved to {os.path.join(result_dir, 'training_curve.png')}")
    
    env.close()


# ============================================================================
# 測試函數
# ============================================================================
def test(agent_type="DQN"):
    """
    測試指定類型的 Agent
    
    支援的 agent_type:
    - "QLearning": 傳統 Q-Learning
    - "DQN": Deep Q-Network
    - "CNN": CNN-based DQN
    """
    print(f"Testing {agent_type} Agent...")
    
    # 根據 agent 類型選擇環境
    env_id = get_env_id(agent_type)
    env = gym.make(env_id, render_mode='human')
    
    # 建立 Agent (Epsilon=0 用於貪心策略)
    if agent_type == "QLearning":
        agent = QLearningAgent(env.action_space, epsilon=0.0)
    elif agent_type == "CNN":
        agent = CNNAgent(env.observation_space, env.action_space, epsilon=0.0)
    elif agent_type == "DQN":
        state_shape = env.observation_space.shape
        agent = DQNAgent(state_shape, env.action_space, epsilon=0.0)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # 設定路徑
    base_dir = f"{agent_type}Agent"
    result_dir = os.path.join(base_dir, "result")
    
    # 優先嘗試讀取最佳模型，再試最終模型
    model_filename = "best_model.pkl" if agent_type == "QLearning" else "best_model.pth"
    model_path = os.path.join(result_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Best model not found at {model_path}, trying final model...")
        model_filename = "final_model.pkl" if agent_type == "QLearning" else "final_model.pth"
        model_path = os.path.join(result_dir, model_filename)

    try:
        agent.load(model_path)
        print(f"Successfully loaded model: {model_path}")
    except FileNotFoundError:
        print(f"No trained model found at {model_path}. Please train first.")
        env.close()
        return

    state, info = env.reset()
    state = preprocess_state(state, agent_type)  # Preprocess
    done = False
    truncated = False
    total_reward = 0
    
    print("Start Testing...")
    while not (done or truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_state(next_state, agent_type)  # Preprocess
        state = next_state
        total_reward += reward
        
        # 處理 Pygame 事件，避免視窗無回應
        if env.unwrapped.render_mode == 'human':
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    print("Test interrupted by user.")

    print(f"Test Finished. Total Reward: {total_reward}")
    
    # 測試結束後暫停一下，讓使用者看到結果
    if env.unwrapped.render_mode == 'human':
        import time
        print("Closing in 3 seconds...")
        time.sleep(3)
        
    env.close()


# ============================================================================
# 主程式
# ============================================================================
if __name__ == "__main__":
    # 選擇要使用的 Agent: "QLearning", "DQN" 或 "CNN"
    AGENT_TYPE = "DQN"
    # AGENT_TYPE = "CNN"
    # AGENT_TYPE = "QLearning"
    
    # train(AGENT_TYPE)
    test(AGENT_TYPE)
