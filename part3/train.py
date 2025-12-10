import gymnasium as gym
from agent import QLearningAgent, DoubleDQNAgent, DQNAgent, PPOAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np
import argparse
from human import human_mode

# ============================================================================
# 狀態預處理函數
# ============================================================================
def preprocess_state(state, agent_type):
    """
    根據 agent 類型預處理狀態
    
    - QLearning: 需要 (row, col, has_key) 三元組
    - DoubleDQN: 需要 (8, H, W) 的 numpy array
    - DQN: 需要 Dict {"image": ..., "scalars": ...}
    - PPO: 需要 (8, H, W) 的 numpy array
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
    
    elif agent_type == "DQN":
        # CNN 環境返回 Dict，直接返回
        return state
    
    elif agent_type == "PPO":
        return state
    
    else:  # DoubleDQN
        # DoubleDQN 現在也使用 Dict 輸入 (Image + Scalar)
        return state


# ============================================================================
# Agent 工廠函數
# ============================================================================
def get_agent(agent_type, env, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq):
    """根據 agent 類型創建相應的 agent"""
    if agent_type == "QLearning":
        return QLearningAgent(
            env.action_space,
            learning_rate=learning_rate,
            discount_factor=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay
        )
    
    elif agent_type == "DQN":
        # DQN (原 CNNAgent)
        return DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=learning_rate,  
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,  
            min_epsilon=min_epsilon,
            batch_size=batch,       
            memory_size=memory,
            target_update_freq=target_update_freq 
        )
    
    elif agent_type == "DoubleDQN":
        # DoubleDQN (原 DQNAgent)
        return DoubleDQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=learning_rate,  
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,  
            min_epsilon=min_epsilon,
            batch_size=batch,       
            memory_size=memory,
            target_update_freq=target_update_freq 
        )
    
    elif agent_type == "PPO":
        state_shape = env.observation_space.shape
        return PPOAgent(
            state_shape=state_shape,
            action_space=env.action_space,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=batch,
            n_epochs=10,
            update_interval=target_update_freq
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================================
# 環境選擇函數
# ============================================================================
def get_env_id(agent_type):
    """根據 agent 類型選擇合適的環境"""
    if agent_type == "DQN" or agent_type == "DoubleDQN":
        return 'dungeon-crawler-cnn-v0'
    elif agent_type == "PPO":
        return 'dungeon-crawler-ppo-v0'
    else:  # QLearning
        return 'dungeon-crawler-dqn-v0'


# ============================================================================
# 訓練函數
# ============================================================================
def train(agent_type="DQN" , episodes=2000 , learning_rate=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.05, batch=512, memory=50000, target_update_freq=1000):
    """
    訓練指定類型的 Agent
    
    支援的 agent_type:
    - "QLearning": 傳統 Q-Learning
    - "DoubleDQN": Double Deep Q-Network
    - "DQN": Standard DQN (formerly CNN based)
    """
    print(f"Training {agent_type} Agent...")
    
    # 根據 agent 類型選擇環境
    env_id = get_env_id(agent_type)
    env = gym.make(env_id, render_mode=None)
    
    # 建立 Agent
    agent = get_agent(agent_type, env, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq)

    # 設定路徑
    base_dir = f"Result/{agent_type}Agent"
    log_dir = os.path.join(base_dir, "log")
    result_dir = os.path.join(base_dir, "result")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    rewards_history = []

    log_path = os.path.join(log_dir, "training_log.txt")
    event_log_path = os.path.join(log_dir, "event_log.txt")

    with open(log_path, "w") as log_file, open(event_log_path, "w") as event_file:
        log_file.write(f"Start Training with {agent_type}...\n")
        event_file.write("Episode, Event\n")
        print(f"Start Training... (Logging to {log_path})")
        best_reward = -float('inf')
        for episode in range(episodes):
            state, info = env.reset()
            state = preprocess_state(state, agent_type)  # Preprocess
            total_reward = 0
            done = False
            truncated = False

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

            rewards_history.append(total_reward)
            event_file.flush()  # 確保即時寫入
            
            # 更新最佳獎勵
            if total_reward > best_reward:
                best_reward = total_reward
            
            # 定期記錄訓練狀況
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                if hasattr(agent, 'epsilon'):
                    log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {agent.epsilon:.4f}\n"
                else:
                    log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}\n"
                log_file.write(log_msg)
                log_file.flush()
                print(log_msg.strip())
                best_reward = -float('inf')

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
    - "DoubleDQN": Double Deep Q-Network
    - "DQN": Standard DQN (formerly CNN based)
    - "PPO": Proximal Policy Optimization
    """
    print(f"Testing {agent_type} Agent...")
    
    # 根據 agent 類型選擇環境
    env_id = get_env_id(agent_type)
    env = gym.make(env_id, render_mode='human')
    
    # 建立 Agent (Epsilon=0 用於貪心策略)
    if agent_type == "QLearning":
        agent = QLearningAgent(env.action_space, epsilon=0.0)
    elif agent_type == "DQN":
        agent = DQNAgent(env.observation_space, env.action_space, epsilon=0.0)
    elif agent_type == "DoubleDQN":
        agent = DoubleDQNAgent(env.observation_space, env.action_space, epsilon=0.0)
    elif agent_type == "PPO":
        state_shape = env.observation_space.shape
        agent = PPOAgent(state_shape=state_shape, action_space=env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # 設定路徑
    base_dir = f"Result/{agent_type}Agent"
    result_dir = os.path.join(base_dir, "result")
    
    # 讀取最終模型
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
    env.close()


# ============================================================================
# 主程式
# ============================================================================
if __name__ == "__main__":

    # 從 command 讀入資訊
    parser = argparse.ArgumentParser(description="Train or test an agent")
    parser.add_argument("--agent", type=str, default="DQN", choices=["QLearning", "DDQN", "DQN", "PPO"], help="Type of agent to use")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "human"], help="Mode to run the agent")
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, help="Epsilon decay rate")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--memory", type=int, default=50000, help="Memory size")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target update frequency")

    args = parser.parse_args()
    if args.agent == "DDQN":
        args.agent = "DoubleDQN"
    AGENT_TYPE = args.agent
    EPISODES = args.episodes
    MODE = args.mode
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    EPSILON = args.epsilon
    EPSILON_DECAY = args.epsilon_decay
    MIN_EPSILON = args.min_epsilon
    BATCH = args.batch
    MEMORY = args.memory
    TARGET_UPDATE_FREQ = args.target_update_freq
    
    if MODE == "human":
        human_mode()
    elif MODE == "train":
        train(AGENT_TYPE, EPISODES, LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON, BATCH, MEMORY, TARGET_UPDATE_FREQ)
    else:
        test(AGENT_TYPE)