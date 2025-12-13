import gymnasium as gym
from agent import QLearningAgent, DoubleDQNAgent, DQNAgent, PPOAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np
import argparse
from human import human_mode

import warnings
# 靜音 pygame/pkg_resources 的棄用警告
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")

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
def get_agent(agent_type, env, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq, gae_lambda=0.95, policy_clip=0.2, n_epochs=10):
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
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            batch_size=batch,
            n_epochs=n_epochs,
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
    
import random, torch, numpy as np

# For testing
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 訓練函數
# ============================================================================
def train(agent_type="DQN" , episodes=2000 , learning_rate=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.05, batch=512, memory=50000, target_update_freq=1000, gae_lambda=0.95, policy_clip=0.2, n_epochs=10):
    """
    訓練指定類型的 Agent
    
    支援的 agent_type:
    - "QLearning": 傳統 Q-Learning
    - "DoubleDQN": Double Deep Q-Network
    - "DQN": Standard DQN (formerly CNN based)
    - "PPO": Proximal Policy Optimization
    """
    print(f"Training {agent_type} Agent...")
    set_seed(42)
    
    # 根據 agent 類型選擇環境
    env_id = get_env_id(agent_type)
    env = gym.make(env_id, render_mode=None)
    
    # 建立 Agent
    agent = get_agent(agent_type, env, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq, gae_lambda, policy_clip, n_epochs)

    # 設定路徑
    base_dir = f"Result/{agent_type}Agent"
    log_dir = os.path.join(base_dir, "log")
    result_dir = os.path.join(base_dir, "result")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    rewards_history = []
    entropy_history = []

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
                action = agent.get_action(state, deterministic=False)
                
                # 2. 執行動作
                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_state(next_state, agent_type)  # Preprocess
                
                # --- 記錄事件 ---
                if "events" in info:
                    for event in info["events"]:
                        event_file.write(f"{episode+1}, {event}\n")
                # ----------------

                # 3. 學習
                agent.learn(state, action, reward, next_state, done or truncated)
                
                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)
            
            # 記錄 PPO 特有的指標
            if agent_type == "PPO" and hasattr(agent, 'last_entropy') and agent.last_entropy is not None:
                entropy_history.append(agent.last_entropy)
            
            event_file.flush()  # 確保即時寫入
            
            # 更新最佳獎勵
            if total_reward > best_reward:
                best_reward = total_reward
            
            # 定期記錄訓練狀況
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                if hasattr(agent, 'epsilon'):
                    log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}, Epsilon: {agent.epsilon:.4f}\n"
                elif agent_type == "PPO" and len(entropy_history) > 0:
                    recent_entropy = [e for e in entropy_history[-50:] if e is not None]
                    avg_entropy = np.mean(recent_entropy)
                    log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}, Avg Entropy: {avg_entropy:.4f}\n"
                else:
                    log_msg = f"Episode {episode+1}/{episodes}, Avg Reward (Last 50): {avg_reward:.2f}, Best: {best_reward:.2f}\n"
                log_file.write(log_msg)
                log_file.flush()
                print(log_msg.strip())

        log_file.write("Training Finished!\n")
        print(f"Training Finished! Best Reward: {best_reward:.2f}")
    
    # 儲存最終模型
    model_filename = "final_model.pkl" if agent_type == "QLearning" else "final_model.pth"
    model_path = os.path.join(result_dir, model_filename)
    agent.save(model_path)
    
    # 對 PPO 進行確定性評估
    if agent_type == "PPO":
        print("\n" + "="*60)
        print("Running deterministic evaluation (greedy policy)...")
        print("="*60)
        mean_reward, std_reward = evaluate(agent, env_id, agent_type, num_episodes=50, render=False)
        eval_msg = f"\nDeterministic Evaluation Results (50 episodes):\n"
        eval_msg += f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n"
        eval_msg += f"Training Best Reward (with sampling): {best_reward:.2f}\n"
        print(eval_msg)
        
        # 將評估結果寫入 log
        with open(log_path, "a") as log_file:
            log_file.write("\n" + "="*60 + "\n")
            log_file.write(eval_msg)
            log_file.write("="*60 + "\n")
    
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
# 評估函數 (使用確定性策略)
# ============================================================================
def evaluate(agent, env_id, agent_type, num_episodes=50, render=False):
    """
    評估訓練好的 Agent，使用確定性策略 (greedy action)
    
    Args:
        agent: 訓練好的 agent
        env_id: 環境 ID
        agent_type: Agent 類型
        num_episodes: 評估的 episode 數量
        render: 是否顯示畫面
    
    Returns:
        平均獎勵和標準差
    """
    render_mode = 'human' if render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = preprocess_state(state, agent_type)
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.get_action(state, deterministic=True)
            
            next_state, reward, done, truncated, info = env.step(action)
            next_state = preprocess_state(next_state, agent_type)
            state = next_state
            total_reward += reward
            
            # 處理 Pygame 事件
            if render and env.unwrapped.render_mode == 'human':
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return np.mean(episode_rewards), np.std(episode_rewards) if episode_rewards else 0
        
        episode_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward


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
        action = agent.get_action(state, deterministic=True)
        
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
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train or evaluate")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "eval", "human"], help="Mode to run the agent")
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, help="Epsilon decay rate")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--batch", type=int, default=512, help="Batch size (PPO recommend 64 or 32)")
    parser.add_argument("--memory", type=int, default=50000, help="Memory size")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target update frequency")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda (for PPO)")
    parser.add_argument("--policy_clip", type=float, default=0.2, help="Policy clip epsilon (for PPO)")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update (for PPO)")

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
    GAE_LAMBDA = args.gae_lambda
    POLICY_CLIP = args.policy_clip
    N_EPOCHS = args.n_epochs
    
    if MODE == "human":
        human_mode()
    elif MODE == "train":
        train(AGENT_TYPE, EPISODES, LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON, BATCH, MEMORY, TARGET_UPDATE_FREQ, GAE_LAMBDA, POLICY_CLIP, N_EPOCHS)
    elif MODE == "eval":
        # 評估模式：載入模型並進行確定性評估
        print(f"Evaluating {AGENT_TYPE} Agent...")
        env_id = get_env_id(AGENT_TYPE)
        
        # 建立 Agent
        if AGENT_TYPE == "QLearning":
            agent = QLearningAgent(gym.make(env_id).action_space, epsilon=0.0)
        elif AGENT_TYPE == "DQN":
            env = gym.make(env_id)
            agent = DQNAgent(env.observation_space, env.action_space, epsilon=0.0)
            env.close()
        elif AGENT_TYPE == "DoubleDQN":
            env = gym.make(env_id)
            agent = DoubleDQNAgent(env.observation_space, env.action_space, epsilon=0.0)
            env.close()
        elif AGENT_TYPE == "PPO":
            env = gym.make(env_id)
            state_shape = env.observation_space.shape
            agent = PPOAgent(state_shape=state_shape, action_space=env.action_space)
            env.close()
        else:
            raise ValueError(f"Unknown agent type: {AGENT_TYPE}")
        
        # 載入模型
        base_dir = f"Result/{AGENT_TYPE}Agent"
        result_dir = os.path.join(base_dir, "result")
        model_filename = "final_model.pkl" if AGENT_TYPE == "QLearning" else "final_model.pth"
        model_path = os.path.join(result_dir, model_filename)
        
        try:
            agent.load(model_path)
            print(f"Successfully loaded model: {model_path}")
        except FileNotFoundError:
            print(f"No trained model found at {model_path}. Please train first.")
            exit(1)
        
        # 評估
        mean_reward, std_reward = evaluate(agent, env_id, AGENT_TYPE, num_episodes=EPISODES, render=False)
        print(f"\nEvaluation Results ({EPISODES} episodes):")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    else:
        test(AGENT_TYPE)




# python train.py --agent PPO --episodes 2000 --learning_rate 0.0001 --batch 64 --target_update_freq 1024 --gae_lambda 0.95 --policy_clip 0.2 --n_epochs 10
