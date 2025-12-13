import gymnasium as gym
from agent import QLearningAgent, DoubleDQNAgent, DQNAgent, PPOAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np
import argparse
from human import human_mode
from dataclasses import dataclass
from typing import Callable

import warnings
# 靜音 pygame/pkg_resources 的棄用警告
warnings.filterwarnings("ignore", category=UserWarning, module=r"pygame\.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")


# ============================================================================
# Agent spec (centralized configuration)
# ============================================================================
@dataclass(frozen=True)
class AgentSpec:
    name: str
    env_id: str
    model_filename: str
    build: Callable


def normalize_agent_type(agent_type: str) -> str:
    if agent_type == "DDQN":
        return "DoubleDQN"
    return agent_type


def _build_qlearning(env, *, training: bool, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, **_):
    return QLearningAgent(
        env.action_space,
        learning_rate=learning_rate,
        discount_factor=gamma,
        epsilon=epsilon if training else 0.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
    )


def _build_dqn(env, *, training: bool, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq, **_):
    return DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon if training else 0.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        batch_size=batch,
        memory_size=memory,
        target_update_freq=target_update_freq,
    )


def _build_double_dqn(env, *, training: bool, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch, memory, target_update_freq, **_):
    return DoubleDQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon if training else 0.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        batch_size=batch,
        memory_size=memory,
        target_update_freq=target_update_freq,
    )


def _build_ppo(env, *, training: bool, learning_rate, gamma, batch, target_update_freq, gae_lambda, policy_clip, n_epochs, **_):
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
        update_interval=target_update_freq,
    )


AGENT_SPECS = {
    "QLearning": AgentSpec(name="QLearning", env_id="dungeon-crawler-dqn-v0", model_filename="final_model.pkl", build=_build_qlearning),
    "DQN": AgentSpec(name="DQN", env_id="dungeon-crawler-cnn-v0", model_filename="final_model.pth", build=_build_dqn),
    "DoubleDQN": AgentSpec(name="DoubleDQN", env_id="dungeon-crawler-cnn-v0", model_filename="final_model.pth", build=_build_double_dqn),
    "PPO": AgentSpec(name="PPO", env_id="dungeon-crawler-ppo-v0", model_filename="final_model.pth", build=_build_ppo),
}


def get_spec(agent_type: str) -> AgentSpec:
    key = normalize_agent_type(agent_type)
    if key not in AGENT_SPECS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_SPECS[key]
    
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

    agent_spec = get_spec(agent_type)
    
    env = gym.make(agent_spec.env_id, render_mode=None)
    agent = agent_spec.build(
        env,
        training=True,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        batch=batch,
        memory=memory,
        target_update_freq=target_update_freq,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        n_epochs=n_epochs,
    )

    # 設定路徑
    base_dir = f"Result/{agent_spec.name}Agent"
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
            total_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                # 1. 獲取動作
                action = agent.get_action(state, deterministic=False)
                
                # 2. 執行動作
                next_state, reward, done, truncated, info = env.step(action)
                
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
    model_path = os.path.join(result_dir, agent_spec.model_filename)
    agent.save(model_path)
    
    # 對 PPO 進行確定性評估
    if agent_spec.name == "PPO":
        print("\n" + "="*60)
        print("Running deterministic evaluation (greedy policy)...")
        print("="*60)
        mean_reward, std_reward = evaluate(agent, agent_spec.env_id, num_episodes=50, render=False)
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
def evaluate(agent, env_id, num_episodes=50, render=False):
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
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.get_action(state, deterministic=True)
            
            next_state, reward, done, truncated, info = env.step(action)
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

    agent_spec = get_spec(agent_type)
    
    env = gym.make(agent_spec.env_id, render_mode='human')
    agent = agent_spec.build(
        env,
        training=False,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=0.0,
        epsilon_decay=1.0,
        min_epsilon=0.0,
        batch=512,
        memory=50000,
        target_update_freq=1000,
        gae_lambda=0.95,
        policy_clip=0.2,
        n_epochs=10,
    )
    
    # 設定路徑
    base_dir = f"Result/{agent_spec.name}Agent"
    result_dir = os.path.join(base_dir, "result")
    
    # 讀取最終模型
    model_path = os.path.join(result_dir, agent_spec.model_filename)
    
    try:
        agent.load(model_path)
        print(f"Successfully loaded model: {model_path}")
    except FileNotFoundError:
        print(f"No trained model found at {model_path}. Please train first.")
        env.close()
        return

    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    print("Start Testing...")
    while not (done or truncated):
        action = agent.get_action(state, deterministic=True)
        
        next_state, reward, done, truncated, info = env.step(action)
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
    parser.add_argument("--agent", type=str, default="DQN", choices=["QLearning", "DQN", "PPO", "DoubleDQN", "DDQN"], help="Type of agent to use")
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
        agent_spec = get_spec(AGENT_TYPE)

        env = gym.make(agent_spec.env_id)
        agent = agent_spec.build(
            env,
            training=False,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            batch=BATCH,
            memory=MEMORY,
            target_update_freq=TARGET_UPDATE_FREQ,
            gae_lambda=GAE_LAMBDA,
            policy_clip=POLICY_CLIP,
            n_epochs=N_EPOCHS,
        )
        env.close()
        
        # 載入模型
        base_dir = f"Result/{agent_spec.name}Agent"
        result_dir = os.path.join(base_dir, "result")
        model_path = os.path.join(result_dir, agent_spec.model_filename)
        
        try:
            agent.load(model_path)
            print(f"Successfully loaded model: {model_path}")
        except FileNotFoundError:
            print(f"No trained model found at {model_path}. Please train first.")
            exit(1)
        
        # 評估
        mean_reward, std_reward = evaluate(agent, agent_spec.env_id, num_episodes=EPISODES, render=False)
        print(f"\nEvaluation Results ({EPISODES} episodes):")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    else:
        test(AGENT_TYPE)




# python train.py --agent PPO --episodes 2000 --learning_rate 0.0001 --batch 64 --target_update_freq 1024 --gae_lambda 0.95 --policy_clip 0.2 --n_epochs 10
