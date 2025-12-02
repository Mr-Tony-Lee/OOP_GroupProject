import gymnasium as gym
from agent import QLearningAgent, DQNAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np

def preprocess_state(state, agent_type):
    if agent_type == "QLearning":
        # state shape: (8, rows, cols)
        # Find player position (Channel 1)
        player_pos = np.where(state[1] == 1)
        if len(player_pos[0]) > 0:
            row, col = player_pos[0][0], player_pos[1][0]
        else:
            row, col = 0, 0 # Should not happen
            
        # Check has_key (Channel 7)
        has_key = 1 if state[7, 0, 0] == 1 else 0
        
        return (row, col, has_key)
    return state

def get_agent(agent_type, env):
    if agent_type == "QLearning":
        return QLearningAgent(
            env.action_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.999
        )
    elif agent_type == "DQN":
        state_shape = env.observation_space.shape # (C, H, W)
        return DQNAgent(
            state_shape=state_shape,
            action_space=env.action_space,
            learning_rate=0.0001, # Lower LR for CNN
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.9995,
            min_epsilon=0.01,
            batch_size=64,
            memory_size=50000
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def train(agent_type="DQN"):
    print(f"Training {agent_type} Agent...")
    
    # 建立環境
    env = gym.make('dungeon-crawler-v0', render_mode=None)
    
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

    episodes = 5000 
    rewards_history = []
    best_reward = -float('inf')

    log_path = os.path.join(log_dir, "training_log.txt")
    event_log_path = os.path.join(log_dir, "event_log.txt")

    with open(log_path, "w") as log_file, open(event_log_path, "w") as event_file:
        log_file.write("Start Training...\n")
        event_file.write("Episode, Event\n")
        print(f"Start Training... (Logging to {log_path})")
        
        for episode in range(episodes):
            state, info = env.reset()
            state = preprocess_state(state, agent_type) # Preprocess
            total_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                action = agent.get_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_state(next_state, agent_type) # Preprocess
                
                # Log events
                if "events" in info:
                    for event in info["events"]:
                        event_file.write(f"{episode+1}, {event}\n")

                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)
            
            # Save Best Model
            if total_reward > best_reward:
                best_reward = total_reward
                model_filename = "best_model.pkl" if agent_type == "QLearning" else "best_model.pth"
                model_path = os.path.join(result_dir, model_filename)
                agent.save(model_path)

            # 只在每 50 回合 flush 一次
            if (episode + 1) % 50 == 0:
                log_msg = f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.2f}\n"
                log_file.write(log_msg)
                log_file.flush()
                event_file.flush() 

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

def test(agent_type="DQN"):
    print(f"Testing {agent_type} Agent...")
    
    env = gym.make('dungeon-crawler-v0', render_mode='human')
    
    # 建立 Agent (Epsilon=0)
    if agent_type == "QLearning":
        agent = QLearningAgent(env.action_space, epsilon=0.0)
    elif agent_type == "DQN":
        state_shape = env.observation_space.shape
        agent = DQNAgent(state_shape, env.action_space, epsilon=0.0)
    
    # 設定路徑
    base_dir = f"{agent_type}Agent"
    result_dir = os.path.join(base_dir, "result")
    
    # 優先嘗試讀取 Best Model
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
    state = preprocess_state(state, agent_type) # Preprocess
    done = False
    truncated = False
    total_reward = 0
    
    print("Start Testing...")
    while not (done or truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_state(next_state, agent_type) # Preprocess
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

if __name__ == "__main__":
    # 選擇要使用的 Agent: "QLearning" 或 "DQN"
    # AGENT_TYPE = "DQN" 
    AGENT_TYPE = "QLearning" 
    
    train(AGENT_TYPE)
    # test(AGENT_TYPE)
