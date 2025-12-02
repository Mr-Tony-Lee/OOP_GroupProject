import gymnasium as gym
from agent import CNNAgent
import matplotlib.pyplot as plt
import os
import dungeon_env
import numpy as np

def train():
    # 建立環境 (不開啟 render 以加快訓練速度)
    env = gym.make('dungeon-crawler-v0', render_mode=None)
    
    # 建立 Agent (改用 CNNAgent)
    agent = CNNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998, # CNN 需要多一點時間探索
        min_epsilon=0.1
    )

    episodes = 2000 # 增加回合數以利收斂
    rewards_history = []

    # 開啟 log 檔案
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    # 同時開啟 training_log 和 event_log
    with open("logs/training_log.txt", "w") as log_file, open("logs/event_log.txt", "w") as event_file:
        log_file.write("Start Training with CNN...\n")
        event_file.write("Episode, Event\n")
        print("Start Training... (Logging to training_log.txt and event_log.txt)")
        
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            truncated = False
            step_count = 0

            while not (done or truncated):
                # 1. 獲取動作
                action = agent.get_action(state)
                
                # 2. 執行動作
                next_state, reward, done, truncated, info = env.step(action)
                
                # --- 保留 Event Log 功能 ---
                if "events" in info:
                    for event in info["events"]:
                        event_file.write(f"{episode+1}, {event}\n")
                # -------------------------

                # 3. 學習
                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # 強制終止過長的回合 (避免卡死)
                if step_count > 300:
                    truncated = True

            rewards_history.append(total_reward)
            event_file.flush() # 確保即時寫入
            
            # 定期記錄訓練狀況
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                log_msg = f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}\n"
                log_file.write(log_msg)
                log_file.flush()
                print(log_msg.strip())

        log_file.write("Training Finished!\n")
        print("Training Finished!")
    
    if not os.path.exists("result"):
        os.makedirs("result")
    
    # 儲存模型
    agent.save("result/cnn_model.pth")
    
    # 繪製訓練曲線
    plt.plot(rewards_history)
    plt.title("CNN Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("result/training_curve.png")
    
    print("Training curve saved to result/training_curve.png")
    
    env.close()

def test():
    # 測試模式：開啟 render，並載入訓練好的模型
    env = gym.make('dungeon-crawler-v0', render_mode='human')
    # 測試時 epsilon 設為 0
    agent = CNNAgent(env.observation_space, env.action_space, epsilon=0.0)
    
    try:
        agent.load("result/cnn_model.pth")
    except FileNotFoundError:
        print("No trained model found. Please train first.")
        return

    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    print("Start Testing...")
    while not (done or truncated):
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Test Finished. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train()
    # test()