import gymnasium as gym
import dungeon_env
from agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np

def train():
    # 建立環境 (不開啟 render 以加快訓練速度)
    env = gym.make('dungeon-crawler-v0', render_mode=None)
    
    # 建立 Agent
    agent = QLearningAgent(
        env.action_space,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995 # 每次 episode 結束後 epsilon * 0.995
    )

    episodes = 1000
    rewards_history = []

    print("Start Training...")
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("Training Finished!")
    agent.save("q_table.pkl")
    
    # 繪製訓練曲線
    plt.plot(rewards_history)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_curve.png")
    print("Training curve saved to training_curve.png")
    
    env.close()

def test():
    # 測試模式：開啟 render，並載入訓練好的 Q-Table
    env = gym.make('dungeon-crawler-v0', render_mode='human')
    agent = QLearningAgent(env.action_space, epsilon=0.0) # Epsilon=0 代表完全不探索，只選最好的
    
    try:
        agent.load("q_table.pkl")
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
        
        # 稍微慢一點才看得到
        import time
        time.sleep(0.2)

    print(f"Test Finished. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # 你可以選擇要訓練還是測試
    # train()
    # test()
    
    # 為了方便，我們先跑訓練再跑測試
    # train()
    test()
