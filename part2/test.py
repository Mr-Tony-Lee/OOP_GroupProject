import argparse
from Agent import QLearningAgent
import matplotlib.pyplot as plt

def Training(agent, map_name = "8x8", episodes=15000, is_slippery = True, render_mode="ansi", is_cheating=False, 
             learning_rate_a=0.5,learning_rate_decay = 0.9995, discount_factor_g=0.99, exploration_decay_rate=0.00015,
             step_penalty=0.00035, closer_to_goal_reward=0.005):
    """
    訓練指定的代理人 (Agent) 在 Frozen Lake 環境中學習。
        - agent: 代理人類別 (例如 QLearningAgent)
        - map_name: 地圖名稱 (例如 "4x4" 或 "8x8")
        - episodes: 訓練的回合數
        - is_slippery: 是否使用滑動地面
        - render_mode: 環境渲染模式
        - is_cheating: 是否啟用作弊模式 (使用環境的轉移機率)
    """  
    if agent == "QLearning":
        agent = QLearningAgent(agent_type=agent, map_name=map_name, episodes=episodes, is_training=True, is_slippery=is_slippery, render_mode=render_mode, is_cheating=is_cheating, 
                               learning_rate_a=learning_rate_a, learning_rate_decay=learning_rate_decay, discount_factor_g=discount_factor_g, exploration_decay_rate=exploration_decay_rate,
                               step_penalty=step_penalty, closer_to_goal_reward=closer_to_goal_reward)
    print(f"Training {map_name}...")
    agent.run()

def Evaluation(agent, map_name = "8x8", episodes=1000, is_slippery = True, render_mode="ansi", is_cheating=False):
    """
    評估指定的代理人 (Agent) 在 Frozen Lake 環境中的表現。
        - agent: 代理人類別 (例如 QLearningAgent)
        - map_name: 地圖名稱 (例如 "4x4" 或 "8x8")
        - episodes: 評估的回合數
        - is_slippery: 是否使用滑動地面
        - render_mode: 環境渲染模式
        - is_cheating: 是否啟用作弊模式 (使用環境的轉移機率)
    """
    if agent == "QLearning":
        agent = QLearningAgent(agent_type=agent, map_name=map_name, episodes=episodes, is_training=False, is_slippery=is_slippery, render_mode=render_mode, is_cheating=is_cheating)
    print(f"\nEvaluating {map_name}...")
    success_rate = agent.run()
    return success_rate

def plot_results(map_name, success_rates, num_runs):
    """
    繪製成功率分佈圖。
        - map_name: 地圖名稱 (例如 "4x4" 或 "8x8")
        - success_rates: 成功率List
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_runs + 1), success_rates, label=f'{map_name} Map')
    plt.xlabel('Run')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate over {num_runs} Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Result/success_rate_distribution.png')

if __name__ == "__main__":
    success_rates = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="8x8", help="Map name")
    parser.add_argument("--agent", type=str, default="QLearning", help="Algorithm to use")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or eval")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--train_episodes", type=int, default=15000, help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--is_slippery", type=bool, default=True, help="Whether the environment is slippery")
    parser.add_argument("--render_mode", type=str, default="ansi", help="Render mode for the environment")
    parser.add_argument("--cheating", type=str, default="false", help="Enable cheating environment")
    args = parser.parse_args()

    map_name = args.map
    agent_type = args.agent
    num_runs = args.runs
    mode = args.mode
    train_episodes = args.train_episodes
    eval_episodes = args.eval_episodes
    is_slippery = args.is_slippery
    render_mode = args.render_mode
    CheatingEnv = (args.cheating.lower() == "true")
    
    
    learning_rate_a = 0.4963125701668435
    discount_factor_g = 0.9559919852791074
    exploration_decay_rate = 0.0004883668598976838
    
    step_penalty = 0.00011214058776498324
    closer_to_goal_reward = 0.4899189987079525
    
    # 0.46 
    if mode == "train":
        print(f"Training Mode: {agent_type} on {map_name} map for {train_episodes} episodes.")
        # Train 8x8
        # Training(agent_type, map_name=map_name, episodes=train_episodes, is_slippery=is_slippery, render_mode=render_mode, is_cheating=CheatingEnv
        #         , learning_rate_a=learning_rate_a, discount_factor_g=discount_factor_g, exploration_decay_rate=exploration_decay_rate
        #         , step_penalty=step_penalty, closer_to_goal_reward=closer_to_goal_reward)
        Training(agent_type, map_name=map_name, episodes=train_episodes, is_slippery=is_slippery, render_mode=render_mode, is_cheating=CheatingEnv)                
    else:
        for i in range(num_runs):
            print(f"--- Run {i+1}/{num_runs} ---")
            # Directly Evaluate without training
            success_rate = Evaluation(agent_type, map_name=map_name, episodes=eval_episodes, is_slippery=is_slippery, render_mode=render_mode, is_cheating=CheatingEnv)
            success_rates.append(success_rate)
        plot_results(map_name, success_rates, num_runs)


# uv run ./test.py --agent QLearning --mode train 
# uv run ./test.py --agent QLearning --mode eval --runs 10
# python3 ./test.py --agent PPO --mode train 
# python3 ./test.py --agent PPO --mode eval --runs 10

