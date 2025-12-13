import optuna
import numpy as np
import joblib
import os
from t import DPAgent
from s import SARSAAgent
from Agent import QLearningAgent, SARSAAgent
def logging_callback(study, frozen_trial):
    try:
        best_trial = study.best_trial
    except:
        return

    if frozen_trial.number == best_trial.number:
        with open("optuna_log.txt", "a") as f:
            f.write(f"New Best Trial Found: {frozen_trial.number}\n")
            f.write(f"Value: {frozen_trial.value}\n")
            f.write(f"Params: {frozen_trial.params}\n")
            f.write("--------------------------------------------------\n")

# def objective(trial):
#     # 定義要調整的超參數範圍
#     learning_rate = trial.suggest_float("learning_rate", 0.05, 0.5)
#     discount_factor = trial.suggest_float("discount_factor", 0.9, 0.9999)
#     exploration_decay_rate = trial.suggest_float("exploration_decay_rate", 0.00001, 0.001)
    
#     step_penalty = trial.suggest_float("step_penalty", 0.0001, 0.1)
#     closer_to_goal_reward = trial.suggest_float("closer_to_goal_reward", 0.001, 0.5)
    
#     # 建立 Agent 並設定參數
#     agent = QLearningAgent(
#         agent_type="QLearning",
#         map_name="8x8",
#         episodes=15000,  # 減少 episodes 以加快搜尋速度
#         is_training=True,
#         is_slippery=True,
#         is_cheating=False,
#         save_artifacts=False # 關閉檔案儲存以避免衝突
#     )
    
#     # 覆蓋 Agent 的參數
#     agent.learning_rate_a = learning_rate
#     agent.discount_factor_g = discount_factor
#     agent.exploration_decay_rate = exploration_decay_rate
#     agent.step_penalty = step_penalty
#     agent.closer_to_goal_reward = closer_to_goal_reward
    
#     # 訓練
#     agent.train()
#     average_success_rate = 0.0
#     for i in range(5):
#         # 測試 (使用訓練好的 Q-table)
#         agent.is_training = False
#         agent.episodes = 1000 # 測試 1000 次
#         average_success_rate += agent.test()
#     average_success_rate /= 5.0
#     return average_success_rate

# def objective(trial):
#     # DPAgent 
#     # gamma = trial.suggest_float("gamma", 0.97, 0.9999)
#     # theta = trial.suggest_float("theta", 1e-8, 1e-6)
#     # step_penalty = trial.suggest_float("step_penalty", 1e-4, 5e-2)
#     hole_penalty_weight = trial.suggest_float("hole_penalty_weight", 0.0, 1.0)
#     # 建立 Agent 並設定參數
#     agent = DPAgent(
#         map_name="8x8",
#         is_slippery=True,
#     )
#     # agent.step_penalty = step_penalty
#     # agent.gamma = gamma
#     # agent.theta = theta
#     agent.hole_penalty_weight = hole_penalty_weight
#     average_success_rate = 0.0
#     for i in range(5):
#         average_success_rate += agent.run_evaluation(episodes=1000)
#     average_success_rate /= 5.0
#     return average_success_rate

def objective(trial):
    # 定義要調整的超參數範圍
    learning_rate = trial.suggest_float("learning_rate", 0.05, 0.5)
    discount_factor = trial.suggest_float("discount_factor", 0.9, 0.9999)
    exploration_decay_rate = trial.suggest_float("exploration_decay_rate", 0.00001, 0.001)
    
    step_penalty = trial.suggest_float("step_penalty", 0.0001, 0.1)
    closer_to_goal_reward = trial.suggest_float("closer_to_goal_reward", 0.001, 0.5)
    
    # 建立 Agent 並設定參數
    agent = SARSAAgent(
        agent_type="SARSA",
        map_name="8x8",
        episodes=15000,  # 減少 episodes 以加快搜尋速度
        is_training=True,
        is_slippery=True,
        is_cheating=False,
        save_artifacts=False # 關閉檔案儲存以避免衝突
    )
    
    # 覆蓋 Agent 的參數
    agent.learning_rate_a = learning_rate
    agent.discount_factor_g = discount_factor
    agent.exploration_decay_rate = exploration_decay_rate
    agent.step_penalty = step_penalty
    agent.closer_to_goal_reward = closer_to_goal_reward
    
    # 訓練
    agent.train()
    average_success_rate = 0.0
    for i in range(5):
        # 測試 (使用訓練好的 Q-table)
        agent.is_training = False
        agent.episodes = 1000 # 測試 1000 次
        average_success_rate += agent.test()
    average_success_rate /= 5.0
    return average_success_rate

def run_optimization(n_trials):
    # 載入共享的 study
    study = optuna.load_study(study_name="frozen_lake_optimization", storage="sqlite:///db.sqlite3")
    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])

if __name__ == "__main__":
    # 刪除舊的資料庫檔案 (如果存在)，以便重新開始
    if os.path.exists("db.sqlite3"):
        os.remove("db.sqlite3")
        
    if os.path.exists("optuna_log.txt"):
        os.remove("optuna_log.txt")

    # 建立共享的 study
    study = optuna.create_study(
        study_name="frozen_lake_optimization", 
        storage="sqlite:///db.sqlite3", 
        direction="maximize"
    )
    
    n_jobs = 5  # 設定平行執行的數量 (例如 CPU 核心數)
    total_trials = 10000
    trials_per_job = total_trials // n_jobs
    
    print(f"Starting optimization with {n_jobs} jobs, {total_trials} total trials...")
    
    # 使用 joblib 進行平行運算
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_optimization)(trials_per_job) for _ in range(n_jobs)
    )

    # 重新載入 study 以獲取最佳結果
    study = optuna.load_study(study_name="frozen_lake_optimization", storage="sqlite:///db.sqlite3")
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
