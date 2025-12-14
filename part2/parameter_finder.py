import gymnasium as gym
import numpy as np
import optuna
from numba import njit

# =========================================================
# 1. 環境預處理 (CDF 矩陣)
# =========================================================
def get_frozenlake_transitions(map_name="8x8", is_slippery=True):
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = env.unwrapped.P
    
    # 建立 [State, Action, Outcome_Index, Data]
    # Data: [CDF, Next_State, Reward, Is_Terminated]
    max_outcomes = 3
    transitions = np.zeros((n_states, n_actions, max_outcomes, 4), dtype=np.float64)
    
    for s in range(n_states):
        for a in range(n_actions):
            outcomes = P[s][a]
            current_prob_sum = 0.0
            for i, (prob, next_state, reward, term) in enumerate(outcomes):
                current_prob_sum += prob
                transitions[s, a, i] = [current_prob_sum, next_state, reward, float(term)]
            
            if len(outcomes) < max_outcomes:
                last_valid = transitions[s, a, len(outcomes)-1]
                for i in range(len(outcomes), max_outcomes):
                    transitions[s, a, i] = [1.0, last_valid[1], last_valid[2], last_valid[3]]

    env.close()
    return n_states, n_actions, transitions

N_STATES, N_ACTIONS, TRANSITIONS_MATRIX = get_frozenlake_transitions()

# =========================================================
# 2. Numba 核心邏輯 (雙層迴圈)
# =========================================================
@njit(fastmath=True)
def run_hybrid_evaluation(lr_init, df, decay, min_eps, lr_stable, train_episodes):
    
    outer_runs = 5  # 外層重複做 5 次
    test_sessions = 5 # 內層 Test 5 次
    test_episodes = 100 # 每次 Test 測 100 局
    
    total_outer_score = 0.0
    
    # --- 外層迴圈：5 次獨立的訓練與評估 ---
    for _ in range(outer_runs):
        
        # === 1. 初始化 (每次 Run 都要重置) ===
        q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        epsilon = 1.0
        learning_rate = lr_init
        
        # 用來記錄訓練階段最後 100 局的勝敗
        train_rewards = np.zeros(train_episodes, dtype=np.float64)
        
        # === 2. 訓練階段 (Train) ===
        for i in range(train_episodes):
            state = 0
            terminated = False
            truncated = False
            step = 0
            
            while not terminated and not truncated:
                # Action (Epsilon-Greedy)
                if np.random.random() < epsilon:
                    action = np.random.randint(0, N_ACTIONS)
                else:
                    best_val = -1e10
                    action = 0
                    for a in range(N_ACTIONS):
                        if q[state, a] > best_val:
                            best_val = q[state, a]
                            action = a
                
                # Step (CDF)
                r_val = np.random.random()
                outcome_data = TRANSITIONS_MATRIX[state, action]
                
                if r_val < outcome_data[0, 0]:
                    outcome = outcome_data[0]
                elif r_val < outcome_data[1, 0]:
                    outcome = outcome_data[1]
                else:
                    outcome = outcome_data[2]
                
                next_state = int(outcome[1])
                reward = outcome[2]
                terminated_bool = outcome[3] > 0.5
                
                # Update Q
                max_next_q = -1e10
                for a in range(N_ACTIONS):
                    if q[next_state, a] > max_next_q:
                        max_next_q = q[next_state, a]
                
                old_val = q[state, action]
                q[state, action] = old_val + learning_rate * (
                    reward + df * max_next_q - old_val
                )
                
                state = next_state
                step += 1
                if step >= 100: truncated = True
                if terminated_bool:
                    terminated = True
                    if reward == 1.0:
                        train_rewards[i] = 1.0
            
            # Decay Logic
            epsilon = max(epsilon - decay, min_eps)
            if epsilon <= min_eps:
                learning_rate = lr_stable
        
        # 計算訓練分數 (S_train): 最後 100 局的勝率
        train_score = 0.0
        start_idx = max(0, train_episodes - 100)
        window_len = train_episodes - start_idx
        for k in range(start_idx, train_episodes):
            train_score += train_rewards[k]
        train_score = (train_score / window_len) * 100.0

        # === 3. 測試階段 (Test x 5) ===
        sum_test_scores = 0.0
        
        for _ in range(test_sessions):
            current_test_success = 0
            for _ in range(test_episodes):
                state = 0
                terminated = False
                truncated = False
                step = 0
                while not terminated and not truncated:
                    # Greedy Action
                    best_val = -1e10
                    action = 0
                    for a in range(N_ACTIONS):
                        if q[state, a] > best_val:
                            best_val = q[state, a]
                            action = a
                    
                    # Step
                    r_val = np.random.random()
                    outcome_data = TRANSITIONS_MATRIX[state, action]
                    if r_val < outcome_data[0, 0]: outcome = outcome_data[0]
                    elif r_val < outcome_data[1, 0]: outcome = outcome_data[1]
                    else: outcome = outcome_data[2]
                    
                    state = int(outcome[1])
                    reward = outcome[2]
                    terminated_bool = outcome[3] > 0.5
                    
                    step += 1
                    if step >= 100: truncated = True
                    if terminated_bool:
                        terminated = True
                        if reward == 1.0:
                            current_test_success += 1
            
            # 累加這次 Test 的勝率
            sum_test_scores += (current_test_success / test_episodes) * 100.0
            
        # === 4. 計算單次 Run 的綜合分數 ===
        # (1次 Train分數 + 5次 Test分數) / 6
        run_score = (train_score + sum_test_scores) / (1 + test_sessions)
        
        total_outer_score += run_score

    # === 5. 回傳 5 次 Runs 的平均 ===
    return total_outer_score / outer_runs

# =========================================================
# 3. Optuna Objective
# =========================================================
def objective(trial):
    # 定義搜尋空間
    lr = trial.suggest_float('lr', 0.01, 1.0)
    df = trial.suggest_float('df', 0.85, 0.9999)
    decay = trial.suggest_float('decay', 1e-5, 5e-4, log=True)
    min_eps = trial.suggest_float('min_eps', 0.0, 0.2)
    lr_stable = trial.suggest_float('lr_stable', 0.0, 0.5)
    
    train_episodes = 15000
    
    # 執行 Numba 函數
    final_score = run_hybrid_evaluation(lr, df, decay, min_eps, lr_stable, train_episodes)
    
    return final_score

# =========================================================
# 4. 主程式
# =========================================================
if __name__ == '__main__':
    print("正在編譯 Numba 函數 (複雜評估邏輯)...")
    # Warmup
    run_hybrid_evaluation(0.1, 0.99, 0.001, 0.01, 0.001, 100)
    print("編譯完成，開始 Optuna 搜尋！")
    
    study = optuna.create_study(direction='maximize')
    
    # 由於運算量大增 (每次 Trial = 5 Runs * (15000 Train + 500 Test))
    # 但因為是純 C 級別運算，依然能在幾秒內跑完一個 Trial
    study.optimize(objective, n_trials=1000)
    
    print("\n" + "="*40)
    print("🏆 最佳參數組合 (5 Runs x (1 Train + 5 Tests) Avg):")
    print("="*40)
    
    best = study.best_params
    
    print(best)
    
    print(f"綜合評分: {study.best_value:.2f}%")