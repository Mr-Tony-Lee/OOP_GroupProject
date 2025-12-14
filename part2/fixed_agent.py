import gymnasium as gym
import numpy as np

def run_policy_test(episodes=10000):
    # 建立環境
    # 'FrozenLake-v1', map_name="8x8", is_slippery=True 預設 max_episode_steps=100
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=None)

    # 定義動作變數 (0:Left, 1:Down, 2:Right, 3:Up)
    x = -1
    h = 2 # 定義 x 為向右 (2)，這是根據你提供的 policy_map 補上的變數

    # 定義靜態策略表 (8x8)
    # policy_map = [
    #     [2, 2, 2, 2, 2, 2, 2, 1], # Row 0
    #     [3, 3, 3, 3, 2, 2, 2, 1], # Row 1
    #     [2, 2, 3, h, 3, 3, 2, 1], # Row 2
    #     [3, 3, 3, 2, 3, h, 2, 1], # Row 3
    #     [3, 3, 3, h, 2, 2, 2, 1], # Row 4
    #     [3, h, h, 2, 2, 1, h, 2], # Row 5
    #     [1, h, 2, 3, h, 1, h, 2], # Row 6
    #     [2, 2, 3, h, 2, 2, 2, h], # Row 7
    # ]

    policy_map = [
        [2, 2, 2, 2, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 2, 2, 1],
        [3, 3, 2, h, 2, 3, 2, 1],
        [3, 3, 3, 3, 2, h, 2, 2],
        [x, x, x, h, 2, 1, 3, 2],
        [x, h, h, 2, 3, 0, h, 2],
        [x, h, 3, 3, h, 2, h, 2],
        [x, x, x, h, x, 2, 1, h],
    ]

    # 初始化統計變數
    success_count = 0
    hole_fall_count = 0     # 掉進洞裡的次數 (未達 100 步)
    fail_over_100_count = 0   # 超過 100 步且失敗的次數 (步數 >= 100 的所有失敗)
    fail_over_100_steps_sum = 0 # 用來計算超過 100 步失敗的平均步數
    
    step_history = []

    print(f"🚀 開始測試靜態策略表 (共 {episodes} 次)...")
    print("策略邏輯: 依照定義的 Policy Map 執行")
    print("環境設定: 8x8, 濕滑 (is_slippery=True), 最大步數限制為 100")
    print("-" * 30)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not terminated and not truncated:
            # 1. 將狀態 (0~63) 轉為座標
            r, c = divmod(state, 8)
            
            # 2. 查表決定動作
            action = policy_map[r][c]
            
            # 3. 執行
            new_state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            state = new_state

        step_history.append(steps)

        if reward == 1.0:
            success_count += 1
        else:
            # --- 失敗原因統計 (已調整邏輯) ---
            
            # 1. 超過 100 步失敗 (步數 >= 100 的所有失敗，無論是超時或剛好在 100 步掉洞)
            if steps >= 100:
                fail_over_100_count += 1
                fail_over_100_steps_sum += steps
            
            # 2. 掉進洞裡 (僅統計步數 < 100 時就 terminated 的案例)
            # 由於 reward != 1.0 且 terminated = True，代表掉洞。
            # 這裡使用 elif 是為了排除步驟 1 中 steps >= 100 的情況。
            elif terminated:
                hole_fall_count += 1

    env.close()

    # --- 計算統計結果 ---
    success_rate = (success_count / episodes) * 100
    avg_steps = np.mean(step_history) if step_history else 0
    avg_fail_over_100_steps = (fail_over_100_steps_sum / fail_over_100_count) if fail_over_100_count > 0 else 0
    
    print(f"✅ 測試結束")
    print(f"總場數: {episodes}")
    print(f"🔥 成功率: {success_rate:.2f}% (成功次數: {success_count})")
    print(f"💀 掉進洞裡次數 (步數 < 100): {hole_fall_count} ({(hole_fall_count/episodes)*100:.2f}%)")
    print(f"⏳ 超過 100 步失敗次數 (步數 >= 100): {fail_over_100_count} ({(fail_over_100_count/episodes)*100:.2f}%)")
    print(f"平均每局步數: {avg_steps:.2f}")

if __name__ == "__main__":
    run_policy_test(100000)