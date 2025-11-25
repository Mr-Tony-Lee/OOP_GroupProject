# Frozen Lake Q-Learning 演算法詳細說明

## 檔案概述
這是一個使用 Q-Learning 強化學習演算法來訓練代理（agent）玩 Frozen Lake 遊戲的程式。

---

## 導入的套件

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
```

- **gymnasium**: OpenAI 的 Gym 環境庫，提供各種強化學習環境
- **numpy**: 數值計算庫，用於處理陣列和矩陣運算
- **matplotlib.pyplot**: 繪圖庫，用於視覺化訓練結果
- **pickle**: Python 序列化庫，用於保存和載入 Q-table

---

## 函數 1: `print_success_rate(rewards_per_episode)`

### 功能
計算並顯示代理的成功率。

### 參數
- `rewards_per_episode`: 一個陣列，記錄每個回合是否成功（1=成功，0=失敗）

### 程式碼逐行說明

```python
def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    # 計算總回合數
    
    success_count = np.sum(rewards_per_episode)
    # 計算成功次數（因為成功=1，失敗=0，所以總和=成功次數）
    
    success_rate = (success_count / total_episodes) * 100
    # 計算成功率百分比
    
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    # 格式化輸出成功率
    
    return success_rate
    # 返回成功率數值
```

---

## 函數 2: `run(episodes, is_training=True, render=False)`

### 功能
主要的訓練或測試函數。

### 參數
- `episodes`: **訓練/測試的回合數**
  - 訓練時通常設定較大值（如 15000）
  - 測試時設定較小值（如 10）

- `is_training`: **訓練模式開關**（預設 True）
  - `True`: 訓練模式，會學習並更新 Q-table
  - `False`: 測試模式，使用已訓練的 Q-table

- `render`: **視覺化顯示開關**（預設 False）
  - `True`: 顯示遊戲畫面
  - `False`: 不顯示（訓練時建議關閉以加快速度）

---

## 主程式碼逐行說明

### 1. 環境初始化

```python
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
```

**建立 Frozen Lake 遊戲環境**
- `'FrozenLake-v1'`: 環境名稱
- `map_name="8x8"`: 使用 8x8 的地圖（64 個格子）
- `is_slippery=True`: 地面是滑的（移動時可能滑到其他方向）
- `render_mode='human' if render else None`: 根據 render 參數決定是否顯示畫面

---

### 2. Q-table 初始化或載入

```python
if(is_training):
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
else:
    f = open('frozen_lake8x8.pkl', 'rb')
    q = pickle.load(f)
    f.close()
```

**Q-table（Q 表）**是強化學習的核心
- **訓練模式**: 創建一個 64×4 的零矩陣
  - 64 行：代表 64 個狀態（8×8 地圖的每個格子）
  - 4 列：代表 4 個動作（左、下、右、上）
  - Q[state][action] 表示在某狀態下執行某動作的預期累積獎勵

- **測試模式**: 從檔案載入已訓練好的 Q-table

---

### 3. 超參數設定

```python
learning_rate_a = 0.9 # alpha or learning rate
```
**學習率 (α, alpha)**
- 範圍：0 到 1
- 控制新資訊的接受程度
- 0.9 表示快速學習新經驗
- 接近 1：快速更新，可能不穩定
- 接近 0：慢速更新，學習緩慢

```python
discount_factor_g = 0.9 # gamma or discount rate
```
**折扣因子 (γ, gamma)**
- 範圍：0 到 1
- 控制對未來獎勵的重視程度
- 0.9 表示相當重視未來獎勵
- 接近 1：重視長期獎勵
- 接近 0：只重視立即獎勵

```python
epsilon = 1         # 1 = 100% random actions
```
**探索率 (ε, epsilon)**
- 範圍：0 到 1
- 控制探索（隨機動作）vs 利用（選擇最佳動作）的平衡
- 1 表示一開始 100% 隨機探索
- 隨著訓練會逐漸衰減

```python
epsilon_decay_rate = 0.00005       # epsilon decay rate
```
**探索率衰減速度**
- 每個回合後，epsilon 會減少這個數值
- 0.00005 表示需要約 20000 回合才會完全衰減到 0

```python
rng = np.random.default_rng()   # random number generator
```
**隨機數生成器**
- 用於生成隨機動作（探索用）

---

### 4. 獎勵記錄初始化

```python
rewards_per_episode = np.zeros(episodes)
```
**建立陣列記錄每個回合的獎勵**
- 長度為回合數
- 成功=1，失敗=0

---

### 5. 主訓練迴圈

```python
for i in range(episodes):
    state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
```
**每個回合開始**
- `env.reset()`: 重置環境，回到起始位置
- `[0]`: 取得初始狀態編號（0-63）
- 0 = 左上角起點，63 = 右下角終點

```python
    terminated = False      # True when fall in hole or reached goal
    truncated = False       # True when actions > 200
```
**回合結束條件**
- `terminated`: 掉入洞或到達目標
- `truncated`: 步數超過上限（200 步）

---

### 6. 單回合內的步驟迴圈

```python
    while(not terminated and not truncated):
```
**持續執行直到回合結束**

```python
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
        else:
            action = np.argmax(q[state,:])
```
**選擇動作（ε-greedy 策略）**
- **訓練模式 + 隨機數 < epsilon**: 隨機探索
  - `env.action_space.sample()`: 隨機選擇動作
  - 動作編碼：0=左, 1=下, 2=右, 3=上
  
- **否則**: 利用已學習的知識
  - `np.argmax(q[state,:])`: 選擇 Q 值最大的動作
  - 從當前狀態的所有動作中選最佳的

```python
        new_state,reward,terminated,truncated,_ = env.step(action)
```
**執行動作並觀察結果**
- `new_state`: 執行動作後的新狀態
- `reward`: 獲得的獎勵（到達目標=1，其他=0）
- `terminated`: 是否結束（掉洞或達標）
- `truncated`: 是否超時
- `_`: 其他資訊（這裡不使用）

---

### 7. Q-Learning 更新公式

```python
        if is_training:
            q[state,action] = q[state,action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
            )
```

**Q-Learning 核心演算法**

這是強化學習的精髓！讓我們拆解這個公式：

**完整公式**：
```
Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
```

**各部分說明**：

1. `q[state,action]`: 當前 Q 值 Q(s,a)

2. `learning_rate_a` (α): 學習率
   - 控制更新幅度

3. **TD Error（時間差分誤差）**:
   ```
   [reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]]
   ```
   
   - `reward`: 立即獎勵 R
   
   - `discount_factor_g * np.max(q[new_state,:])`: 未來最大預期獎勵
     - `np.max(q[new_state,:])`: 新狀態下所有動作的最大 Q 值
     - 乘以 γ 進行折扣
   
   - `reward + discount_factor_g * np.max(q[new_state,:])`: **目標 Q 值**
     - 實際獲得的獎勵 + 未來預期獎勵
   
   - 減去 `q[state,action]`: 與當前估計的差距
   
   - 這個差距就是 **預測誤差**

4. **更新過程**：
   - 用學習率縮放這個誤差
   - 加到原本的 Q 值上
   - 逐漸使 Q 值接近真實的期望累積獎勵

**白話解釋**：
- 我們根據實際經驗（獲得的獎勵和到達的新狀態）
- 來修正我們對「在某狀態執行某動作有多好」的估計
- 每次只修正一小步（由學習率控制）
- 逐漸學會最佳策略

```python
        state = new_state
```
**更新當前狀態**
- 移動到新狀態，繼續下一步

---

### 8. Epsilon 衰減

```python
    epsilon = max(epsilon - epsilon_decay_rate, 0)
```
**逐漸減少探索率**
- 每個回合後 epsilon 減少 `epsilon_decay_rate`
- `max(..., 0)` 確保不會低於 0
- 隨著訓練進行，從探索轉為利用已學知識

```python
    if(epsilon==0):
        learning_rate_a = 0.0001
```
**當完全停止探索時**
- 大幅降低學習率到 0.0001
- 因為此時主要在微調已學到的策略

---

### 9. 記錄獎勵

```python
    if reward == 1:
        rewards_per_episode[i] = 1
```
**記錄這個回合是否成功**
- 只有到達目標才會獲得 reward = 1
- 記錄在陣列中供後續分析

---

### 10. 關閉環境

```python
env.close()
```
**釋放環境資源**
- 關閉遊戲視窗和相關資源

---

### 11. 計算移動平均獎勵

```python
sum_rewards = np.zeros(episodes)
for t in range(episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
```
**計算滑動視窗總和**
- 對每個時間點 t
- 計算過去 100 個回合（或從開始到 t）的成功次數
- 用於平滑化趨勢，觀察學習進度

**範例**：
- t=0: 只有第 0 回合
- t=50: 第 0-50 回合的總和
- t=150: 第 50-150 回合的總和（最近 100 個）

---

### 12. 繪製學習曲線

```python
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')
```
**視覺化訓練進度**
- 繪製移動平均成功次數
- 保存為圖片檔案
- Y 軸：最近 100 回合的成功次數
- X 軸：回合編號
- 如果學習良好，曲線應該上升

---

### 13. 輸出成功率（測試模式）

```python
if is_training == False:
    print(print_success_rate(rewards_per_episode))
```
**測試模式時顯示成功率**
- 只在測試時執行
- 顯示模型表現如何

---

### 14. 儲存 Q-table（訓練模式）

```python
if is_training:
    f = open("frozen_lake8x8.pkl","wb")
    pickle.dump(q, f)
    f.close()
```
**保存訓練好的 Q-table**
- 只在訓練模式執行
- 使用 pickle 序列化 Q-table
- 保存為 `frozen_lake8x8.pkl`
- 之後可以載入使用，不需重新訓練

---

## 主程式執行區塊

```python
if __name__ == '__main__':
    # run(15000, is_training=True, render=False)

    run(10, is_training=False, render=True)
```

**當直接執行此檔案時**

### 選項 1：訓練（已註解）
```python
run(15000, is_training=True, render=False)
```
- 執行 15000 回合
- 訓練模式
- 不顯示畫面（加快訓練速度）
- 訓練完成後會保存 Q-table

### 選項 2：測試（目前啟用）
```python
run(10, is_training=False, render=True)
```
- 執行 10 回合
- 測試模式（使用已訓練的模型）
- 顯示遊戲畫面
- 觀察訓練好的代理如何玩遊戲

---

## Frozen Lake 遊戲說明

### 遊戲目標
- 從起點（左上角）走到目標（右下角）
- 避免掉入冰洞

### 地圖符號
- **S** (Start): 起點
- **F** (Frozen): 安全的冰面
- **H** (Hole): 冰洞（掉入則失敗）
- **G** (Goal): 目標

### 8x8 地圖範例
```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F F F H F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

### 動作空間
- 0: 向左移動
- 1: 向下移動
- 2: 向右移動
- 3: 向上移動

### 狀態空間
- 64 個狀態（0-63）
- 對應 8×8 地圖的每個格子
- 編號從左到右、從上到下

### 獎勵系統
- 到達目標：+1
- 其他情況：0

### 滑動特性 (`is_slippery=True`)
- 執行動作時有 1/3 機率滑到垂直方向
- 例如：想向右走，可能會向上或向下滑
- 增加遊戲難度，更接近真實冰面

---

## Q-Learning 演算法總結

### 核心概念
1. **Q-table**: 記錄每個（狀態，動作）配對的價值
2. **探索 vs 利用**: ε-greedy 策略平衡探索新動作和使用已知最佳動作
3. **時間差分學習**: 根據實際經驗更新 Q 值估計
4. **Bellman 方程**: Q-Learning 更新公式的理論基礎

### 學習過程
1. 初始：隨機探索（epsilon=1）
2. 中期：逐漸減少探索，增加利用
3. 後期：主要使用學到的策略（epsilon≈0）

### 收斂條件
- 當 Q 值不再顯著變化
- 或達到預定訓練回合數
- 成功率趨於穩定

---

## 使用建議

### 訓練階段
```python
run(15000, is_training=True, render=False)
```
- 需要數分鐘到數十分鐘
- 建議不開啟 render 以加快訓練
- 訓練完成會自動保存模型

### 測試階段
```python
run(10, is_training=False, render=True)
```
- 確保先完成訓練
- 可以看到代理的實際表現
- 調整回合數觀察穩定性

### 調參建議
- **提高學習率**: 加快學習，但可能不穩定
- **增加折扣因子**: 更重視長期獎勵
- **調整 epsilon 衰減率**: 控制探索/利用的時機

---

## 輸出檔案

### frozen_lake8x8.pkl
- 訓練好的 Q-table
- 可重複使用，不需重新訓練
- 二進制格式

### frozen_lake8x8.png
- 學習曲線圖
- 顯示訓練進度
- Y 軸：最近 100 回合的成功次數
- X 軸：總回合數

---

## 常見問題

### Q1: 為什麼成功率不高？
- Frozen Lake 本身很難（特別是 8x8 + slippery）
- 可能需要更多訓練回合
- 可以嘗試調整超參數

### Q2: 如何改善訓練效果？
- 增加訓練回合數
- 調整學習率和折扣因子
- 減慢 epsilon 衰減速度

### Q3: 測試時顯示檔案不存在？
- 確保先執行訓練模式
- 檢查 `frozen_lake8x8.pkl` 是否存在

### Q4: 如何使用 4x4 地圖？
- 修改 `map_name="4x4"`
- 調整檔案名稱避免覆蓋 8x8 的模型

---

## 延伸學習

### 可能的改進
1. **Double Q-Learning**: 減少過度估計
2. **Experience Replay**: 重複利用經驗
3. **Deep Q-Network (DQN)**: 用神經網路取代 Q-table
4. **Policy Gradient**: 直接學習策略而非價值

### 相關演算法
- SARSA（另一種時間差分學習）
- Actor-Critic
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)

---

## 參考資源

- [Gymnasium 官方文件](https://gymnasium.farama.org/)
- [Q-Learning 演算法介紹](https://en.wikipedia.org/wiki/Q-learning)
- [強化學習入門書籍推薦](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

---

**最後更新**: 2025-11-19
