# 快速使用指南

## 📋 環境信息

### 已安裝的環境
- `dungeon-crawler-cnn-v0` → CNNDungeonEnv（CNN 專用）
- `dungeon-crawler-dqn-v0` → DQNDungeonEnv（DQN 和 QLearning 通用）
- `dungeon-crawler-v0` → DQNDungeonEnv（預設環境，後向兼容）

### 支援的 Agent
1. **QLearningAgent** - 傳統 Q-Learning（表格型）
2. **DQNAgent** - Deep Q-Network（CNN 架構）
3. **CNNAgent** - CNN 混合架構（圖像 + 標量）

---

## 🚀 訓練

### 方式 1：訓練 DQN Agent
```python
# train.py 中修改
if __name__ == "__main__":
    AGENT_TYPE = "DQN"
    train(AGENT_TYPE)
    # test(AGENT_TYPE)
```
輸出目錄：`DQNAgent/log/`, `DQNAgent/result/`

### 方式 2：訓練 CNN Agent
```python
# train.py 中修改
if __name__ == "__main__":
    AGENT_TYPE = "CNN"
    train(AGENT_TYPE)
    # test(AGENT_TYPE)
```
輸出目錄：`CNNAgent/log/`, `CNNAgent/result/`

### 方式 3：訓練 Q-Learning Agent
```python
# train.py 中修改
if __name__ == "__main__":
    AGENT_TYPE = "QLearning"
    train(AGENT_TYPE)
    # test(AGENT_TYPE)
```
輸出目錄：`QLearningAgent/log/`, `QLearningAgent/result/`

---

## 🧪 測試

### 測試已訓練的模型
```python
# train.py 中修改
if __name__ == "__main__":
    AGENT_TYPE = "DQN"  # 選擇要測試的 agent
    # train(AGENT_TYPE)
    test(AGENT_TYPE)
```

會自動載入：
1. `{AgentType}Agent/result/best_model.pth`（或 `.pkl`）
2. 若找不到，則嘗試 `{AgentType}Agent/result/final_model.pth`

---

## 📊 訓練輸出

每個 agent 的訓練會生成：

```
{AgentType}Agent/
├── log/
│   ├── training_log.txt      # 訓練進度日誌
│   └── event_log.txt         # 遊戲事件記錄
└── result/
    ├── best_model.pth        # 最佳模型
    ├── final_model.pth       # 最終模型
    └── training_curve.png    # 訓練曲線圖
```

### training_log.txt 範例
```
Start Training with DQN...
Episode 50/2000, Avg Reward (Last 50): 25.34, Best: 45.20, Epsilon: 0.9950
Episode 100/2000, Avg Reward (Last 50): 30.12, Best: 55.30, Epsilon: 0.9900
...
```

### event_log.txt 範例
```
Episode, Event
1, found_key
1, opened_door
1, won
2, died
...
```

---

## 🔧 配置參數

### DQN Agent
```python
DQNAgent(
    state_shape=(8, 11, 12),
    action_space=env.action_space,
    learning_rate=0.0001,       # 學習率（較低以保持穩定）
    discount_factor=0.99,       # 折扣因子（越高越看重未來獎勵）
    epsilon=1.0,                # 初始探索率
    epsilon_decay=0.9995,       # 衰減速率
    min_epsilon=0.01,           # 最小探索率
    batch_size=64,              # 批量大小
    memory_size=50000           # 經驗重放緩衝區大小
)
```

### CNN Agent
```python
CNNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=0.00025,      # CNN 使用更低的學習率
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,        # CNN 衰減更慢（更多探索）
    min_epsilon=0.05
)
```

### Q-Learning Agent
```python
QLearningAgent(
    env.action_space,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.999
)
```

---

## 📈 訓練建議

### Q-Learning
- 最快收斂（數百回合）
- 適合快速原型設計
- 局限於離散、小狀態空間

### DQN
- 中等收斂速度（1000+ 回合）
- 通用性好，適合大多數問題
- 需要調整超參數以達到最佳性能

### CNN
- 最慢收斂（1500+ 回合）
- 最靈活，可處理複雜視覺信息
- 需要最多的調參

---

## 🔍 常見問題

### Q: 如何修改獎勵設計？
A: 編輯 `dungeon_env.py` 中 `DungeonCrawlerEnv.step()` 方法中的獎勵計算邏輯。

### Q: 如何自定義超參數？
A: 修改 `train.py` 中 `get_agent()` 函數中的超參數，或在 `train()` 函數中直接傳入。

### Q: 如何新增新的 Agent 類型？
A: 
1. 在 `agent.py` 中繼承 `Agent` 類
2. 實作 `get_action()`, `learn()`, `save()`, `load()` 方法
3. 在 `train.py` 中的 `get_agent()` 和 `preprocess_state()` 中添加邏輯

### Q: 如何新增新的環境？
A: 
1. 在 `dungeon_env.py` 中繼承 `DungeonCrawlerEnv`
2. 實作 `_define_observation_space()` 和 `_get_obs()` 方法
3. 在文件末尾註冊新環境

### Q: 訓練過程中 GPU 不被使用？
A: 檢查 PyTorch 是否正確安裝 CUDA 版本。使用：
```python
import torch
print(torch.cuda.is_available())  # 應該返回 True
print(torch.cuda.get_device_name(0))
```

---

## 📝 文件結構速查

```
Dungeon_Crawler/
├── dungeon_env.py          # 環境定義（基類 + 2 子類）
├── agent.py                # Agent 定義（3 個 Agent 類）
├── train.py                # 訓練和測試函數
├── dungeon_game.py         # 遊戲邏輯（無需修改）
├── MERGE_SUMMARY.md        # 合併詳細文檔
├── README.md               # 本文件
├── DQNAgent/
│   ├── log/
│   │   ├── training_log.txt
│   │   └── event_log.txt
│   └── result/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── training_curve.png
├── CNNAgent/
│   └── (同上)
└── QLearningAgent/
    └── (同上)
```

---

## 💡 開發提示

### 調試技巧
1. 在 `dungeon_env.py` 中設置 `render_mode='human'` 查看遊戲過程
2. 檢查 `event_log.txt` 確認遊戲事件是否正常觸發
3. 監控 `training_log.txt` 的獎勵趨勢

### 性能優化
1. 增加 `batch_size` 加速訓練（需要更多 GPU 內存）
2. 調整 `epsilon_decay` 控制探索-利用平衡
3. 增加 `memory_size` 改進樣本多樣性

### 模型評估
1. 比較 `best_reward` 和 `final_reward`
2. 查看 `training_curve.png` 確認學習趨勢
3. 執行多次測試確保穩定性

---

## 🎯 下一步

1. ✅ 執行訓練：`python train.py` (配置好 `AGENT_TYPE` 和訓練/測試開關)
2. ✅ 監控進度：查看 `{AgentType}Agent/log/training_log.txt`
3. ✅ 評估模型：檢查 `result/training_curve.png`
4. ✅ 測試模型：改為 `test(AGENT_TYPE)` 並執行
5. ✅ 比較性能：训練不同的 Agent 並比较結果

---

祝訓練順利！🚀

有問題請查看 `MERGE_SUMMARY.md` 獲取詳細技術文檔。