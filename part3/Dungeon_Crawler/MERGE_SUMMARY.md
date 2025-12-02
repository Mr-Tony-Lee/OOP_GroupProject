# Merge Conflict 解決總結

## 概述
成功解決 CNN 分支和 DQN 分支之間的 merge conflict，並實現了完全相容的架構，支援三種 Agent：Q-Learning、DQN 和 CNN。

---

## 1. dungeon_env.py - 環境架構重構

### 設計理念
使用 **抽象基類 + 子類** 模式（策略模式），讓基類定義通用邏輯，子類實作特定的觀察格式。

### 類別結構

#### 基類：`DungeonCrawlerEnv` (ABC)
- **核心方法**：`reset()`, `step()`, `render()`, `close()`
- **抽象方法**：`_define_observation_space()`, `_get_obs()`
- **共享邏輯**：所有 reward 計算、遊戲狀態轉移、事件追蹤

#### 子類 1：`CNNDungeonEnv`
```python
# 觀察空間：字典格式
{
    "image": (7, H, W),      # 7 個通道，不含 has_key
    "scalars": (1,)          # has_key 信息
}

# 通道定義：
# 0: Player
# 1: Wall
# 2: Key
# 3: Door
# 4: Treasure
# 5: Trap
# 6: Monster
```
**優勢**：CNN 可以分別處理空間和標量信息，更靈活

#### 子類 2：`DQNDungeonEnv`
```python
# 觀察空間：Box 格式
(8, H, W)  # 8 個通道，統一表示

# 通道定義：
# 0: Wall
# 1: Player
# 2: Enemy
# 3: Key
# 4: Door
# 5: Treasure
# 6: Trap
# 7: HasKey (全局廣播)
```
**優勢**：DQN 用統一的 CNN 架構處理所有信息

### 環境註冊
```python
'dungeon-crawler-cnn-v0'  -> CNNDungeonEnv
'dungeon-crawler-dqn-v0'  -> DQNDungeonEnv
'dungeon-crawler-v0'      -> DQNDungeonEnv (預設，後向兼容)
```

### 保留的功能
✅ 所有 reward 計算邏輯（撞牆、找鑰匙、開門、距離獎勵等）
✅ 事件追蹤和日誌記錄
✅ 遊戲狀態的完整追蹤
✅ Pygame 渲染支援

---

## 2. train.py - 訓練框架統一

### 新增功能

#### 狀態預處理函數
```python
preprocess_state(state, agent_type)
# QLearning: (8, H, W) → (row, col, has_key)
# DQN:       (8, H, W) → 原樣返回
# CNN:       Dict      → 原樣返回
```

#### Agent 工廠函數
```python
get_agent(agent_type, env)
# 根據 agent_type 創建相應的 agent
# 支援："QLearning", "DQN", "CNN"
```

#### 環境選擇函數
```python
get_env_id(agent_type)
# CNN → 'dungeon-crawler-cnn-v0'
# 其他 → 'dungeon-crawler-dqn-v0'
```

### 訓練邏輯整合
- ✅ 支援三種 agent 類型
- ✅ 自動選擇合適的環境
- ✅ 統一的日誌和模型保存機制
- ✅ 動態配置目錄結構（`{AgentType}Agent/log/`, `{AgentType}Agent/result/`）
- ✅ 最佳模型和最終模型的保存

### 使用範例
```python
# 訓練 DQN
AGENT_TYPE = "DQN"
train(AGENT_TYPE)

# 測試 CNN
AGENT_TYPE = "CNN"
test(AGENT_TYPE)

# 訓練 Q-Learning
AGENT_TYPE = "QLearning"
train(AGENT_TYPE)
```

---

## 3. agent.py - Agent 統一框架

### 類別階層

```
Agent (ABC)
├── QLearningAgent
├── DQNAgent
└── CNNAgent
```

### 1. QLearningAgent
**特點**：表格型 Q-Learning，適合簡單狀態空間

**輸入**：`(row, col, has_key)` 三元組

**配置**：
```python
QLearningAgent(
    action_space,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.999
)
```

**儲存格式**：Pickle 文件 (`.pkl`)

### 2. DQNAgent
**特點**：標準 Deep Q-Network，適合連續狀態空間

**輸入**：`(8, H, W)` numpy array

**網絡架構**：
```
Input (8, H, W)
  ↓
Conv2d(8 → 32)  + ReLU
  ↓
Conv2d(32 → 64) + ReLU
  ↓
Conv2d(64 → 64) + ReLU
  ↓
Flatten → Linear(flatten_size → 512) → ReLU
  ↓
Linear(512 → n_actions) → Q-values
```

**特性**：
- Double DQN 邏輯（分離 policy 和 target 網絡）
- 經驗重放 (Replay Buffer)
- 定期更新目標網絡

**配置**：
```python
DQNAgent(
    state_shape=(8, 11, 12),
    action_space=env.action_space,
    learning_rate=0.0001,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    min_epsilon=0.01,
    batch_size=64,
    memory_size=50000
)
```

**儲存格式**：PyTorch `.pth` 文件

### 3. CNNAgent
**特點**：混合架構，分別處理空間和標量信息

**輸入**：
```python
{
    "image": (7, H, W),
    "scalars": (1,)
}
```

**網絡架構**：
```
image (7, H, W)          scalars (1,)
    ↓                          ↓
Conv layers               （直接傳入）
    ↓
Flatten
    ↓
  Concat ← 整合兩種信息
    ↓
Linear(concat_size → 128) → ReLU
    ↓
Linear(128 → n_actions) → Q-values
```

**特性**：
- 兩個網絡：Policy Net 和 Target Net
- Huber Loss (SmoothL1Loss) 提高穩定性
- 經驗重放和定期網絡同步

**配置**：
```python
CNNAgent(
    observation_space=env.observation_space,  # Dict
    action_space=env.action_space,
    learning_rate=0.00025,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,
    min_epsilon=0.05
)
```

**儲存格式**：PyTorch `.pth` 文件

---

## 4. 相容性矩陣

| Agent Type | 環境 | 觀察格式 | 儲存格式 |
|----------|------|---------|---------|
| QLearning | dungeon-crawler-dqn-v0 | (row, col, has_key) | .pkl |
| DQN | dungeon-crawler-dqn-v0 | (8, H, W) | .pth |
| CNN | dungeon-crawler-cnn-v0 | Dict{image, scalars} | .pth |

---

## 5. 文件結構
```
Dungeon_Crawler/
├── dungeon_env.py       ✅ 重構 (基類 + 2 個子類)
├── agent.py             ✅ 整合 (3 個 Agent 類)
├── train.py             ✅ 統一 (支援 3 種訓練)
├── dungeon_game.py      (無需修改)
├── QLearningAgent/
│   ├── log/
│   └── result/
├── DQNAgent/
│   ├── log/
│   └── result/
└── CNNAgent/
    ├── log/
    └── result/
```

---

## 6. 訓練和測試流程

### 訓練 DQN
```python
AGENT_TYPE = "DQN"
train(AGENT_TYPE)
# 輸出：
# - DQNAgent/log/training_log.txt
# - DQNAgent/log/event_log.txt
# - DQNAgent/result/best_model.pth
# - DQNAgent/result/final_model.pth
# - DQNAgent/result/training_curve.png
```

### 測試 DQN
```python
AGENT_TYPE = "DQN"
test(AGENT_TYPE)
# 自動載入 best_model.pth，若不存在則用 final_model.pth
```

### 訓練 CNN
```python
AGENT_TYPE = "CNN"
train(AGENT_TYPE)
# 同上，但環境為 dungeon-crawler-cnn-v0
```

---

## 7. Merge 的設計決策

### 為什麼保留三個 Agent？
- **Q-Learning**：簡單、快速、適合學習
- **DQN**：標準深度學習，通用性強
- **CNN**：混合架構，最接近原始需求

### 為什麼創建兩個環境子類？
- CNN 需要分離的圖像和標量輸入
- DQN 需要統一的多通道表示
- 基類共享所有複雜邏輯（reward 計算、事件追蹤）
- 避免重複代碼

### 為什麼使用 Abstract Base Class？
- 強制子類實作必要方法
- 提高代碼可維護性
- 易於擴展新的環境類型或 agent 類型

---

## 8. 驗證清單

- ✅ 所有 merge conflict 已解決（無 `<<<<<<<`, `=======`, `>>>>>>>`）
- ✅ 代碼通過語法檢查（無編譯錯誤）
- ✅ 三個 Agent 都支援 save/load
- ✅ 訓練和測試函數支援動態 agent 選擇
- ✅ 環境自動選擇合適的子類
- ✅ 日誌記錄機制完整
- ✅ 模型管理機制清晰（最佳 + 最終）

---

## 9. 後續可能的改進

1. **配置文件**：使用 YAML/JSON 管理超參數
2. **多環境並行訓練**：支援多個 agent 同時訓練
3. **超參數優化**：自動調整學習率、epsilon 衰減等
4. **模型評估指標**：詳細的性能分析和比較
5. **可視化工具**：實時訓練監控儀表板

---

## 10. 快速開始

```python
# train.py 中
if __name__ == "__main__":
    AGENT_TYPE = "DQN"  # 選擇：DQN, CNN, QLearning
    
    # 訓練
    train(AGENT_TYPE)
    
    # 測試
    # test(AGENT_TYPE)
```

祝訓練順利！🚀