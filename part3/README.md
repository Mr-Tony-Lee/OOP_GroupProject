# Dungeon Crawler RL Project (Part 3)

這是一個結合 **物件導向程式設計 (OOP)** 與 **強化學習 (Reinforcement Learning)** 的期末專案。我們從零開始打造了一個符合 Gymnasium 標準的地牢探險環境，並實作了多種 RL Agent (Q-Learning, DQN, Double DQN) 來自動破解關卡。

## 🌟 專案特色 (Features)

1.  **物件導向架構 (OOP Architecture)**:
    *   使用 `GameObject` 作為基底類別，衍生出 `Character` (角色) 與 `Item` (物品)。
    *   利用 **多型 (Polymorphism)** 處理不同物件的互動邏輯 (如：撞牆、踩陷阱、撿鑰匙、開門)。
    *   利用 **繼承 (Inheritance)** 實作 `Player` 與 `Enemy`，共享移動邏輯但擁有不同的行為模式。

2.  **自定義 Gymnasium 環境 (Custom Environment)**:
    *   完全符合 Gymnasium API 標準 (`reset`, `step`, `render`)。
    *   支援多種觀察空間 (Observation Space)：
        *   **純量 (Scalar)**: 供 Q-Learning 使用 (座標, 狀態)。
        *   **多模態 (Multimodal)**: 供 DQN 使用 (圖像 + 純量)。
    *   設計了包含獎勵 (Reward) 與懲罰 (Penalty) 的機制來引導 Agent 學習。

3.  **多種 RL 演算法實作**:
    *   **Q-Learning**: 表格型強化學習，適合簡單狀態。
    *   **DQN (Deep Q-Network)**: 結合 CNN 與神經網路，處理圖像輸入。
    *   **Double DQN (DDQN)**: 改進版 DQN，減少價值高估問題，提升穩定性。

## 📂 檔案結構 (File Structure)

位於 `part3/` 資料夾下：

*   **`dungeon_game.py`**: 遊戲核心邏輯 (Pygame)。包含所有類別 (`GameObject`, `Player`, `Enemy` 等) 與遊戲迴圈。
*   **`dungeon_env.py`**: Gymnasium 環境封裝。將遊戲包裝成標準 RL 環境。
*   **`agent.py`**: 包含 `BaseDQNAgent`, `DQNAgent`, `DoubleDQNAgent`, `QLearningAgent` 的實作。
*   **`train.py`**: 統一的訓練與測試入口。支援命令列參數 (CLI) 來調整訓練設定。
*   **`human.py`**: 人類手動試玩腳本。

## 📦 安裝需求 (Requirements) && Dependencies

請確保安裝以下 Python 套件：

```
part3
├── gymnasium v1.2.2
│   ├── cloudpickle v3.1.2
│   ├── farama-notifications v0.0.4
│   ├── numpy v2.3.5
│   └── typing-extensions v4.15.0
├── matplotlib v3.10.7
│   ├── contourpy v1.3.3
│   │   └── numpy v2.3.5
│   ├── cycler v0.12.1
│   ├── fonttools v4.61.0
│   ├── kiwisolver v1.4.9
│   ├── numpy v2.3.5
│   ├── packaging v25.0
│   ├── pillow v12.0.0
│   ├── pyparsing v3.2.5
│   └── python-dateutil v2.9.0.post0
│       └── six v1.17.0
├── pygame v2.6.1
├── tensorflow v2.20.0
│   ├── absl-py v2.3.1
│   ├── astunparse v1.6.3
│   │   ├── six v1.17.0
│   │   └── wheel v0.45.1
│   ├── flatbuffers v25.9.23
│   ├── gast v0.7.0
│   ├── google-pasta v0.2.0
│   │   └── six v1.17.0
│   ├── grpcio v1.76.0
│   │   └── typing-extensions v4.15.0
│   ├── h5py v3.15.1
│   │   └── numpy v2.3.5
│   ├── keras v3.12.0
│   │   ├── absl-py v2.3.1
│   │   ├── h5py v3.15.1 (*)
│   │   ├── ml-dtypes v0.5.4
│   │   │   └── numpy v2.3.5
│   │   ├── namex v0.1.0
│   │   ├── numpy v2.3.5
│   │   ├── optree v0.18.0
│   │   │   └── typing-extensions v4.15.0
│   │   ├── packaging v25.0
│   │   └── rich v14.2.0
│   │       ├── markdown-it-py v4.0.0
│   │       │   └── mdurl v0.1.2
│   │       └── pygments v2.19.2
│   ├── libclang v18.1.1
│   ├── ml-dtypes v0.5.4 (*)
│   ├── numpy v2.3.5
│   ├── opt-einsum v3.4.0
│   ├── packaging v25.0
│   ├── protobuf v6.33.2
│   ├── requests v2.32.5
│   │   ├── certifi v2025.11.12
│   │   ├── charset-normalizer v3.4.4
│   │   ├── idna v3.11
│   │   └── urllib3 v2.6.1
│   ├── setuptools v80.9.0
│   ├── six v1.17.0
│   ├── tensorboard v2.20.0
│   │   ├── absl-py v2.3.1
│   │   ├── grpcio v1.76.0 (*)
│   │   ├── markdown v3.10
│   │   ├── numpy v2.3.5
│   │   ├── packaging v25.0
│   │   ├── pillow v12.0.0
│   │   ├── protobuf v6.33.2
│   │   ├── setuptools v80.9.0
│   │   ├── tensorboard-data-server v0.7.2
│   │   └── werkzeug v3.1.4
│   │       └── markupsafe v3.0.3
│   ├── termcolor v3.2.0
│   ├── typing-extensions v4.15.0
│   └── wrapt v2.0.1
└── torch v2.9.1
    ├── filelock v3.20.0
    ├── fsspec v2025.12.0
    ├── jinja2 v3.1.6
    │   └── markupsafe v3.0.3
    ├── networkx v3.6.1
    ├── setuptools v80.9.0
    ├── sympy v1.14.0
    │   └── mpmath v1.3.0
    └── typing-extensions v4.15.0
```

```bash
pip install -r requirements.txt
```

## 🚀 如何執行 (How to Run)

所有操作都可以透過 `train.py` 或特定腳本執行。

### 1. 手動試玩 (Human Play)
親自挑戰這個 11x12 的複雜迷宮！
*   **操作**：方向鍵移動，`R` 重置，`ESC` 離開。
*   **目標**：避開怪物與陷阱 -> 拿到鑰匙 (K) -> 打開門 (D) -> 取得寶藏 (T)。

```bash
# 方法一：使用 human.py
python part3/human.py

# 方法二：使用 train.py
python part3/train.py --mode human
```

### 2. 訓練 AI (Train Agent)
讓 AI 從零開始學習。程式會顯示訓練日誌並定期儲存模型。

**基本指令**:
```bash
# 訓練 Standard DQN
python part3/train.py --mode train --agent DQN

# 訓練 Double DQN (推薦)
python part3/train.py --mode train --agent DDQN

# 訓練 Q-Learning
python part3/train.py --mode train --agent QLearning
```

**進階參數**:
你可以透過參數調整超參數 (Hyperparameters)：
```bash
usage: train.py [-h] [--agent {QLearning,DDQN,DQN}] [--episodes EPISODES] [--mode {train,test,human}] [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--epsilon EPSILON]
                [--epsilon_decay EPSILON_DECAY] [--min_epsilon MIN_EPSILON] [--batch BATCH] [--memory MEMORY] [--target_update_freq TARGET_UPDATE_FREQ]
                options:
  -h, --help                                -> show this help message and exit
  --agent {QLearning,DDQN,DQN}              -> Type of agent to use
  --episodes EPISODES                       -> Number of episodes to train
  --mode {train,test,human}                 -> Mode to run the agent
  --learning_rate LEARNING_RATE             -> Learning rate
  --gamma GAMMA                             -> Discount factor
  --epsilon EPSILON                         -> Initial epsilon
  --epsilon_decay EPSILON_DECAY             -> Epsilon decay rate
  --min_epsilon MIN_EPSILON                 -> Minimum epsilon
  --batch BATCH                             -> Batch size
  --memory MEMORY                           -> Memory size
  --target_update_freq TARGET_UPDATE_FREQ   -> Target update frequency
```

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--agent` | DQN | 選擇 Agent 類型 (`DQN`, `DDQN`, `QLearning`) |
| `--episodes` | 2000 | 訓練總回合數 |
| `--mode` | train | 選擇模式 (`train`, `test`, `human`) |
| `--learning_rate` | 0.00025 | 學習率 |
| `--gamma` | 0.99 | 折扣因子 (Discount Factor) |
| `--epsilon` | 0.1 | 初始 epsilon |
| `--epsilon_decay` | 0.995 | Epsilon 退火率 |
| `--min_epsilon` | 0.01 | 最小 epsilon |
| `--batch` | 512 | 批量大小 (Batch Size) |
| `--memory` | 10000 | 變換記憶體大小 |
| `--target_update_freq` | 1000 | Target Network 更新頻率 |

### 3. 測試模型 (Test Agent)
載入訓練好的模型 (`final_model`) 並觀看 AI 實際遊玩。

```bash
python part3/train.py --mode test --agent DDQN
```
*   注意：測試模式會讀取 `Result/{agent}/result/final_model.pth`，請先確保訓練完成。

## 🎮 遊戲規則 (Game Rules)

*   **地圖**: 11x12 的大型迷宮。
*   **目標**: 拿到 **寶藏 (Treasure, 📦)**。
*   **障礙**:
    *   **牆壁 (Wall)**: 無法穿越。
    *   **門 (Door)**: 鎖住的，需要鑰匙才能通過。
    *   **陷阱 (Trap)**: 踩到會扣血 (-1 HP) 並扣分。
    *   **怪物 (Enemy)**: 3 隻巡邏怪，碰到會大量扣血 (-1 HP) 並重罰。
*   **道具**:
    *   **鑰匙 (Key)**: 藏在迷宮某處，可開啟門。

## 📈 學習成果 (Results)
經過訓練後，Double DQN Agent 能夠展現出穩定的策略：
1.  有效避開移動中的怪物。
2.  繞過陷阱區域。
3.  準確地撿起鑰匙並開啟大門。
