# OOP Group Project - Group 32

## 🌟 專案總覽 (Project Overview)

本專案分為兩個部分：

### 1. Frozen Lake RL Project (Part 2)

#### 📂 檔案結構 (File Structure)

位於 `part2/` 資料夾下：

*   **`Agent.py`**: 核心邏輯。包含 `QLearningAgent` 類別，負責 Q-Table 的更新與動作選擇。
*   **`CheatingEnv.py`**: 自定義環境。包含 `LessSlipperyFrozenLakeEnv`，提供更友善的學習環境。
*   **`main.py`**: 主程式。負責解析參數、執行訓練迴圈、評估並儲存結果。
*   **`Result/`**: 存放訓練結果圖表與數據。

#### 📦 安裝需求 (Requirements) && dependencies

請確保安裝以下 Python 套件：
```
part2
├── gymnasium v1.2.2
│   ├── cloudpickle v3.1.2
│   ├── farama-notifications v0.0.4
│   ├── numpy v2.3.5
│   └── typing-extensions v4.15.0
└── matplotlib v3.10.7
    ├── contourpy v1.3.3
    │   └── numpy v2.3.5
    ├── cycler v0.12.1
    ├── fonttools v4.61.0
    ├── kiwisolver v1.4.9
    ├── numpy v2.3.5
    ├── packaging v25.0
    ├── pillow v12.0.0
    ├── pyparsing v3.2.5
    └── python-dateutil v2.9.0.post0
        └── six v1.17.0
```
```bash
pip install -r requirements.txt
```

#### 🚀 如何執行 (How to Run)

所有操作都可以透過 `main.py` 執行，支援豐富的命令列參數 (CLI)。

##### 1. 訓練與評估 (Train & Evaluate)

**基本指令 (預設設定)**:
```bash
python part2/main.py
```
*   預設使用 `8x8` 地圖，開啟滑動模式，並使用優化過的環境 (`cheating=True`)。

**自定義參數範例**:
在 4x4 地圖上訓練，關閉作弊模式 (使用原始 Gymnasium 環境)：
```bash
python part2/main.py \
    --map 4x4 \
    --runs 10 \
    --cheating False \
    --train_episodes 15000 \
    --eval_episodes 1000 \
    --is_slippery True \
    --render_mode ansi
```

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--map` | 8x8 | 地圖大小 (`4x4` 或 `8x8`) |
| `--runs` | 10 | 實驗重複次數 (取平均用) |
| `--cheating` | True | 是否使用自定義的 LessSlippery 環境 |
| `--train_episodes` | 15000 | 訓練回合數 |
| `--eval_episodes` | 1000 | 評估回合數 |
| `--is_slippery` | True | 是否開啟滑動模式 |
| `--render_mode` | ansi | 渲染模式 (`ansi` 文字模式 或 `human` 視窗模式) |

##### 2. 查看幫助 (Help)
查看所有可用的參數說明：
```bash
python part2/main.py --help
```
--- 

### 2. Dungeon Crawler RL Project (Part 3)

> 📌 **UML 類別圖**：[part3/uml/class_diagram.png](part3/uml/class_diagram.png)

#### 📂 檔案結構 (File Structure)

位於 `part3/` 資料夾下：

*   **`dungeon_game.py`**: 遊戲核心邏輯 (Pygame)。包含所有類別 (`GameObject`, `Player`, `Enemy` 等) 與遊戲迴圈。
*   **`dungeon_env.py`**: Gymnasium 環境封裝。將遊戲包裝成標準 RL 環境。
*   **`agent.py`**: 包含 `BaseDQNAgent`, `DQNAgent`, `DoubleDQNAgent`, `QLearningAgent` 的實作。
*   **`train.py`**: 統一的訓練與測試入口。支援命令列參數 (CLI) 來調整訓練設定。
*   **`human.py`**: 人類手動試玩腳本。

#### 📦 安裝需求 (Requirements) && Dependencies

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

#### 🚀 如何執行 (How to Run)

所有操作都可以透過 `train.py` 或特定腳本執行。

##### 1. 手動試玩 (Human Play)
親自挑戰這個 11x12 的複雜迷宮！
*   **操作**：方向鍵移動，`R` 重置，`ESC` 離開。
*   **目標**：避開怪物與陷阱 -> 拿到鑰匙 (K) -> 打開門 (D) -> 取得寶藏 (T)。

```bash
# 方法一：使用 human.py
python part3/human.py

# 方法二：使用 train.py
python part3/train.py --mode human
```

##### 2. 訓練 AI (Train Agent)
讓 AI 從零開始學習。程式會顯示訓練日誌並定期儲存模型。

**基本指令**:
```bash
# 訓練 Standard DQN
python part3/train.py --mode train --agent DQN

# 訓練 Double DQN (推薦)
python part3/train.py --mode train --agent DDQN

# 訓練 Q-Learning
python part3/train.py --mode train --agent QLearning

# 訓練 PPO（範例）
python part3/train.py --mode train --agent PPO --batch 64
```

**進階參數**:
你可以透過參數調整超參數 (Hyperparameters)：
```bash
usage: train.py [-h] [--agent {QLearning,DDQN,DQN,PPO}] [--episodes EPISODES] [--mode {train,test,eval,human}] [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--epsilon EPSILON]
                                [--epsilon_decay EPSILON_DECAY] [--min_epsilon MIN_EPSILON] [--batch BATCH] [--memory MEMORY] [--target_update_freq TARGET_UPDATE_FREQ] [--gae_lambda GAE_LAMBDA]
                                [--policy_clip POLICY_CLIP] [--n_epochs N_EPOCHS]
                options:
  -h, --help                                -> show this help message and exit
  --agent {QLearning,DDQN,DQN,PPO}          -> Type of agent to use
  --episodes EPISODES                       -> Number of episodes to train
  --mode {train,test,eval,human}            -> Mode to run the agent 
  --learning_rate LEARNING_RATE             -> Learning rate
  --gamma GAMMA                             -> Discount factor
  --epsilon EPSILON                         -> Initial epsilon
  --epsilon_decay EPSILON_DECAY             -> Epsilon decay rate
  --min_epsilon MIN_EPSILON                 -> Minimum epsilon
  --batch BATCH                             -> Batch size
  --memory MEMORY                           -> Memory size
  --target_update_freq TARGET_UPDATE_FREQ   -> Target update frequency
  --gae_lambda GAE_LAMBDA                   -> (PPO) GAE lambda
  --policy_clip POLICY_CLIP                 -> (PPO) Clip epsilon
  --n_epochs N_EPOCHS                       -> (PPO) Epochs per update
```

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--agent` | DQN | 選擇 Agent 類型 (`DQN`, `DDQN`, `QLearning`, `PPO`) |
| `--episodes` | 2000 | 訓練總回合數 |
| `--mode` | train | 選擇模式 (`train`, `test`, `eval`, `human`) |
| `--learning_rate` | 0.00025 | 學習率 |
| `--gamma` | 0.99 | 折扣因子 (Discount Factor) |
| `--epsilon` | 1.0 | 初始 epsilon (DQN 系列使用) |
| `--epsilon_decay` | 0.998 | Epsilon 退火率 |
| `--min_epsilon` | 0.05 | 最小 epsilon |
| `--batch` | 512 | 批量大小 (Batch Size) |
| `--memory` | 50000 | 變換記憶體大小 |
| `--target_update_freq` | 1000 | Target Network 更新頻率；PPO 則是 update_interval |
| `--gae_lambda` | 0.95 | (PPO) GAE lambda |
| `--policy_clip` | 0.2 | (PPO) Clip epsilon |
| `--n_epochs` | 10 | (PPO) 每次更新的 epoch 數 |

##### 3. 測試模型 (Test Agent)
載入訓練好的模型 (`final_model`) 並觀看 AI 實際遊玩。

```bash
python part3/train.py --mode test --agent DDQN
```
*   注意：測試模式會讀取 `Result/{agent}/result/final_model.pth`，請先確保訓練完成。

##### 4. 評估模型 (Eval Mode)
使用確定性策略（貪婪動作）跑多個 episodes，計算平均與標準差，適合比較不同超參或演算法的最終表現。

```bash
# 評估 PPO，跑 50 回合（預設）
python part3/train.py --mode eval --agent PPO --episodes 50

# 也可套用到其他 agent（會使用各自的貪婪策略）
python part3/train.py --mode eval --agent DQN --episodes 50
```

*   輸出格式：`Mean Reward: <平均> ± <標準差>`。
*   PPO 評估使用 `deterministic=True`，避免訓練時的探索噪音。

---

## 👥 工作分工 (Contribution List)

| 組員 | 學號 | Github username | 負責項目 |
|------|------|--------|---------------------|
| 徐子皓 | B124040036 | HaoHao041003 | part3 CNN agent ... |
| 陳彥維 | B123040039 | Shuaige0709 | part3 PPO agent, eval mode, AgentSpec class, UML  |
| 李承諺 | B123040032 | Mr-Tony-Lee | part2, part3 ... |