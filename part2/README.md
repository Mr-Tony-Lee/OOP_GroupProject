# Frozen Lake RL Project (Part 2)

這是一個基於 **Gymnasium Frozen Lake** 環境的強化學習專案。我們實作了 Q-Learning Agent，並針對滑動 (Slippery) 環境進行了優化，甚至設計了一個更容易學習的自定義環境 (`CheatingEnv`)。

## 🌟 專案總覽 (Overview)
1. **Q-Learning 實作**:
    *   實作了標準的 Q-Learning 演算法。

2. **Q-Learning 優化**:
    *   使用環境的轉移機率 (Transition Probability, `env.unwrapped.P`) 來計算期望目標值，加速收斂 (Model-Based approach)。

3. **自定義環境 (Cheating Environment)**:
    *   **LessSlipperyFrozenLakeEnv**: 修改自 Gymnasium 的原始環境，降低了冰面滑動的機率，讓 Agent 更容易學習到有效策略。

4. **完整的實驗數據**:
    *   在 4x4 與 8x8 的滑動地圖上進行測試。
    *   4x4 Slippery map 成功率穩定在 **74%** 左右。
    *   8x8 Slippery map 成功率穩定在 **63%** 左右。

## 📂 檔案結構 (File Structure)

位於 `part2/` 資料夾下：

*   **`Agent.py`**: 核心邏輯。包含 `QLearningAgent` 類別，負責 Q-Table 的更新與動作選擇。
*   **`CheatingEnv.py`**: 自定義環境。包含 `LessSlipperyFrozenLakeEnv`，提供更友善的學習環境。
*   **`main.py`**: 主程式。負責解析參數、執行訓練迴圈、評估並儲存結果。
*   **`Result/`**: 存放訓練結果圖表與數據。

## 📦 安裝需求 (Requirements) && dependencies

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

## 🚀 如何執行 (How to Run)

所有操作都可以透過 `main.py` 執行，支援豐富的命令列參數 (CLI)。

### 1. 訓練與評估 (Train & Evaluate)

**基本指令 (預設設定)**:
```bash
python part2/main.py
```
*   預設使用 `8x8` 地圖，開啟滑動模式 (`cheating=False`)。

**自定義參數範例**:
在 8x8 地圖上訓練，使用原始 Gymnasium 環境：
```bash
python part2/main.py \
    --map 8x8 \
    --agent DP
    --runs 10 \
    --cheating False \
    --eval_episodes 1000 \
    --is_slippery True \
    --render_mode ansi
```

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--map` | 8x8 | 地圖大小 (`4x4` 或 `8x8`) |
| `--runs` | 10 | 實驗重複次數 (取平均用) |
| `--agent` | QLearning | 使用的 Agent 類型 (`QLearning` 或 `DP`) |
| `--cheating` | True | 是否使用自定義的 LessSlippery 環境 |
| `--train_episodes` | 15000 | 訓練回合數 |
| `--eval_episodes` | 1000 | 評估回合數 |
| `--is_slippery` | True | 是否開啟滑動模式 |
| `--render_mode` | ansi | 渲染模式 (`ansi` 文字模式 或 `human` 視窗模式) |

### 2. 查看幫助 (Help)
查看所有可用的參數說明：
```bash
python part2/main.py --help
```

## 📈 實驗結果 (Results)

我們在不同設定下進行了廣泛測試，詳細數據存放在 `Result/` 資料夾中。

*   **4x4 Slippery**: 透過 Model-Based Q-Learning，Agent 能有效克服滑動，達到約 74% 的勝率。
*   **8x8 Slippery**: 即使在地圖變大且滑動的情況下，仍能維持約 63% 的勝率。
