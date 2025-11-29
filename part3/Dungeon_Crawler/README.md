# Dungeon Crawler RL Project

這是一個結合 **物件導向程式設計 (OOP)** 與 **強化學習 (Reinforcement Learning)** 的期末專案。我們從零開始打造了一個符合 Gymnasium 標準的地牢探險環境，並實作了 Q-Learning Agent 來自動破解關卡。

## 專案特色 (Features)

1.  **物件導向架構 (OOP Architecture)**:
    *   使用 `GameObject` 作為基底類別，衍生出 `Character` (角色) 與 `Item` (物品)。
    *   利用 **多型 (Polymorphism)** 處理不同物件的互動邏輯 (如：撞牆、踩陷阱、撿鑰匙、開門)。
    *   利用 **繼承 (Inheritance)** 實作 `Player` 與 `Enemy`，共享移動邏輯但擁有不同的行為模式。

2.  **自定義 Gymnasium 環境 (Custom Environment)**:
    *   完全符合 Gymnasium API 標準 (`reset`, `step`, `render`)。
    *   定義了自定義的 Action Space (上下左右) 與 Observation Space。
    *   設計了包含獎勵 (Reward) 與懲罰 (Penalty) 的機制來引導 Agent 學習。

3.  **強化學習實作 (RL Implementation)**:
    *   實作 **Q-Learning** 演算法。
    *   Agent 能夠自主探索環境，學習「先撿鑰匙 -> 再開門 -> 最後拿寶藏」的最佳路徑。
    *   包含訓練 (Training) 與測試 (Testing) 流程，並可儲存/載入模型。

## 檔案結構 (File Structure)

位於 `part3/Dungeon_Crawler/` 資料夾下：

*   **`dungeon_game.py`**: 遊戲核心邏輯。包含所有 OOP 類別 (`GameObject`, `Player`, `Enemy`, `Key`, `Door` 等) 以及遊戲迴圈。
*   **`dungeon_env.py`**: Gymnasium 環境封裝。將遊戲邏輯包裝成標準 RL 環境，定義狀態 (State) 與獎勵 (Reward)。
*   **`agent.py`**: 定義 `Agent` 抽象類別與 `QLearningAgent` 實作。包含 Q-Table 更新公式與 Epsilon-Greedy 策略。
*   **`train.py`**: 訓練與測試腳本。負責執行訓練迴圈，繪製學習曲線，並展示最終成果。
*   **`human.py`**: 人類手動試玩腳本。提供圖形介面讓你親自挑戰迷宮。

## 安裝需求 (Requirements)

請確保安裝以下 Python 套件：

```bash
pip install gymnasium pygame numpy matplotlib
```

## 如何執行 (How to Run)

### 1. 手動試玩 (Human Play)
親自挑戰這個 11x12 的複雜迷宮！
*   **操作**：方向鍵移動，`R` 重置，`ESC` 離開。
*   **目標**：避開怪物與陷阱 -> 拿到鑰匙 (K) -> 打開門 (D) -> 取得寶藏 (T)。
```bash
python part3/Dungeon_Crawler/human.py
```

### 2. 訓練 AI (Train Agent)
讓 AI 從零開始學習如何玩這個遊戲。
*   程式會執行 1000 個回合的訓練。
*   訓練完成後會自動儲存模型 (`q_table.pkl`) 並繪製學習曲線 (`training_curve.png`)。
*   最後會自動開啟視窗展示 AI 的學習成果。
```bash
python part3/Dungeon_Crawler/train.py
```

### 3. 測試環境 (Test Environment)
單純測試 Gymnasium 環境是否能正常運作 (隨機亂走)。
```bash
python part3/Dungeon_Crawler/test_env.py
```

## 遊戲規則 (Game Rules)

*   **地圖**: 11x12 的大型迷宮，包含多個房間與長廊。
*   **目標**: 拿到寶藏 (Treasure, 📦)。
*   **障礙**:
    *   **牆壁 (Wall)**: 無法穿越。
    *   **門 (Door)**: 需要鑰匙才能通過，擋在寶藏房門口。
    *   **陷阱 (Trap)**: 踩到會扣血並扣分。
    *   **怪物 (Enemy)**: 3 隻巡邏的怪物，會追蹤玩家，碰到會大量扣血。
*   **道具**:
    *   **鑰匙 (Key)**: 藏在迷宮深處，用來打開門。

## 學習成果 (Results)
經過約 500-800 個回合的訓練，Q-Learning Agent 能夠學會：
1.  避開陷阱與怪物。
2.  繞路去撿鑰匙。
3.  打開門並直奔寶藏。
