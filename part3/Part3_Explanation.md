# Part 3: Choose Your Own Adventure - 任務說明

這張投影片說明了專案第三部分的目標與建議方向，主要目的是讓學生運用物件導向程式設計 (OOP) 的原則來設計並實作一個專案。

## 核心目標 (Goal)
*   **設計並實作一個專案**：
    *   必須展示 **OOP 原則** 的應用，例如：
        *   **抽象 (Abstraction)**
        *   **繼承 (Inheritance)**
        *   **多型 (Polymorphism)**
    *   同時藉此擴展你對程式設計的理解與實作經驗。

## 建議方向 (Suggestions)
投影片提供了兩個主要的實作方向，你可以選擇其中一個：

### 選項 1：使用經典控制環境 (Use a Classic Control Environment)
利用 Gymnasium 函式庫中現有的經典控制問題（例如：CartPole, MountainCar, 或 Acrobot）。
*   **實作要求**：
    *   為 **Agent (代理人)**、**Environment Interaction (環境互動)** 和 **Training Loop (訓練迴圈)** 實作專屬的類別 (Classes)。
    *   強調程式碼的 **模組化 (Modularity)** 和 **可重用性 (Reusability)**。

### 選項 2：建立自定義環境 (Create a Custom Environment)
設計並實作一個你自己簡單的環境，用來探索或視覺化某些概念。
*   **實作要求**：
    *   定義一個自定義的 `Env` 類別，必須遵循 **Gymnasium 的介面規範**（即必須包含 `reset`, `step`, `render` 等方法）。
    *   測試你的 Agent 如何適應你所設計的狀態 (State) 與獎勵結構 (Reward Structure) 並從中學習。

---
**總結**：這是一個開放式的任務，重點在於展現你如何將 OOP 的架構應用在強化學習 (RL) 的情境中，無論是封裝現有的演算法與環境，或是創造全新的環境。

## 範例程式碼說明 (Example Code Explanation)
`part3` 資料夾中提供了兩個 Python 檔案作為 **自定義環境 (Custom Environment)** 的範例實作，展示如何從頭開始建立一個符合 Gymnasium 標準的強化學習環境。

### 1. `warehouse_robot.py` (遊戲邏輯核心)
這個檔案負責定義 **遊戲本身的邏輯與規則**，它完全獨立於 Gymnasium，單純是一個用 Python 和 Pygame 寫的小遊戲。

*   **功能**：
    *   **定義角色與動作**：定義了機器人 (`Robot`)、目標 (`Target`) 以及機器人可以做的動作 (`LEFT`, `DOWN`, `RIGHT`, `UP`)。
    *   **遊戲狀態管理**：管理網格 (Grid) 的大小、機器人的位置、目標的位置。
    *   **核心邏輯**：
        *   `reset()`: 重置遊戲，將機器人放回起點 (0,0)，並隨機放置目標。
        *   `perform_action()`: 接收一個動作指令，移動機器人，並檢查是否撞牆或到達目標。
    *   **視覺化 (Rendering)**：使用 `pygame` 載入圖片 (sprites) 並繪製遊戲畫面。

### 2. `oop_project_env.py` (Gymnasium 介面封裝)
這個檔案是 **連接遊戲邏輯與 Gymnasium 框架的橋樑**。它將 `warehouse_robot.py` 的遊戲邏輯包裝成一個標準的 Gymnasium 環境 (`gym.Env`)。

*   **功能**：
    *   **繼承 `gym.Env`**：建立一個名為 `WarehouseRobotEnv` 的類別。
    *   **定義空間 (Spaces)**：
        *   `action_space`: 定義 Agent 可以做什麼動作。
        *   `observation_space`: 定義 Agent 可以看到什麼資訊 (機器人座標 + 目標座標)。
    *   **實作標準方法**：
        *   `reset()`: 呼叫遊戲的 `reset()`，並回傳初始觀察值。
        *   `step(action)`: 接收 Agent 的動作，呼叫遊戲的 `perform_action()`，並回傳觀察值、獎勵等資訊。
        *   `render()`: 呼叫遊戲的 `render()` 來顯示畫面。
    *   **註冊環境**：使用 `register()` 將這個環境註冊為 `warehouse-robot-v0`。

---
**架構總結**：這兩個檔案展示了 **關注點分離 (Separation of Concerns)** 的 OOP 原則：
*   `warehouse_robot.py` 專注於 **"怎麼玩" (遊戲機制)**。
*   `oop_project_env.py` 專注於 **"怎麼訓練" (RL 介面)**。
