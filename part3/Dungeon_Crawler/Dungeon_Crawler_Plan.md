# 進階版「地牢探險」實作計畫 (Dungeon Crawler Implementation Plan)

這個計畫將引導你如何將原本簡單的 `warehouse_robot.py` 擴充為一個具有 OOP 架構的地牢探險遊戲。

## 階段一：OOP 架構設計 (Design Phase)
在開始寫程式之前，先定義好類別繼承關係。這是本專案最核心的部分。

### 1. 類別階層 (Class Hierarchy)
建議建立一個新的檔案 `dungeon_game.py`，並設計以下類別結構：

*   **`GameObject` (抽象基底類別)**
    *   屬性：`position` (x, y), `image` (sprite), `name`
    *   方法：`render(surface)`
*   **`Character` (繼承自 GameObject)**
    *   屬性：`hp` (血量)
    *   方法：`move(direction)`
    *   **`Player` (繼承自 Character)**: 玩家控制的角色。
    *   **`Enemy` (繼承自 Character)**: 會自動巡邏的怪物。
*   **`Item` (繼承自 GameObject)**
    *   方法：`on_collect(player)` (被撿起時發生的事)
    *   **`Treasure`**: 終點/寶藏 (遊戲勝利)。
    *   **`Trap`**: 陷阱 (扣分/扣血)。
    *   **`Key`**: 鑰匙 (解鎖門)。
    *   **`Door`**: 門 (有鑰匙才能通過)。
    *   **`Wall`**: 牆壁 (阻擋移動)。

---

## 階段二：重構遊戲核心 (Refactoring Core)
將 `warehouse_robot.py` 的邏輯移植到新的 OOP 架構。

### 步驟 1：建立基礎類別
在 `dungeon_game.py` 中定義 `GameObject` 及其子類別。
*   **任務**：把原本 `warehouse_robot.py` 裡面的 `RobotAction` 和 `GridTile` 保留，但邏輯要改寫。
*   **重點**：不再是用 `if tile == ROBOT` 這種寫法，而是每個格子裡存放的是一個 `GameObject` 的實例 (Instance)。

### 步驟 2：地圖設計 (Map Design)
設計一個更複雜的地圖。
*   可以使用一個二維陣列 (List of Lists) 來代表地圖佈局。
    ```python
    # W=Wall, P=Player, T=Treasure, .=Floor, X=Trap
    map_layout = [
        ["W", "W", "W", "W", "W"],
        ["W", "P", ".", "X", "W"],
        ["W", ".", "W", ".", "W"],
        ["W", ".", ".", "T", "W"],
        ["W", "W", "W", "W", "W"]
    ]
    ```
*   在 `__init__` 時讀取這個陣列，並生成對應的 `Wall`, `Player`, `Trap` 物件。

---

## 階段三：實作互動邏輯 (Interaction Logic)
這是遊戲好不好玩的關鍵。

### 步驟 1：移動與碰撞 (Movement & Collision)
修改 `perform_action` 方法。
*   當玩家想往右走時，先檢查右邊那格是什麼物件。
*   **多型 (Polymorphism) 的應用**：
    *   呼叫目標物件的 `can_pass()` 方法 (自定義)。
    *   如果是 `Wall` -> `can_pass()` 回傳 `False` (擋住)。
    *   如果是 `Door` -> 檢查玩家有沒有鑰匙，有則 `True`，無則 `False`。
    *   如果是 `Floor` 或 `Trap` -> `can_pass()` 回傳 `True`。

### 步驟 2：觸發事件 (Trigger Events)
當玩家進入某個格子時，觸發該物件的效果。
*   如果是 `Trap` -> 呼叫 `player.take_damage()`。
*   如果是 `Key` -> 呼叫 `player.add_item(key)` 並從地圖移除該物件。
*   如果是 `Treasure` -> 遊戲勝利。

---

## 階段四：Gymnasium 整合 (Gym Integration)
修改 `oop_project_env.py` 來適配你的新遊戲。

### 步驟 1：更新 Observation Space
原本的觀察值只有 `[robot_x, robot_y, target_x, target_y]`。
*   如果地圖變複雜了，建議改用 **Grid Observation** (回傳整個地圖的狀態)，或者簡化為 **Ray Casting** (玩家前後左右有什麼)。
*   **簡單版**：保持原本的觀察值，但加入「是否有鑰匙」的狀態：`[p_x, p_y, t_x, t_y, has_key]`。

### 步驟 2：更新 Reward Function
*   走到 `Trap`: reward = -1
*   走到 `Floor`: reward = -0.1 (鼓勵走快點)
*   拿到 `Treasure`: reward = +10
*   撞牆: reward = -0.5

---

## 階段五：進階功能 (Optional)
如果還有時間，可以加入怪物。

*   **怪物 AI**：在 `Enemy` 類別中實作 `update()` 方法。
*   簡單邏輯：隨機移動，或者向玩家靠近。
*   在 `env.step()` 的時候，除了移動玩家，也要呼叫 `enemy.update()` 讓怪物移動。

## 實作檢核表 (Checklist)
- [ ] 定義 `GameObject` 及其子類別 (Wall, Trap, Treasure)。
- [ ] 設計一張包含牆壁和陷阱的地圖。
- [ ] 實作玩家移動與牆壁碰撞偵測。
- [ ] 實作陷阱扣分與寶藏得分邏輯。
- [ ] (進階) 實作鑰匙與門的機制。
- [ ] 更新 `oop_project_env.py` 以支援新遊戲。
