# 資料夾說明

```
part2/
|-- Class_version/      # 使用類別(Class)實作的版本( 我覺得太醜把它改成class的版本，然後改的內容都在裡面)
|-- Origin/             # 原始程式碼版本(一開始老師給的)
|-- README.md           # 資料夾說明
```

## Detail

### Class_version
- class_frozen_lake.py : 修改過後的使用 Q-Learning 演算法解決 Frozen Lake 問題的類別實作版本。
    - 修改內容：
        - 一些 training 的參數調整。 
        - Q-learning 更新邏輯改為使用環境的轉移機率 ( env.unwrapped.P ) 計算期望目標值。
    - 成功率：
        - 4x4 slippery 穩定在 74% 上下
        - 8x8 slippery 穩定在 63% 上下
        

- dp_frozen_lake.py : 使用 動態規劃(Dynamic Programming) 演算法解決 Frozen Lake 問題的類別實作版本。(網路上抄的，目的是想知道Q-Learning 的上限在哪)
    - 新增內容：
        - 實作一個 FrozenLakeDP 類別，使用 **Value Iteration** (動態規劃) 來解決環境。
        - 包含價值迭代和提取最優策略的方法。
    - 成功率：
        - 4x4 slippery 穩定在 74% 左右
        - 8x8 slippery 穩定在 64% 左右
        
### Origin
- Frozen_lake_Explanation.md : 說明文件，介紹 Frozen Lake 問題及 Q-Learning 演算法的基本概念、使用建議、輸出檔案說明及常見問題解答。
- frozen_lake.py : 一開始老師給的，使用 Q-Learning 演算法解決 Frozen Lake 問題的原始程式碼版本。