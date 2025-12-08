# 資料夾說明

```
part2/
|-- Result/             # 存放結果圖表和模型
|-- Agent.py            # FrozenLake Agent 類別實作
|-- CheatingEnv.py      # 自定義的 Frozen Lake 環境類別
|-- main.py             # 主程式，負責執行訓練和評估流程
|-- README.md           # 資料夾說明
```

## Detail

### Agent.py : FrozenLake Agent 類別實作
#### QLearningAgent : 
- 修改內容：
    - 一些 training 的參數調整。 
    - Q-learning 更新邏輯改為使用環境的轉移機率 ( env.unwrapped.P ) 計算期望目標值。
- 成功率：
    - 4x4 slippery 穩定在 74% 上下
    - 8x8 slippery 穩定在 63% 上下
### CheatingEnv.py : 自定義的 Frozen Lake 環境類別
#### LessSlipperyFrozenLakeEnv : 
- 修改自 gymnasium 的 FrozenLakeEnv，將冰面滑動機率降低，使得 agent 更容易學習。

### main.py : 主程式，負責解析命令列參數並執行訓練和評估流程。 
- 可從 command line 調整參數
    - --map 指定地圖
    - --runs 指定執行次數
    - --train_episodes 指定訓練回合數
    - --eval_episodes 指定評估回合數
    - --is_slippery 指定環境是否為滑動模式 (True/False)
    - --render_mode 指定環境的渲染模式 (ansi/human)
    - --cheating 指定是否作弊 (True/False)
```bash
Usage:
    python3 main.py --help # 查看幫助訊息
    python3 main.py [--map MAP_NAME] [--runs NUM_RUNS] [--cheating CHEATING] [--train_episodes TRAIN_EPISODES] [--eval_episodes EVAL_EPISODES] [--is_slippery IS_SLIPPERY] [--render_mode RENDER_MODE]

Examples:
    python3 main.py --map 4x4 --runs 10 --cheating True --train_episodes 15000 --eval_episodes 1000 --is_slippery True --render_mode ansi

Default:
    python3 main.py --map 8x8 --runs 10 --cheating True --train_episodes 15000 --eval_episodes 1000 --is_slippery True --render_mode ansi
```
