import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import dungeon_game as dg

# 註冊環境
register(
    id='dungeon-crawler-v0',
    entry_point='dungeon_env:DungeonCrawlerEnv',
)

class DungeonCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.max_steps = 500 # 設定最大步數限制，避免無限迴圈
        
        # 初始化遊戲 (如果 render_mode 是 None，則不開啟圖形介面)
        self.game = dg.DungeonGame(no_graphics=(render_mode is None))
        
        # Action Space: 上下左右 (4個離散動作)
        self.action_space = spaces.Discrete(len(dg.Direction))

        # Observation Space: 
        # 使用 Multi-Channel Grid (C, H, W)
        # Channels: 0:Wall, 1:Player, 2:Enemy, 3:Key, 4:Door, 5:Treasure, 6:Trap, 7:HasKey
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(8, self.game.grid_rows, self.game.grid_cols),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置遊戲
        self.game.reset()
        
        # 初始化 Base Observation (包含靜態物件)
        # Channels: 0:Wall, 1:Player, 2:Enemy, 3:Key, 4:Door, 5:Treasure, 6:Trap, 7:HasKey
        self.base_obs = np.zeros((8, self.game.grid_rows, self.game.grid_cols), dtype=np.float32)
        
        self.current_step = 0 # 重置步數計數器

        self.key_pos = None
        self.door_pos = None
        self.treasure_pos = None

        for r in range(self.game.grid_rows):
            for c in range(self.game.grid_cols):
                obj = self.game.grid[r][c]
                if isinstance(obj, dg.Wall):
                    self.base_obs[0, r, c] = 1
                elif isinstance(obj, dg.Key):
                    self.base_obs[3, r, c] = 1
                    self.key_pos = (r, c)
                elif isinstance(obj, dg.Door):
                    self.base_obs[4, r, c] = 1
                    self.door_pos = (r, c)
                elif isinstance(obj, dg.Treasure):
                    self.base_obs[5, r, c] = 1
                    self.treasure_pos = (r, c)
                elif isinstance(obj, dg.Trap):
                    self.base_obs[6, r, c] = 1
        
        # 建構觀察值
        obs = self._get_obs()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return obs, info

    def step(self, action):
        # 執行動作
        # 注意：dungeon_game 的 perform_action 回傳的是 "是否結束遊戲"
        # 但我們需要更詳細的資訊來計算 reward
        
        # 為了計算 reward，我們需要知道動作前後的狀態變化
        # 這裡我們稍微修改一下邏輯，直接呼叫 game.perform_action
        # 但因為 game.perform_action 封裝了太多邏輯 (包含 print)，
        # 在 RL 訓練時通常不希望有太多 print。
        # 不過為了作業方便，我們先直接用。
        
        prev_score = self.game.player.score
        prev_hp = self.game.player.hp
        prev_row, prev_col = self.game.player.row, self.game.player.col
        prev_has_key = self.game.player.has_key
        
        # 預先檢查是否嘗試開門
        target_r, target_c = self.game.player.move(dg.Direction(action), self.game.grid_rows, self.game.grid_cols)
        target_obj = self.game.grid[target_r][target_c]
        is_door = isinstance(target_obj, dg.Door)

        terminated = self.game.perform_action(dg.Direction(action))
        
        # 計算 Reward
        reward = -0.1 # 每一步扣一點分，鼓勵盡快完成
        events = []

        # 0. 撞牆懲罰 (位置沒變且沒結束)
        if not terminated and self.game.player.row == prev_row and self.game.player.col == prev_col:
            reward -= 0.5 # 降低撞牆扣分 (原本 -5.0 太重了，會導致 Agent 不敢探索)
        
        # 1. 拿到寶藏 (遊戲結束且勝利)
        if terminated and self.game.player.hp > 0:
            reward += 100
            events.append("won")
        
        # 2. 踩到陷阱 (扣血)
        hp_loss = prev_hp - self.game.player.hp
        if hp_loss > 0:
            reward -= hp_loss # 扣多少血就扣多少分
            
        # 3. 撿到鑰匙 (給予大獎勵)
        if not prev_has_key and self.game.player.has_key:
            reward += 50 # 鼓勵去撿鑰匙
            events.append("found_key")
            # 更新 Base Obs: 移除鑰匙
            self.base_obs[3, self.game.player.row, self.game.player.col] = 0

        # 4. 開門 (給予獎勵)
        if is_door and self.game.player.row == target_r and self.game.player.col == target_c:
            reward += 10 # 開門獎勵
            events.append("opened_door")
            # 更新 Base Obs: 移除門
            self.base_obs[4, target_r, target_c] = 0

        # 5. 距離獎勵 (Distance Reward)
        # 找出當前目標 (如果有鑰匙 -> 找門/寶藏，如果沒鑰匙 -> 找鑰匙)
        target_pos = None
        if not self.game.player.has_key:
            # 找鑰匙
            target_pos = self.key_pos
        else:
            # 找門或寶藏
            # 優先找門 (如果門還在)
            if self.door_pos:
                r, c = self.door_pos
                if isinstance(self.game.grid[r][c], dg.Door):
                    target_pos = self.door_pos
                else:
                    target_pos = self.treasure_pos
            else:
                target_pos = self.treasure_pos
        
        if target_pos:
            # 計算曼哈頓距離
            curr_dist = abs(self.game.player.row - target_pos[0]) + abs(self.game.player.col - target_pos[1])
            prev_dist = abs(prev_row - target_pos[0]) + abs(prev_col - target_pos[1])
            
            if curr_dist < prev_dist:
                reward += 0.1 # 靠近目標
            elif curr_dist > prev_dist:
                reward -= 0.1 # 遠離目標

        # 檢查是否死亡 (HP <= 0)
        if self.game.player.hp <= 0:
            terminated = True
            reward -= 50 # 死亡懲罰
            events.append("died")

        # 檢查是否超時
        self.current_step += 1
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"events": events}

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.game.render()

    def _get_obs(self):
        # 複製 Base Obs (包含 Wall, Key, Door, Treasure, Trap)
        obs = self.base_obs.copy()
        
        # 更新動態物件
        # Player
        obs[1, self.game.player.row, self.game.player.col] = 1
        
        # Enemies
        for enemy in self.game.enemies:
            obs[2, enemy.row, enemy.col] = 1
            
        # Has Key (Global info broadcast to whole channel)
        if self.game.player.has_key:
            obs[7, :, :] = 1
            
        return obs

    def close(self):
        if self.render_mode == "human":
            import pygame
            pygame.display.quit()
            pygame.quit()
