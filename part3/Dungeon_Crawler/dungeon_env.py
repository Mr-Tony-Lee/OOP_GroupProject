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
        
        # 初始化遊戲 (如果 render_mode 是 None，則不開啟圖形介面)
        self.game = dg.DungeonGame(no_graphics=(render_mode is None))
        
        # Action Space: 上下左右 (4個離散動作)
        self.action_space = spaces.Discrete(len(dg.Direction))

        # Observation Space: 
        # 為了簡化，我們回傳一個向量包含：
        # [player_row, player_col, has_key(0/1)]
        # 如果要讓 AI 更聰明，可以考慮回傳整個地圖的狀態，或者 Ray Casting
        self.observation_space = spaces.Dict({
            #  player, wall, key, door, treasure, trap, monster
            "image": spaces.Box(low=0, high=1, shape=(7, self.game.grid_rows, self.game.grid_cols), dtype=np.float32),
            # has key
            "scalars": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置遊戲
        self.game.reset()
        
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
            reward -= 0.5 # 撞牆扣分
        
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
            # print("Reward: Found Key! (+50)")

        # 4. 開門 (給予獎勵)
        if is_door and self.game.player.row == target_r and self.game.player.col == target_c:
            reward += 10 # 開門獎勵
            events.append("opened_door")
            # print("Reward: Opened Door! (+10)")

        # 檢查是否死亡 (HP <= 0)
        if self.game.player.hp <= 0:
            terminated = True
            reward -= 50 # 死亡懲罰
            events.append("died")

        obs = self._get_obs()
        info = {"events": events}

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        self.game.render()

    def _get_obs(self):
        image = np.zeros((7, self.game.grid_rows, self.game.grid_cols), dtype=np.float32)

        # Channel 0: 玩家位置 (Player)
        image[0, self.game.player.row, self.game.player.col] = 1.0

        for r in range(self.game.grid_rows):
            for c in range(self.game.grid_cols):
                obj = self.game.grid[r][c]
                
                # Channel 1: 牆壁 (Wall) - 永久障礙
                if isinstance(obj, dg.Wall):
                    image[1, r, c] = 1.0
                
                # Channel 2: 鑰匙 (Key)
                elif isinstance(obj, dg.Key) and not obj.collected:
                    image[2, r, c] = 1.0

                # Channel 3: 門 (Door) - 需要鑰匙的障礙
                # 注意: 根據你的 game 邏輯，門打開後會變 Floor，這裡就會自動變回 0，這很棒
                elif isinstance(obj, dg.Door):
                    image[3, r, c] = 1.0
                
                # Channel 4: 寶藏 (Treasure)
                elif isinstance(obj, dg.Treasure) and not obj.collected:
                    image[4, r, c] = 1.0
                
                # Channel 5: 陷阱 (Trap)
                elif isinstance(obj, dg.Trap):
                    image[5, r, c] = 1.0
        
        # Channel 6: 怪物 (Monster)
        for enemy in self.game.enemies:
            image[6, enemy.row, enemy.col] = 1.0

        has_key_val = 1.0 if self.game.player.has_key else 0.0
        scalars = np.array([has_key_val], dtype=np.float32)

        return {"image": image, "scalars": scalars}

    def close(self):
        if self.render_mode == "human":
            import pygame
            pygame.display.quit()
            pygame.quit()
