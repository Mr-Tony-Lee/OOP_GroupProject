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
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.game.grid_rows, self.game.grid_cols, 1]),
            shape=(3,),
            dtype=np.int32
        )

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
        
        terminated = self.game.perform_action(dg.Direction(action))
        
        # 計算 Reward
        reward = -0.1 # 每一步扣一點分，鼓勵盡快完成
        
        # 根據遊戲狀態變化給予額外獎勵
        # 1. 拿到寶藏 (遊戲結束且勝利)
        if terminated and self.game.player.hp > 0:
            reward += 100
        
        # 2. 踩到陷阱 (扣血)
        hp_loss = prev_hp - self.game.player.hp
        if hp_loss > 0:
            reward -= hp_loss # 扣多少血就扣多少分
            
        # 3. 撿到鑰匙 (雖然 game 裡面沒有直接回傳，但我們可以檢查 has_key)
        # 這裡比較難偵測 "剛撿到"，除非我們在 game 裡加 flag
        # 簡單做法：如果這一布導致 has_key 變成 True，給獎勵
        # (需要紀錄上一步的 has_key，這裡先省略，假設 AI 會自己學)

        # 檢查是否死亡 (HP <= 0)
        if self.game.player.hp <= 0:
            terminated = True
            reward -= 50 # 死亡懲罰

        obs = self._get_obs()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        self.game.render()

    def _get_obs(self):
        return np.array([
            self.game.player.row, 
            self.game.player.col, 
            1 if self.game.player.has_key else 0
        ], dtype=np.int32)

    def close(self):
        if self.render_mode == "human":
            import pygame
            pygame.display.quit()
            pygame.quit()
