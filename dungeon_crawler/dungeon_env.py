import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import dungeon_game as dg
from abc import ABC, abstractmethod


# ============================================================================
# 基礎類別：DungeonCrawlerEnv (適配器模式 + 策略模式)
# ============================================================================
class DungeonCrawlerEnv(gym.Env, ABC):
    """
    DungeonCrawler 環境的基類
    
    設計特點:
    - 使用抽象基類定義通用接口
    - 所有子類共享 step(), reset(), render() 等核心邏輯
    - 只在觀察空間和觀察值構建上有差異
    
    Channels (統一):
    - 0: Wall (靜態)
    - 1: Player (動態)
    - 2: Enemy (動態)
    - 3: Key (靜態/半動態)
    - 4: Door (靜態/半動態)
    - 5: Treasure (靜態)
    - 6: Trap (靜態)
    - 7: HasKey (動態 - 全局信息)
    """
    
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 300  # 設定最大步數限制，避免無限迴圈

        # 初始化遊戲 (如果 render_mode 是 None，則不開啟圖形介面)
        self.game = dg.DungeonGame(no_graphics=(render_mode is None))
        
        # Action Space: 上下左右 (4個離散動作)
        self.action_space = spaces.Discrete(len(dg.Direction))
        
        # 觀察空間由子類定義
        self.observation_space = self._define_observation_space()
        
        # 初始化位置追蹤
        self.key_pos = None
        self.door_pos = None
        self.treasure_pos = None
        self.base_obs = None

    @abstractmethod
    def _define_observation_space(self):
        """由子類定義觀察空間"""
        pass

    @abstractmethod
    def _get_obs(self):
        """由子類實作觀察值的構建"""
        pass

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
        """
        執行一個動作步驟
        
        Reward 設計:
        - 基礎: -0.1 (每步懲罰，鼓勵快速完成)
        - 撞牆: -5.0 (避免卡住)
        - 找到鑰匙: +50
        - 開門: +10
        - 靠近目標: +0.1
        - 遠離目標: -0.1
        - 踩到陷阱: -hp_loss
        - 贏得遊戲: +100
        - 死亡: -50
        """
        
        prev_hp = self.game.player.hp
        prev_row, prev_col = self.game.player.row, self.game.player.col
        prev_has_key = self.game.player.has_key
        
        # 預先檢查是否嘗試開門
        target_r, target_c = self.game.player.move(dg.Direction(action), self.game.grid_rows, self.game.grid_cols)
        target_obj = self.game.grid[target_r][target_c]
        is_door = isinstance(target_obj, dg.Door)

        terminated = self.game.perform_action(dg.Direction(action))
        
        # 計算 Reward
        reward = -0.1  # 每一步扣一點分，鼓勵盡快完成
        events = []

        # 0. 撞牆懲罰 (位置沒變且沒結束)
        if not terminated and self.game.player.row == prev_row and self.game.player.col == prev_col:
            reward -= 5.0  # 大幅增加撞牆扣分，避免 Agent 卡在牆邊
        
        # 1. 拿到寶藏 (遊戲結束且勝利)
        if terminated and self.game.player.hp > 0:
            reward += 100
            events.append("won")
        
        # 2. 踩到陷阱 (扣血)
        hp_loss = prev_hp - self.game.player.hp
        if hp_loss > 0:
            reward -= hp_loss  # 扣多少血就扣多少分
            
        # 3. 撿到鑰匙 (給予大獎勵)
        if not prev_has_key and self.game.player.has_key:
            reward += 50  # 鼓勵去撿鑰匙
            events.append("found_key")
            # 更新 Base Obs: 移除鑰匙
            self.base_obs[3, self.game.player.row, self.game.player.col] = 0

        # 4. 開門 (給予獎勵)
        if is_door and self.game.player.row == target_r and self.game.player.col == target_c:
            reward += 10  # 開門獎勵
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

    def close(self):
        if self.render_mode == "human":
            import pygame
            pygame.display.quit()
            pygame.quit()


# ============================================================================
# 子類 1: CNNDungeonEnv - 為 CNN 優化的觀察格式
# ============================================================================
class CNNDungeonEnv(DungeonCrawlerEnv):
    """
    CNN 專用環境
    
    觀察空間特點:
    - 返回 Dict 格式: {"image": ..., "scalars": ...}
    - image: (7, H, W) - 7 個通道 (不含 haskey 通道)
    - scalars: (1,) - has_key 信息
    
    這個設計讓 CNN 可以：
    1. 用卷積層處理空間信息
    2. 用額外輸入層處理標量信息 (has_key)
    3. 整合兩種信息進行決策
    """
    
    def _define_observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(
                low=0, 
                high=1, 
                shape=(7, self.game.grid_rows, self.game.grid_cols), 
                dtype=np.float32
            ),
            "scalars": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    def _get_obs(self):
        """
        返回 CNN 格式的觀察
        
        image channels:
        - 0: 玩家位置
        - 1: 牆壁
        - 2: 鑰匙
        - 3: 門
        - 4: 寶藏
        - 5: 陷阱
        - 6: 怪物
        
        scalars:
        - 0: has_key (0 或 1)
        """
        image = np.zeros((7, self.game.grid_rows, self.game.grid_cols), dtype=np.float32)

        # Channel 0: 玩家位置
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

                # Channel 3: 門 (Door)
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


# ============================================================================
# 子類 2: DQNDungeonEnv - 為 DQN 優化的觀察格式
# ============================================================================
class DQNDungeonEnv(DungeonCrawlerEnv):
    """
    DQN 專用環境
    
    觀察空間特點:
    - 返回單一 Box 格式: (8, H, W)
    - 8 個通道包含所有信息 (含 haskey 通道)
    - 所有物體和狀態都在同一張 3D 張量中
    
    這個設計讓 DQN 可以：
    1. 用統一的 CNN 架構處理所有信息
    2. Channel 7 (haskey) 充當全局廣播信息
    3. 不需要複雜的多輸入邏輯
    """
    
    def _define_observation_space(self):
        return spaces.Box(
            low=0,
            high=1,
            shape=(8, self.game.grid_rows, self.game.grid_cols),
            dtype=np.float32
        )

    def _get_obs(self):
        """
        返回 DQN 格式的觀察
        
        channels:
        - 0: 牆壁 (Wall)
        - 1: 玩家 (Player)
        - 2: 怪物 (Enemy)
        - 3: 鑰匙 (Key)
        - 4: 門 (Door)
        - 5: 寶藏 (Treasure)
        - 6: 陷阱 (Trap)
        - 7: 是否有鑰匙 (HasKey) - 全局廣播到整個通道
        """
        # 複製 Base Obs (包含 Wall, Key, Door, Treasure, Trap)
        obs = self.base_obs.copy()
        
        # 更新動態物件
        # Player (Channel 1)
        obs[1, self.game.player.row, self.game.player.col] = 1
        
        # Enemies (Channel 2)
        for enemy in self.game.enemies:
            obs[2, enemy.row, enemy.col] = 1
            
        # Has Key (Channel 7) - Global info broadcast to whole channel
        if self.game.player.has_key:
            obs[7, :, :] = 1
            
        return obs


# ============================================================================
# 子類 3: PPODungeonEnv - 與 PPOAgent 相容的單一張量觀察
# ============================================================================
class PPODungeonEnv(DungeonCrawlerEnv):
    """
    PPO 專用環境

    觀察空間：單一 Box, shape = (8, H, W)
    - 與 DQNDungeonEnv 相同的通道設計，方便直接餵給 PPO 的 CNN Actor-Critic。
    - 若未來想改為多輸入(image + scalar)，可再新增一個變體。
    """

    def _define_observation_space(self):
        return spaces.Box(
            low=0,
            high=1,
            shape=(8, self.game.grid_rows, self.game.grid_cols),
            dtype=np.float32
        )

    def _get_obs(self):
        # 與 DQNDungeonEnv 相同：
        obs = self.base_obs.copy()

        # Player (Channel 1)
        obs[1, self.game.player.row, self.game.player.col] = 1

        # Enemies (Channel 2)
        for enemy in self.game.enemies:
            obs[2, enemy.row, enemy.col] = 1

        # Has Key (Channel 7) - Global broadcast
        if self.game.player.has_key:
            obs[7, :, :] = 1

        return obs


# ============================================================================
# 環境註冊
# ============================================================================
# 註冊 CNN 版本
register(
    id='dungeon-crawler-cnn-v0',
    entry_point='dungeon_env:CNNDungeonEnv',
)

# 註冊 DQN 版本
register(
    id='dungeon-crawler-dqn-v0',
    entry_point='dungeon_env:DQNDungeonEnv',
)

# 保留原始名稱指向 DQN 版本 (後向兼容)
register(
    id='dungeon-crawler-v0',
    entry_point='dungeon_env:DQNDungeonEnv',
)

# 註冊 PPO 版本
register(
    id='dungeon-crawler-ppo-v0',
    entry_point='dungeon_env:PPODungeonEnv',
)