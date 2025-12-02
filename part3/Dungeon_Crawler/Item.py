import random
import pygame
from enum import Enum
from abc import ABC, abstractmethod
from os import path

# 定義方向枚舉
class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

# 抽象基底類別：所有遊戲物件的父類別
class GameObject(ABC):
    def __init__(self, row, col, image_path=None):
        self.row = row
        self.col = col
        self.image = None
        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path):
        # 這裡假設 sprites 資料夾在上一層目錄的 sprites 中，或者你可以調整路徑
        # 為了方便，我們先預留載入圖片的邏輯
        try:
            full_path = path.join(path.dirname(__file__), "sprites", image_path)
            if path.exists(full_path):
                self.image = pygame.image.load(full_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    @abstractmethod
    def can_pass(self, player):
        """回傳此物件是否可以被穿越"""
        pass

    @abstractmethod
    def on_enter(self, player):
        """當玩家進入此格子時觸發的事件"""
        pass

# 地板：最基本的物件
class Floor(GameObject):
    def __init__(self, row, col):
        super().__init__(row, col, "floor.png")

    def can_pass(self, player):
        return True

    def on_enter(self, player):
        # 走在地板上沒事發生，或者可以扣一點點分數當作時間成本
        return 0 # Reward

# 牆壁：不可穿越
class Wall(GameObject):
    def __init__(self, row, col):
        super().__init__(row, col, "wall.png") 

    def can_pass(self, player):
        return False

    def on_enter(self, player):
        return 0

# 角色基底類別
class Character(GameObject):
    def __init__(self, row, col, image_path, hp=100):
        super().__init__(row, col, image_path)
        self.hp = hp

    def can_pass(self, player):
        return False # 通常角色不能重疊

    def on_enter(self, player):
        return 0

    def move(self, direction, grid_rows, grid_cols):
        new_row, new_col = self.row, self.col
        
        if direction == Direction.LEFT:
            new_col -= 1
        elif direction == Direction.RIGHT:
            new_col += 1
        elif direction == Direction.UP:
            new_row -= 1
        elif direction == Direction.DOWN:
            new_row += 1
            
        # 邊界檢查
        if 0 <= new_row < grid_rows and 0 <= new_col < grid_cols:
            return new_row, new_col
        return self.row, self.col

# 玩家類別
class Player(Character):
    def __init__(self, row, col):
        super().__init__(row, col, "bot_blue.png")
        self.has_key = False
        self.score = 0

    def add_score(self, points):
        self.score += points

    def take_damage(self, damage):
        self.hp -= damage

# 物品基底類別
class Item(GameObject):
    def __init__(self, row, col, image_path):
        super().__init__(row, col, image_path)
        self.collected = False

    def can_pass(self, player):
        return True

# 鑰匙
class Key(Item):
    def __init__(self, row, col):
        super().__init__(row, col, "key.png")

    def on_enter(self, player):
        if not self.collected:
            self.collected = True
            player.has_key = True
            return 10
        return 0

# 門
class Door(GameObject):
    def __init__(self, row, col):
        super().__init__(row, col, "door.png")

    def can_pass(self, player):
        if player.has_key:
            return True
        else:
            return False

    def on_enter(self, player):
        return 0

# 寶藏 (終點)
class Treasure(Item):
    def __init__(self, row, col):
        super().__init__(row, col, "package.png")

    def on_enter(self, player):
        if not self.collected:
            self.collected = True
            return 100 # 獲得大獎勵
        return 0

# 陷阱
class Trap(Item):
    def __init__(self, row, col):
        # 假設有個 trap.png
        super().__init__(row, col, "trap.png")

    def on_enter(self, player):
        player.take_damage(10)
        return -10 # 扣分

# 怪物
class Enemy(Character):
    def __init__(self, row, col):
        # 暫時用 bot_blue.png，如果有 enemy.png 更好
        super().__init__(row, col, "enemy.png") 
        self.current_dir = random.choice(list(Direction)) # 初始隨機方向

    def update(self, player_pos, grid_rows, grid_cols, grid):
        # 巡邏 AI (Patrol AI):
        # 怪物會持續往同一個方向移動，直到撞牆才轉彎。
        # 這樣比較像 "自己會移動" (有自己的意圖)，而且比追蹤玩家容易預測 (降低難度)。
        
        # 嘗試往當前方向移動
        new_r, new_c = self.move(self.current_dir, grid_rows, grid_cols)
        
        blocked = False
        # 檢查邊界
        if 0 <= new_r < grid_rows and 0 <= new_c < grid_cols:
            target_obj = grid[new_r][new_c]
            
            # 檢查障礙物: 怪物不能穿牆、門
            # 但可以走在地板、陷阱、寶藏、鑰匙上
            if isinstance(target_obj, (Wall, Door)):
                blocked = True
        else:
            blocked = True
            
        if not blocked:
            self.row = new_r
            self.col = new_c
        else:
            # 撞牆了，隨機換一個方向 (排除原本的方向)
            possible_dirs = list(Direction)
            if self.current_dir in possible_dirs:
                possible_dirs.remove(self.current_dir)
            self.current_dir = random.choice(possible_dirs)

    def can_pass(self, player):
        return False # 玩家不能穿過怪物

    def on_enter(self, player):
        player.take_damage(100)
        return -100