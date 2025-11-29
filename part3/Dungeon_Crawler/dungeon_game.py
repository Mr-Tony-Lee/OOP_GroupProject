import pygame
import random
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
        # 假設沒有 wall.png，暫時用 floor.png 或其他替代，之後可以換
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
            print("Got the key!")
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
            print("You need a key to open this door!")
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
        super().__init__(row, col, "enemy.png") 

    def update(self, player_pos, grid_rows, grid_cols, grid):
        # 簡單 AI: 嘗試往玩家方向移動
        p_row, p_col = player_pos
        
        # 決定移動方向
        move_dir = None
        if random.random() < 0.6: # 60% 機率追蹤玩家
            d_row = p_row - self.row
            d_col = p_col - self.col
            
            if abs(d_row) > abs(d_col):
                move_dir = Direction.DOWN if d_row > 0 else Direction.UP
            else:
                move_dir = Direction.RIGHT if d_col > 0 else Direction.LEFT
        else:
            # 40% 機率隨機移動
            move_dir = random.choice(list(Direction))
            
        if move_dir:
            new_r, new_c = self.move(move_dir, grid_rows, grid_cols)
            
            # 檢查是否可以移動
            if 0 <= new_r < grid_rows and 0 <= new_c < grid_cols:
                target_obj = grid[new_r][new_c]
                # 假設怪物可以走在地板、陷阱、寶藏、鑰匙上，但不能穿牆或門
                if isinstance(target_obj, (Floor, Item, Trap, Treasure, Key)):
                    self.row = new_r
                    self.col = new_c

    def can_pass(self, player):
        return False # 玩家不能穿過怪物

    def on_enter(self, player):
        player.take_damage(20)
        return -20

# 遊戲核心類別
class DungeonGame:
    def __init__(self, map_layout=None, no_graphics=False):
        self.cell_size = 64
        self.map_layout = map_layout
        self.no_graphics = no_graphics
        if self.map_layout is None:
            # 預設地圖: W=Wall, P=Player, T=Treasure, .=Floor, X=Trap, K=Key, D=Door, E=Enemy
            self.map_layout = [
                ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W"],
                ["W", "P", ".", ".", ".", "W", ".", ".", ".", "E", ".", "W"],
                ["W", ".", "W", "W", ".", "W", ".", "W", "W", "W", ".", "W"],
                ["W", ".", "W", "K", ".", ".", ".", ".", ".", "W", ".", "W"],
                ["W", ".", "W", "W", "W", "W", "W", "W", ".", "W", ".", "W"],
                ["W", ".", ".", ".", ".", "E", ".", ".", ".", ".", ".", "W"],
                ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "D", "W"],
                ["W", ".", ".", "X", ".", ".", ".", ".", ".", ".", ".", "W"],
                ["W", ".", "W", "W", "W", "W", "W", "W", "W", "W", ".", "W"],
                ["W", ".", ".", ".", "E", ".", ".", "X", ".", ".", "T", "W"],
                ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W"]
            ]
        
        self.grid_rows = len(self.map_layout)
        self.grid_cols = len(self.map_layout[0])
        self.grid = [] # 存放 GameObject 的二維陣列
        self.player = None
        self.enemies = []
        self.game_over = False
        
        if not self.no_graphics:
            self._init_pygame()
        self.reset()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.window_size = (self.cell_size * self.grid_cols, self.cell_size * self.grid_rows)
        self.window_surface = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

    def reset(self):
        self.grid = []
        self.enemies = []
        self.game_over = False
        
        # 解析地圖並建立物件
        for r, row_data in enumerate(self.map_layout):
            grid_row = []
            for c, char in enumerate(row_data):
                obj = None
                if char == "W":
                    obj = Wall(r, c)
                elif char == "P":
                    # 地板上站著玩家
                    obj = Floor(r, c) 
                    self.player = Player(r, c)
                elif char == "T":
                    obj = Treasure(r, c)
                elif char == "X":
                    obj = Trap(r, c)
                elif char == "K":
                    obj = Key(r, c)
                elif char == "D":
                    obj = Door(r, c)
                elif char == "E":
                    # 地板上站著怪物
                    obj = Floor(r, c)
                    self.enemies.append(Enemy(r, c))
                else:
                    obj = Floor(r, c)
                grid_row.append(obj)
            self.grid.append(grid_row)

    def perform_action(self, action: Direction) -> bool:
        if self.game_over:
            return True

        # 1. 玩家移動
        new_r, new_c = self.player.move(action, self.grid_rows, self.grid_cols)
        target_obj = self.grid[new_r][new_c]
        
        # 檢查是否撞到怪物
        hit_enemy = False
        for enemy in self.enemies:
            if enemy.row == new_r and enemy.col == new_c:
                hit_enemy = True
                enemy.on_enter(self.player)
                print(f"Ouch! Ran into a monster! HP: {self.player.hp}")
                break
        
        if not hit_enemy and target_obj.can_pass(self.player):
            self.player.row = new_r
            self.player.col = new_c
            
            reward = target_obj.on_enter(self.player)
            
            if isinstance(target_obj, Treasure) and target_obj.collected:
                self.game_over = True
                print("You found the treasure!")
                return True 
            
            if isinstance(target_obj, Key) and target_obj.collected:
                self.grid[new_r][new_c] = Floor(new_r, new_c)
            
            if isinstance(target_obj, Trap):
                print(f"Ouch! Trap! HP: {self.player.hp}")
        elif not hit_enemy:
            print("Bonk! Hit a wall or door.")

        # 2. 怪物移動
        for enemy in self.enemies:
            enemy.update((self.player.row, self.player.col), self.grid_rows, self.grid_cols, self.grid)
            # 檢查怪物是否撞到玩家
            if enemy.row == self.player.row and enemy.col == self.player.col:
                enemy.on_enter(self.player)
                print(f"Monster attacked you! HP: {self.player.hp}")

        # 檢查死亡
        if self.player.hp <= 0:
            self.game_over = True
            print("You died!")
            return True

        return False

    def render(self):
        if self.no_graphics:
            return

        # 處理事件以保持視窗回應
        pygame.event.pump()

        # 繪製地圖
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                obj = self.grid[r][c]
                if obj.image:
                    img = pygame.transform.scale(obj.image, (self.cell_size, self.cell_size))
                    self.window_surface.blit(img, (c * self.cell_size, r * self.cell_size))
        
        # 繪製怪物
        for enemy in self.enemies:
            if enemy.image:
                # 為了區分，可以把怪物畫成紅色的 (如果用的是 bot_blue.png)
                # 這裡簡單用原圖
                img = pygame.transform.scale(enemy.image, (self.cell_size, self.cell_size))
                self.window_surface.blit(img, (enemy.col * self.cell_size, enemy.row * self.cell_size))

        # 繪製玩家
        if self.player and self.player.image:
            img = pygame.transform.scale(self.player.image, (self.cell_size, self.cell_size))
            self.window_surface.blit(img, (self.player.col * self.cell_size, self.player.row * self.cell_size))
            
        pygame.display.flip()
        self.clock.tick(30)

