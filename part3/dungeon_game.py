import pygame
from Item import Direction, Floor, Wall, Player, Treasure, Trap, Key, Door, Enemy

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
                ["W", "P", ".", ".", ".", "W", ".", ".", ".", ".", "X", "W"],
                ["W", ".", "W", "W", ".", "W", ".", "W", "W", "W", ".", "W"],
                ["W", ".", "W", "K", ".", ".", ".", ".", ".", "W", ".", "W"],
                ["W", ".", "W", "W", "W", "W", "W", "W", ".", "W", ".", "W"],
                ["W", ".", ".", ".", ".", ".", ".", ".", "E", ".", ".", "W"],
                ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "D", "W"],
                ["W", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "W"],
                ["W", ".", "W", "X", "W", ".", "W", "W", ".", "W", ".", "W"],
                ["W", "E", ".", ".", ".", ".", "T", ".", ".", ".", ".", "W"],
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
                break
        
        if not hit_enemy and target_obj.can_pass(self.player):
            self.player.row = new_r
            self.player.col = new_c
            
            reward = target_obj.on_enter(self.player)
            
            if isinstance(target_obj, Treasure) and target_obj.collected:
                self.game_over = True
                return True 
            
            if isinstance(target_obj, Key) and target_obj.collected:
                self.grid[new_r][new_c] = Floor(new_r, new_c)
            
            if isinstance(target_obj, Door):
                self.grid[new_r][new_c] = Floor(new_r, new_c)
            
            if isinstance(target_obj, Trap):
                pass
        elif not hit_enemy:
            pass

        # 2. 怪物移動
        for enemy in self.enemies:
            enemy.update((self.player.row, self.player.col), self.grid_rows, self.grid_cols, self.grid)
            # 檢查怪物是否撞到玩家
            if enemy.row == self.player.row and enemy.col == self.player.col:
                enemy.on_enter(self.player)
                pass

        # 檢查死亡
        if self.player.hp <= 0:
            self.game_over = True
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
                img = pygame.transform.scale(enemy.image, (self.cell_size, self.cell_size))
                self.window_surface.blit(img, (enemy.col * self.cell_size, enemy.row * self.cell_size))

        # 繪製玩家
        if self.player and self.player.image:
            img = pygame.transform.scale(self.player.image, (self.cell_size, self.cell_size))
            self.window_surface.blit(img, (self.player.col * self.cell_size, self.player.row * self.cell_size))
            
        pygame.display.flip()
        self.clock.tick(30)

