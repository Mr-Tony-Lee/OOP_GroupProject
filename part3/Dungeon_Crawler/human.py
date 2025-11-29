import pygame
import sys
from dungeon_game import DungeonGame, Direction

def main():
    # Initialize the game
    # no_graphics=False 確保開啟視窗
    game = DungeonGame(no_graphics=False)
    
    print("=== Dungeon Crawler Human Mode ===")
    print("Use Arrow Keys to move.")
    print("Goal: Get the Key (K) -> Open the Door (D) -> Get the Treasure (T)")
    print("Avoid: Walls (W), Traps (X), Enemies (E)")
    print("Press 'R' to reset level.")
    print("Press 'ESC' to quit.")

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    game.perform_action(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    game.perform_action(Direction.RIGHT)
                elif event.key == pygame.K_UP:
                    game.perform_action(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    game.perform_action(Direction.DOWN)
                elif event.key == pygame.K_r: # Reset
                    game.reset()
                    print("Game Reset")

        # Render the game
        game.render()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
