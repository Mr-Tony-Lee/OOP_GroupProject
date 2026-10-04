import gymnasium as gym
import pygame
import sys

def get_intended_pos(row, col, action):
    if action == 0: # Left
        col = max(0, col - 1)
    elif action == 1: # Down
        row = min(7, row + 1)
    elif action == 2: # Right
        col = min(7, col + 1)
    elif action == 3: # Up
        row = max(0, row - 1)
    return row, col

def main():
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human')
    
    state, _ = env.reset()
    env.render()

    print("="*45)
    print("🧊 FrozenLake 8x8 手動挑戰 (V3 - 打滑提示版) 🧊")
    print("請點擊「遊戲視窗」獲得焦點")
    print("使用方向鍵 (↑↓←→) 或 (WASD) 移動")
    print("判斷標準：如果實際位置與指令方向不符，即為打滑")
    print("="*45)

    action_map = {
        pygame.K_LEFT: 0,  pygame.K_a: 0,
        pygame.K_DOWN: 1,  pygame.K_s: 1,
        pygame.K_RIGHT: 2, pygame.K_d: 2,
        pygame.K_UP: 3,    pygame.K_w: 3
    }
    
    action_names = {0: "⬅️ 左", 1: "⬇️ 下", 2: "➡️ 右", 3: "⬆️ 上"}

    running = True
    step_count = 0
    game_over = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                if game_over:
                    if event.key == pygame.K_r:
                        state, _ = env.reset()
                        step_count = 0
                        game_over = False
                        print("\n" + "="*20 + " 🔄 遊戲重置 " + "="*20)
                    continue

                if event.key in action_map:
                    action = action_map[event.key]
                    
                    # 1. 取得當前位置
                    curr_r, curr_c = divmod(state, 8)
                    
                    # 2. 計算「理論上」應該去的位置 (Intended Position)
                    intended_r, intended_c = get_intended_pos(curr_r, curr_c, action)
                    
                    # 3. 執行動作 (實際發生什麼事)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    step_count += 1
                    
                    # 4. 取得實際新位置
                    actual_r, actual_c = divmod(new_state, 8)
                    
                    # 5. 判斷是否打滑
                    # 如果「實際位置」不等於「預期位置」，就是滑走了
                    # 注意：如果你往牆壁走，預期是原地不動，實際也是原地不動，這算「沒滑」
                    is_slip = (actual_r, actual_c) != (intended_r, intended_c)
                    
                    slip_msg = "⚠️ 打滑! (Slip)" if is_slip else "✅ 穩住 (Stable)"

                    # 格式化輸出
                    print(f"步數:{step_count:3} | 指令:{action_names[action]} | {slip_msg} | 位置:({curr_r},{curr_c})->({actual_r},{actual_c})")

                    if terminated:
                        game_over = True
                        if reward == 1.0:
                            print(f"\n🏆 恭喜抵達終點！總步數: {step_count}")
                        else:
                            print(f"\n💀 掉進洞裡了... (按 R 重玩)")
                    
                    if truncated:
                        game_over = True
                        print(f"\n⏳ 超過步數限制 (按 R 重玩)")

                    state = new_state

        pygame.time.Clock().tick(30)

    env.close()

if __name__ == "__main__":
    main()