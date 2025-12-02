import pickle
import numpy as np
import os

model_path = r"d:\中山大學\三上\物件導向程式設計\GroupProject\OOP_GroupProject\part3\Dungeon_Crawler\QLearningAgent\result\model.pkl"

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        q_table = pickle.load(f)
    
    print(f"Loaded Q-Table with {len(q_table)} states.")
    
    # Check start state (1, 1, 0)
    start_state = (1, 1, 0)
    if start_state in q_table:
        print(f"Q-values for start state {start_state}: {q_table[start_state]}")
        actions = ["LEFT", "DOWN", "RIGHT", "UP"]
        best_action = np.argmax(q_table[start_state])
        print(f"Best action: {actions[best_action]}")
    else:
        print(f"Start state {start_state} not in Q-Table!")
        
    # Check some other states
    print("\nSample states:")
    for i, (state, values) in enumerate(q_table.items()):
        if i >= 5: break
        print(f"State {state}: {values}")
else:
    print("Model file not found.")
