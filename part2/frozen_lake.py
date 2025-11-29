import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, map_name="8x8", is_training=True, render=False):

    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True, render_mode="ansi")

    if(is_training):
        # Optimistic Initialization: Initialize with small positive values to encourage exploration
        q = np.random.uniform(low=0.0, high=0.001, size=(env.observation_space.n, env.action_space.n))
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.5 # alpha or learning rate
    learning_rate_decay = 0.9995 # Adjusted for smooth decay
    min_learning_rate = 0.0001

    discount_factor_g = 0.99 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    
    exploration_rate = 1         # 1 = 100% random actions
    exploration_decay_rate = 0.0002    # epsilon decay rate. 1/0.0001 = 10,000
    min_exploration_rate = 0.001

    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < exploration_rate:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                target = reward
                if not terminated:
                    target += discount_factor_g * np.max(q[new_state,:])
                
                q[state,action] = q[state,action] + learning_rate_a * (
                    target - q[state,action]
                )

            state = new_state

        exploration_rate = max(exploration_rate - exploration_decay_rate, min_exploration_rate)
        
        # Smooth learning rate decay
        learning_rate_a = max(learning_rate_a * learning_rate_decay, min_learning_rate)

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    plt.figure()
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards (Last 100 Episodes)') 
    plt.title('Frozen Lake Rewards over Episodes')

    if is_training:
        plt.savefig(f'frozen_lake_Training{map_name}.png')
    else:
        plt.savefig(f'frozen_lake_Evaluation{map_name}.png')
    plt.close()
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))

    if is_training:
        f = open(f"frozen_lake{map_name}.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    
    # Train
    print("Training...")
    run(15000, "8x8", is_training=True, render=False)
    
    # Evaluate
    print("\nEvaluating...")
    run(1000, "8x8", is_training=False, render=True)
