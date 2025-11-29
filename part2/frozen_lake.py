import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 123 
def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False, decay_rate = 0.5, min_exploration = 0, seed = 42 ):

    # env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    # env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='ansi' if render else None)
    env = gym.make("FrozenLake-v1", render_mode="ansi")

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.8 # alpha or learning rate
    learning_rate_decay = 0.99
    min_learning_rate = 0.001
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 1/(episodes*decay_rate)     # epsilon decay rate. 1/0.0001 = 10,000
    min_exploration_rate = min_exploration
    rng = np.random.default_rng(seed)   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset(seed = seed+i)[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        if(epsilon==min_exploration_rate):
            # learning_rate_a = 0.0001
            learning_rate_a = max(learning_rate_a*learning_rate_decay, min_learning_rate )

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    rate = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    final_rate = 0.65
    min_explo = [0,0.025,0.05,0.75,0.10]
    final_explo = 0.025
    # for each_rate in rate:
    #     for each_min_explo in min_explo:
    #         print(f"Now rate : {each_rate}, Now min explo : {each_min_explo}")
    #         run(15000, is_training=True, render=False, decay_rate = each_rate, min_exploration = each_min_explo)
    #         run(1000, is_training=False, render=True, decay_rate = each_rate, min_exploration = each_min_explo)
    run(15000, is_training=True, render=False, decay_rate = final_rate, min_exploration = final_explo)
    run(1000, is_training=False, render=True, decay_rate = final_rate, min_exploration = final_explo)
