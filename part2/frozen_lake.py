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

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    para_dict = {'lr': 0.14348124890515385, 'df': 0.9859576385195953, 'decay': 7.46045707566907e-05, 'min_eps': 4.501175498793086e-05, 'lr_stable': 0.11276572044653185}

    learning_rate_a = para_dict['lr'] # alpha or learning rate
    discount_factor_g = para_dict['df'] # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = para_dict['decay']    # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator
    min_epsilon = para_dict['min_eps']  # minimum epsilon

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 100

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

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        # if(epsilon==0):
        #     learning_rate_a = 0.0001

        if(epsilon <= min_epsilon):
            learning_rate_a = para_dict['lr_stable'] 

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
    for i in range(10):
        run(15000, is_training=True, render=False)
        run(10000, is_training=False, render=False)