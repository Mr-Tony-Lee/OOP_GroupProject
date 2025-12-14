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

def get_time_bucket(step, max_steps, num_buckets):
    bucket_size = max_steps / num_buckets
    bucket = int(step / bucket_size)
    return min(bucket, num_buckets - 1)

def run(episodes, is_training=True, render=False):
    max_steps = 100

    para_dict = {
        'lr': 0.15,
        'df': 0.995,
        'decay': 1 / (episodes * 0.7),
        'min_eps': 0.005, 
        'lr_stable': 0.05,
        'init_reward': -1,
        'hole_penalty': -5,
        'goal_reward': 10000000,
        'num_buckets': 3
    }

    num_buckets = para_dict['num_buckets'] 
    
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, 
                   render_mode='human' if render else None, 
                   max_episode_steps=max_steps)

    if(is_training):
        q = np.zeros((env.observation_space.n, num_buckets, env.action_space.n)) 
    else:
        f = open('frozen_lake8x8_optimized.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    para_dict = {
        'lr': 0.15,
        'df': 0.995,
        'decay': 1 / (episodes * 0.7),
        'min_eps': 0.005, 
        'lr_stable': 0.05,
        'init_reward': -1,
        'hole_penalty': -5,
        'goal_reward': 10000000
    }

    learning_rate_a = para_dict['lr']
    discount_factor_g = para_dict['df']
    epsilon = 1
    epsilon_decay_rate = para_dict['decay']
    rng = np.random.default_rng()
    min_epsilon = para_dict['min_eps']

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        step_count = 0
        
        time_bucket = get_time_bucket(step_count, max_steps, num_buckets)

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, time_bucket, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            custom_reward = para_dict['init_reward']
            if terminated and reward == 0:
                custom_reward -= para_dict['hole_penalty']
            
            if reward == 1:
                custom_reward += para_dict['goal_reward']

            new_step_count = step_count + 1
            new_time_bucket = get_time_bucket(new_step_count, max_steps, num_buckets)

            if is_training:
                q[state, time_bucket, action] = q[state, time_bucket, action] + learning_rate_a * (
                    custom_reward + discount_factor_g * np.max(q[new_state, new_time_bucket, :]) - q[state, time_bucket, action]
                )

            state = new_state
            step_count = new_step_count
            time_bucket = new_time_bucket

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        if(epsilon <= min_epsilon):
            learning_rate_a = para_dict['lr_stable'] 

        if reward > 0: 
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8_optimized.png') # 存成新圖檔
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))

    if is_training:
        f = open("frozen_lake8x8_optimized.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(15000, is_training=True, render=False)
    run(100, is_training=False, render=False)