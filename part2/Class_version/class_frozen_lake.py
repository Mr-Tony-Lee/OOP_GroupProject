import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class FrozenLakeAgent:
    def __init__(self, map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode = "ansi"):
        # -------- env generation parameters --------
        self.map_name = map_name
        self.is_training = is_training
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.render = False


        self.rng = np.random.default_rng()   # random number generator
        
        # -------- training parameters --------
        self.episodes = episodes
        self.is_training = is_training
        self.q = None   # Q-table

        self.learning_rate_a = 0.5 # alpha or learning rate
        self.learning_rate_decay = 0.9995 # Adjusted for smooth decay
        self.min_learning_rate = 0.0001

        self.discount_factor_g = 0.99 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
        
        self.exploration_rate = 1         # 1 = 100% random actions
        self.exploration_decay_rate = 0.00015    # epsilon decay rate. 1/0.0001 = 10,000
        self.min_exploration_rate = 0.001
    
    def init_q_table(self, env):
        """Initialize the Q-table."""
        if(self.is_training):
            # Optimistic Initialization: Initialize with small positive values to encourage exploration
            # self.q = np.zeros((env.observation_space.n, env.action_space.n)) 
            self.q = np.random.uniform(low=0.0, high=0.001, size=(env.observation_space.n, env.action_space.n))
        else:
            f = open(f'frozen_lake{self.map_name}.pkl', 'rb')
            self.q = pickle.load(f)
            f.close()

    def print_success_rate(self, rewards_per_episode):
        """Calculate and print the success rate of the agent."""
        total_episodes = len(rewards_per_episode)
        success_count = np.sum(rewards_per_episode)
        success_rate = (success_count / total_episodes) * 100
        print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
        return success_rate

    def plot_rewards(self, rewards_per_episode):
        """Plot the sum of rewards over episodes."""
        plt.figure()  # 建立一個新的圖表，避免與上一張圖重疊
        
        sum_rewards = np.zeros(self.episodes)
        for t in range(self.episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        

        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards (Last 100 Episodes)') 
        plt.title('Frozen Lake Rewards over Episodes')
        
        if self.is_training == True :
            plt.savefig(f'frozen_lake_Training{self.map_name}.png')
        else:
            plt.savefig(f'frozen_lake_Evaluation{self.map_name}.png')
            
        plt.close() # 存檔後關閉圖表，釋放記憶體

    def run(self):
        # env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
        # env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='ansi' if render else None)
        env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)

        self.init_q_table(env)

        rewards_per_episode = np.zeros(self.episodes)

        for i in range(self.episodes):
            state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
            terminated = False      # True when fall in hole or reached goal
            truncated = False       # True when actions > 200

            while(not terminated and not truncated):
                if self.is_training and self.rng.random() < self.exploration_rate:
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    action = np.argmax(self.q[state,:])

                new_state, reward, terminated, truncated,_ = env.step(action)

                if self.is_training:
                    # 利用轉移機率計算總期望值 ( 算小作弊? )
                    expected_target = 0
                    transitions = env.unwrapped.P[state][action]
                    
                    for prob, next_s, r, term in transitions:
                        target = r
                        if not term:
                            target += self.discount_factor_g * np.max(self.q[next_s, :])
                        expected_target += prob * target

                    self.q[state,action] = self.q[state,action] + self.learning_rate_a * (
                        expected_target - self.q[state,action]
                    )

                state = new_state

            self.exploration_rate = max(self.exploration_rate - self.exploration_decay_rate, self.min_exploration_rate)
            
            # Smooth learning rate decay
            self.learning_rate_a = max(self.learning_rate_a * self.learning_rate_decay, self.min_learning_rate)

            if reward == 1:
                rewards_per_episode[i] = 1

        env.close()
        
        self.plot_rewards(rewards_per_episode)
        
        if self.is_training == False:
            print(self.print_success_rate(rewards_per_episode))

        if self.is_training:
            f = open(f"frozen_lake{self.map_name}.pkl","wb")
            pickle.dump(self.q, f)
            f.close()

if __name__ == '__main__':
    
    # Train
    print("Training...")
    agent = FrozenLakeAgent(map_name="8x8", episodes=15000, is_training=True, is_slippery=True, render_mode="ansi")
    agent.run()
    
    # Evaluate
    print("\nEvaluating...")
    agent = FrozenLakeAgent(map_name="8x8", episodes=1000, is_training=False, is_slippery=True, render_mode="ansi")
    agent.run()