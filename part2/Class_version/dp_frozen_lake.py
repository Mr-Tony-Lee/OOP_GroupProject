import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class FrozenLakeDP:
    def __init__(self, map_name="8x8", is_slippery=True, render_mode="ansi"):
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.gamma = 0.99  # Discount factor
        self.theta = 1e-9  # Convergence threshold
        self.policy = None
        self.V = None

    def value_iteration(self, env):
        """
        Perform Value Iteration to find the optimal policy.
        """
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.V = np.zeros(n_states)
        
        print("Starting Value Iteration...")
        iteration = 0
        while True:
            delta = 0
            for s in range(n_states):
                v = self.V[s]
                # Calculate the value for each action
                action_values = np.zeros(n_actions)
                for a in range(n_actions):
                    for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                        target = reward
                        if not terminated:
                            target += self.gamma * self.V[next_s]
                        action_values[a] += prob * target
                
                # Update the value of the state to the maximum action value
                self.V[s] = np.max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            iteration += 1
            if delta < self.theta:
                print(f"Value Iteration converged in {iteration} iterations.")
                break
        
        # Extract the optimal policy
        self.policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_s, reward, terminated in env.unwrapped.P[s][a]:
                    target = reward
                    if not terminated:
                        target += self.gamma * self.V[next_s]
                    action_values[a] += prob * target
            self.policy[s] = np.argmax(action_values)

    def print_success_rate(self, rewards_per_episode):
        """Calculate and print the success rate of the agent."""
        total_episodes = len(rewards_per_episode)
        success_count = np.sum(rewards_per_episode)
        success_rate = (success_count / total_episodes) * 100
        print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
        return success_rate

    def run_evaluation(self, episodes=1000):
        env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery, render_mode=self.render_mode)
        
        # Solve the MDP first
        self.value_iteration(env)
        
        print(f"\nEvaluating DP Policy for {episodes} episodes...")
        rewards_per_episode = np.zeros(episodes)
        
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                action = self.policy[state]
                new_state, reward, terminated, truncated, _ = env.step(action)
                state = new_state
                
                if reward == 1:
                    rewards_per_episode[i] = 1
        
        env.close()
        self.print_success_rate(rewards_per_episode)

if __name__ == '__main__':
    agent_4 = FrozenLakeDP(map_name="4x4", is_slippery=True)
    agent_4.run_evaluation(episodes=10000)

    agent_8 = FrozenLakeDP(map_name="8x8", is_slippery=True)
    agent_8.run_evaluation(episodes=10000)