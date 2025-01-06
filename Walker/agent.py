import gymnasium as gym

class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.reward_list = []
        self.timestep_list = []

    def simulate_episode(self):
        """Run a single training episode."""
        raise NotImplementedError

    def train(self, num_timesteps=100000, start_timesteps=0):
        """Train the agent over a given number of episodes."""
        timesteps = start_timesteps
        episodes = 0

        while timesteps < num_timesteps:
            elapsed_timesteps, reward = self.simulate_episode()
            timesteps += elapsed_timesteps
            episodes += 1

            self.reward_list.append(reward)
            self.timestep_list.append(timesteps)

            print(
                f"[Episode {episodes} / timestep {timesteps}] Received reward {reward:.3f}"
            )

    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError

    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError

    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError
    