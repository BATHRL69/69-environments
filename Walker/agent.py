class Agent:
    def simulate_episode(self):
        """Run a single training episode."""
        raise NotImplementedError

    def train(self, num_timesteps=100000, print_interval=50):
        """Train the agent over a given number of episodes."""
        timesteps = 0
        episodes = 0

        while timesteps < num_timesteps:
            timesteps += self.simulate_episode()
            episodes += 1

            if (episodes % print_interval == 0):
                print(f"Training {100 * timesteps / num_timesteps:.2f}% complete...")

    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError
