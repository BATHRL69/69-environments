class Agent:
    def simulate_episode(self):
        """Run a single training episode."""
        raise NotImplementedError

    def train(self, num_episodes=1000):
        """Train the agent over a given number of episodes."""
        ten_percent = min(1, int(num_episodes / 10)) # in case num_episodes < 10 and we try to divide by 0

        for i in range(num_episodes):
            self.simulate_episode()

            if i % ten_percent == 0:
                print(f"Training {10 * i / ten_percent}% complete...")

    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError
