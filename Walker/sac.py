import torch.nn as nn

from agent import Agent


class SACPolicyNetwork(nn.module):
    def __init__(self):
        pass


class SACValueNetwork(nn.module):
    def __init__(self):
        pass


class SACAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.actor = SACPolicyNetwork()
        self.critic_1 = SACValueNetwork()
        self.critic_2 = SACValueNetwork()
        self.critic_target_1 = SACValueNetwork()
        self.critic_target_2 = SACValueNetwork()
    
    def simulate_episode(self):
        """Run a single training episode."""
        raise NotImplementedError

    def predict(self, state):
        """Predict the best action for the current state."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the agent's data to the path specified."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the data from the path specified."""
        raise NotImplementedError