import torch.nn as nn
import torch
from agent import Agent 

class SACPolicyNetwork(nn.Module):
    def __init__(self):
        super(SACPolicyNetwork,self).__init__()
        self._input_dim = 100 #change these values later
        self.ann = nn.Sequential(
            nn.Linear(self._embeddings_dim, self._mlp_input_dim),
            nn.ReLU(),
            nn.Linear(self._mlp_input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,input:torch.Tensor)->torch.Tensor:
        x = self.ann(input)
        return x


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