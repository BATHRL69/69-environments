import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import torch


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

    def render(self, num_timesteps=10_000):
        """Renders the environment using cv2 for n timesteps, and plots a graph based on reward list stored in model

        Args:
            num_timesteps (int, optional): Number of timesteps to render for. Defaults to 10_000.
        """
        state, _info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        for _ in range(num_timesteps):
            action = self.predict(state)
            new_state, _reward, is_finished, is_truncated, _info = self.env.step(action)
            img = self.env.render()
            state = torch.tensor(new_state, dtype=torch.float32)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("", img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            cv2.waitKey(1)  # Add a small delay after rendering each frame

            if is_finished or is_truncated:
                state, _info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32)
                cv2.waitKey(100)

        # plt.figure()
        # plt.title("Training Reward Curve")
        # plt.xlabel("Timesteps")
        # plt.ylabel("Reward")
        # plt.plot(self.timestep_list, self.reward_list, color="green")
        # plt.show()
