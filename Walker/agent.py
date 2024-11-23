import cv2
import torch


class Agent:
    def simulate_episode(self):
        """Run a single training episode."""
        raise NotImplementedError

    def train(self, num_timesteps=100000, print_interval=50):
        """Train the agent over a given number of episodes."""
        timesteps = 0
        episodes = 0
        reward_total = 0
        reward_list = []

        while timesteps < num_timesteps:
            elapsed_timesteps, reward = self.simulate_episode()
            timesteps += elapsed_timesteps
            reward_total += reward
            episodes += 1

            print(f"[EPISODE {episodes}] Reward of {reward} received")

            if (episodes % print_interval == 0):
                print(f"Training {100 * timesteps / num_timesteps:.2f}% complete...")
                reward_list.append(reward_total/print_interval)
                reward_total = 0

        plt.figure()
        plt.plot(reward_list, color="green")
        plt.show()


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
        state, _info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        for _ in range(num_timesteps):
            action = self.predict(state)
            new_state, _reward, is_finished, _is_truncated, _info = self.env.step(
                action
            )
            img = self.env.render()
            state = torch.tensor(new_state, dtype=torch.float32)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if is_finished:
                state, _info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32)
