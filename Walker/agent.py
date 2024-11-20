import matplotlib.pyplot as plt

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
        critic_loss_list = []
        actor_loss_list = []

        while timesteps < num_timesteps:
            elapsed_timesteps, reward, critic_loss_history, actor_loss_history = self.simulate_episode()
            timesteps += elapsed_timesteps
            reward_total += reward
            episodes += 1


            if (episodes % print_interval == 0):
                #print(f"Training {100 * timesteps / num_timesteps:.2f}% complete...")
                print(f"Average reward of {reward_total / print_interval} received")
                reward_list.append(reward_total/print_interval)
                critic_loss_list.append(critic_loss_history[-1])
                actor_loss_list.append(actor_loss_history[-1])
                reward_total = 0
        plt.figure()
        plt.plot(reward_list, color="green")
        plt.plot(critic_loss_list, color="blue")
        plt.plot(actor_loss_list, color="red")
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
