import gymnasium as gym
from sarsa import SarsaAgent, NStepSarsaAgent
import cv2

env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")

model = NStepSarsaAgent(env)
model.train(num_episodes=10000)

obs, info = env.reset()

for i in range(10000):
    action = model.predict(obs)
    obs, reward, done, trunacted, info = env.step(action)
    img = env.render()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Double Inverted Pendulum", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if done:
        obs, info = env.reset()