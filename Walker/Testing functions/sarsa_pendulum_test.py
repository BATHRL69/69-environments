import cv2
import gymnasium as gym
import os
import sys

# i'm sure there's a better way to do an import from the parent folder but i don't know what it is
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from sarsa import SarsaAgent, NStepSarsaAgent

SAVE_PATH = "Walker/Testing functions/sarsa_pendulum.data"

env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")

model = NStepSarsaAgent(env)
model.load(SAVE_PATH)
model.train(num_episodes=500)
model.save(SAVE_PATH)

obs, info = env.reset()

for i in range(10000):
    action = model.predict(obs)
    obs, reward, done, trunacted, info = env.step(action)
    img = env.render()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Inverted Pendulum", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if done:
        obs, info = env.reset()
