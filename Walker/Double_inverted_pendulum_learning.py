import gymnasium as gym
from stable_baselines3 import PPO
import cv2

env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=300_000)
model.save("ppo_inverted_double_pendulum")

obs, info = env.reset()

for i in range(10_000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, trunacted ,info = env.step(action)
    img = env.render()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Double Inverted Pendulum", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if done:
        obs, info = env.reset()