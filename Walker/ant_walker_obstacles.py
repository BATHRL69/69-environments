import gymnasium as gym
from stable_baselines3 import SAC, PPO
import cv2
from update_xml import update_env_xml
import torch

custom_xml_path = update_env_xml()
env = gym.make(
    "Ant-v4",
    render_mode="rgb_array",
    xml_file=custom_xml_path,
)


num_timesteps = 500_000

model = SAC("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=num_timesteps)
model.save("ppo_inverted_double_pendulum")


obs, info = env.reset()

for i in range(num_timesteps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunacted, info = env.step(action)
    img = env.render()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if done:
        obs, info = env.reset()
