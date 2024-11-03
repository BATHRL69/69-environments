import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import os

default_path = os.path.join(
    os.path.dirname(gym.__file__), "envs", "mujoco", "assets", "ant.xml"
)

with open(default_path, "r") as f:
    xml_string = f.read()

# Add my obstacles
obstacle_xml = """
    <body name="obstacle_1" pos="2 2 0.5">
        <geom name="obs1" type="cylinder" size="0.2 0.5" rgba="1 0 0 1"/>
    </body>
    <body name="obstacle_2" pos="4 4 0.5">
        <geom name="obs2" type="cylinder" size="0.2 0.5" rgba="1 0 0 1"/>
    </body>
    <body name="obstacle_3" pos="6 6 0.5">
        <geom name="obs3" type="cylinder" size="0.2 0.5" rgba="1 0 0 1"/>
    </body>
"""

modified_xml = xml_string.replace("</worldbody>", f"{obstacle_xml}\n</worldbody>")

custom_xml_path = (
    "C:/Users/Solly/_/python/69/Walker/temp_ant.xml"  # UPDATE ME TO YOUR path
)
with open(custom_xml_path, "w") as f:
    f.write(modified_xml)

env = gym.make(
    "Ant-v4",
    render_mode="rgb_array",
    xml_file=custom_xml_path,
)

model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=100_000)
model.save("ppo_inverted_double_pendulum")


obs, info = env.reset()

for i in range(100_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunacted, info = env.step(action)
    img = env.render()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if done:
        obs, info = env.reset()
