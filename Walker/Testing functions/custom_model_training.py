# Setup
# git clone https://github.com/google-deepmind/mujoco_menagerie.git#
# Updated xml_file path to your directoy
# pip install -r from custom_requirements.txt
# run


import gymnasium
import cv2

env = gymnasium.make(
    "Ant-v5",
    xml_file="C:/Users/fabie/OneDrive - University of Bath/Desktop/mujoco_menagerie/unitree_go1/scene.xml",
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode="rgb_array",
)


def policy(state):
    # Dummy policy: replace with your actual policy
    return env.action_space.sample()


# Training loop
num_episodes = 1000
print("started")
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

state = env.reset()
for _ in range(1000):
    action = policy(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    cv2.imshow("Environment", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if terminated or truncated:
        state = env.reset()

cv2.destroyAllWindows()

# Save frames as GIF
