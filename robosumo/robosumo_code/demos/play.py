import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
import click
import robosumo
import robosumo.envs
import shimmy.openai_gym_compatibility




# i don't know how imports work so ive just put this in here for now
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from robosumo.policy_zoo import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, set_from_flat

POLICY_FUNC = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
}

@click.command()
@click.option("--env", type=str,
              default="RoboSumo-Ant-vs-Bug-v0", show_default=True,
              help="Name of the environment.")
@click.option("--policy-names", nargs=2, type=click.Choice(["mlp", "lstm"]),
              default=("mlp", "mlp"), show_default=True,
              help="Policy names.")
@click.option("--param-versions", nargs=2, type=int,
              default=(1, 1), show_default=True,
              help="Policy parameter versions.")
@click.option("--max_episodes", type=int,
              default=20, show_default=True,
              help="Number of episodes.")

def main(env, policy_names, param_versions, max_episodes):
    # Construct paths to parameters
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    params_dir = os.path.join(curr_dir, "../robosumo/policy_zoo/assets")
    agent_names = [env.split('-')[1].lower(), env.split('-')[3].lower()]
    param_paths = []
    for a, p, v in zip(agent_names, policy_names, param_versions):
        param_paths.append(
            os.path.join(params_dir, a, p, "agent-params-v%d.npy" % v)
        )

    # Create environment
    print(env)
    env = gym.make("GymV21Environment-v0", env_id=env)

    for agent in env.agents:
        agent._adjust_z = -0.5

    # TensorFlow 2.x session setup
    tf.config.set_soft_device_placement(True)
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Initialize policies
    policy = []
    for i, name in enumerate(policy_names):
        scope = "policy" + str(i)
        policy.append(
            POLICY_FUNC[name](scope=scope, reuse=False,
                              ob_space=env.observation_space.spaces[i],
                              ac_space=env.action_space.spaces[i],
                              hiddens=[64, 64], normalize=True)
        )
    tf.keras.backend.set_session(tf.compat.v1.Session())
    sess = tf.compat.v1.Session()
    sess.__enter__()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load policy parameters
    params = [load_params(path) for path in param_paths]
    for i in range(len(policy)):
        set_from_flat(policy[i].get_variables(), params[i])

    # Play matches between the agents
    num_episodes, nstep = 0, 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    observation, _ = env.reset()
    print("-" * 5 + "Episode %d " % (num_episodes + 1) + "-" * 5)
    while num_episodes < max_episodes:
        env.render()
        action = tuple([
            pi.act(stochastic=True, observation=observation[i])[0]
            for i, pi in enumerate(policy)
        ])
        observation, reward, done, _, infos = env.step(action)

        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        if done[0]:
            num_episodes += 1
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}"
                          .format(i, total_scores, num_episodes))
            if draw:
                print("Match tied: Agent {}, Scores: {}, Total Episodes: {}"
                      .format(i, total_scores, num_episodes))
            observation, _ = env.reset()
            nstep = 0
            total_reward = [0.0  for _ in range(len(policy))]

            for i in range(len(policy)):
                policy[i].reset()

            if num_episodes < max_episodes:
                print("-" * 5 + "Episode %d " % (num_episodes + 1) + "-" * 5)

if __name__ == "__main__":
    print('hello')


    main()
