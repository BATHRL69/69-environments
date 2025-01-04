## Agent implementations

### Recreating results

1. Requirements can be found in `requirements.txt`
2. In order to generate results for each agent, uncomment the required agent, and run `get_results.py`. Note, we generated results over three runs for each agent, which would take days. In order to reduce time our get_results runs agents once.
3. These will save results in a format that can be plotted.

### Third party sources

All of the RL agents were implemented "from scratch". However, our environment was implemented using third party sources, as detailed below:

| Library | What we used it for | Modifications Made |
| ------- | ------------------- | ------------------ |
| [Gymnasium / Gym](https://github.com/Farama-Foundation/Gymnasium) | The interface for training our agents. For instance implementing the rewards function, and state and action spaces used by our agents | None |
| [Mujoco / Ant-v4](https://github.com/openai/mujoco-py) | Environment ant was implemented in, for instance implementing transition dynamics | None |
