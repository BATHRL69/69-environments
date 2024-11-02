---

# Reinforcement Learning Training and Testing with Stable Baselines3

This project provides a framework for training and testing reinforcement learning models using [stable-baselines3](https://stable-baselines3.readthedocs.io/) and [gymnasium](https://gymnasium.farama.org/) with MuJoCo environments.

---

## Prerequisites

Before you begin, make sure to install the required packages:

```bash
pip install "gymnasium[mujoco]" "stable-baselines3[extra]"
```

---

## Training a Model

To train a model, use the following command. Replace `<algorithm>` with your choice of reinforcement learning algorithm (`SAC`, `A2C`, or `TD3`).

```bash
python .\sb3.py Humanoid-v4 <algorithm> -t
```

### Examples

```bash
python .\sb3.py Humanoid-v4 SAC -t
python .\sb3.py Humanoid-v4 A2C -t
python .\sb3.py Humanoid-v4 TD3 -t
```

---

## Monitoring Training Progress

You can monitor the training progress in real-time with TensorBoard. To do this, run the following command and open the provided localhost link in your browser:

```bash
tensorboard --logdir ./logs
```

---

## Testing a Model

To test a trained model, use the following command. Update `<algorithm>` with the algorithm used during training and specify the model’s checkpoint file in place of `<steps>`. Make sure to match the checkpoint with the correct number of training steps.

```bash
python .\sb3.py Humanoid-v4 <algorithm> -s ./models/<algorithm>_<steps>.zip
```

### Examples

```bash
python .\sb3.py Humanoid-v4 SAC -s ./models/SAC_25000.zip
python .\sb3.py Humanoid-v4 A2C -s ./models/A2C_100000.zip
python .\sb3.py Humanoid-v4 TD3 -s ./models/TD3_25000.zip
```

---
