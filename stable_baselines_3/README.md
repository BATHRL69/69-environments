Prerequisites:
gymnasium[mujoco]
stable-baselines3[extra]

To train, run the following command (just change the name of the algorithm):
e.g. 
python .\sb3.py Humanoid-v4 SAC -t
python .\sb3.py Humanoid-v4 A2C -t
python .\sb3.py Humanoid-v4 TD3 -t

To monitor training, run the following command and open the local host link in the browser:
tensorboard --logdir ./logs

To test a model (change the number of steps to the number of steps the model was trained for):
e.g.
python .\sb3.py Humanoid-v4 SAC -s ./models/SAC_25000.zip
python .\sb3.py Humanoid-v4 A2C -s ./models/A2C_100000.zip
python .\sb3.py Humanoid-v4 TD3 -s ./models/TD3_25000.zip