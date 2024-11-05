# This file serves the purpose of being a general training script
# One should simply run in the terminal a command as follows:
# ~python -m MODEL_TYPE -e ENV_TYPE -t TIMESTEPS -v VERBOSITY -sf SAVE_FILEPATH
# e.g. python -m SAC -e Humanoid-v4 -t 250000 -v 1 -sf 69-environments/Walker/
### This is designed to be adjusted later for our own algorithms

### Need to add logging still

import os
import argparse
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
import gymnasium as gym

def train(env:str,model_type:str,timesteps:int,verbosity:int,save_path:os.PathLike):

    model_path = os.path.join(save_path, "models")
    log_path = os.path.join(save_path, "logs")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    env = gym.make(env,render_mode=None)

    # Note: this will have to get changed later to a parser for our own models
    try:
        model_class = getattr(sb3, model_type)
    except AttributeError:
        raise ValueError(f"Model type '{model_type}' is not available in stable_baselines3.")

    model = model_class("MlpPolicy", env, verbose=verbosity, tensorboard_log=log_path)
    # rawr

    # Will implement the rest of the below later
    # save_frequency = 10000
    # iteration = 0
    # while(iteration*save_frequency<timesteps):
    #     model.learn(total_timesteps=save_frequency)
    #

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_path,name_prefix="model")
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10,min_evals=100,verbose=verbosity)
    eval_callback = EvalCallback(env,best_model_save_path=model_path,log_path=log_path,eval_freq=5000,deterministic=True,render=False,callback_after_eval=stop_callback)

    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(model_path, f"{model_type}_final"))


def main():
    parser = argparse.ArgumentParser(description="Model training script")

    # set default as ppo because faster to run for debugging
    parser.add_argument("-m", "--model_type", type=str,default="PPO", required=False, help="Model type e.g. PPO")
    parser.add_argument("-e", "--env_name", type=str,default="Humanoid-v4", required=False, help="Gymnasium environment name")
    parser.add_argument("-t", "--timesteps", type=int,default=30000, required=False, help="Number of training timesteps")
    parser.add_argument("-v", "--verbosity", type=int,default=0, required=False, help="Verbosity level")
    parser.add_argument("-sf", "--save_path", type=str,default="", required=False, help="Directory to save models and logs")

    args = parser.parse_args()
    train(args.env_name, args.model_type, args.timesteps, args.verbosity, args.save_path)

if __name__ == "__main__":
    main()