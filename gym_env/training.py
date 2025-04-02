import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import DQN
import os
import my_robotenv



def train_dqn():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('my_robotenv-v0')

    model = DQN(
        policy="MlpPolicy",  # Política de red neuronal
        env=env,  # Entorno de Gym
        verbose=1,
        tensorboard_log=log_dir,  # Carpeta para logs de TensorBoard
        device="cuda"  # Aceleración con GPU si está disponible
    )
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/dqn_{TIMESTEPS*iters}")


if __name__=="__main__":
    train_dqn()
    # test_dqn()