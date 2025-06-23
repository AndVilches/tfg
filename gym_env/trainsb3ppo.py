import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import os
import my_robotenv
from stable_baselines3.common.callbacks import BaseCallback

NUM_ENVS = 8

class TrainAndLogCallback(BaseCallback):
    def __init__(self, check_freq, save_path, start_steps=0, verbose=1):
        super(TrainAndLogCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_steps = start_steps

    def _init_callback(self):
        # Creamos la carpeta de guardado si no existe.
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Guardamos el modelo cada check_freq pasos.
        if self.n_calls % self.check_freq == 0:
            if self.save_path is not None:
                self.model.save(os.path.join(self.save_path, "model_{}".format(self.n_calls + int(self.start_steps))))
        return True

#TODO: (Opcional) Implementad otros callbacks si lo considerais necesario.

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

# def make_env(rank):
#     def _init():
#         env = gym.make("my_robotenv-v1")
#         env.reset(seed=rank)
#         return env
#     return _init

def train_dqn():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Vectorización con 32 entornos
    env = make_vec_env(
        "my_robotenv-v1",
        n_envs=NUM_ENVS,
        vec_env_cls=DummyVecEnv, # Puede ser DummyVecEnv o SubprocVecEnv. Cambiadlo en base a vuestra CPU y analisis de rendimiento.
        monitor_dir="./monitor_dir/",
        seed=33,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        learning_rate=linear_schedule(3e-4),
        batch_size=128,
        n_steps= 2048,
        n_epochs=5
    )

    total_timesteps = 1e7
    iters = 0
    cb = TrainAndLogCallback(check_freq=10000, save_path="./mario_models/", start_steps=0, verbose=1)

    iters += 1
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(f"{model_dir}/ppo_1")


def test_dqn():
    model_dir = "models"
    env = gym.make('my_robotenv-v1', render_mode="human")  # usa render si tu entorno lo soporta

    # Buscar el último modelo guardado
    model_files = sorted([f for f in os.listdir(model_dir) if f.startswith("ppo_")])
    if not model_files:
        print("No se encontraron modelos entrenados.")
        return
    last_model_path = os.path.join(model_dir, "ppo_2500000.zip")
    print(f"Cargando modelo: {last_model_path}")

    model = PPO.load(last_model_path, env=env, device="cuda")

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        print("Accion tomada: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated
        total_reward += reward
        if hasattr(env, "render"):
            env.render()

    print(f"Recompensa total del episodio: {total_reward}")
    env.close()

if __name__=="__main__":
    train_dqn()
    # test_dqn()

'''
Nota con el problema que me pasa, después de todos los timesteps parado hace esto

-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 2.17e+03    |
|    ep_rew_mean          | 1.18e+03    |
| time/                   |             |
|    fps                  | 1607        |
|    iterations           | 405         |
|    time_elapsed         | 4126        |
|    total_timesteps      | 6635520     |
| train/                  |             |
|    approx_kl            | 0.013951356 |
|    clip_fraction        | 0.179       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.94        |
|    explained_variance   | 0.956       |
|    learning_rate        | 0.000101    |
|    loss                 | 0.0948      |
|    n_updates            | 2020        |
|    policy_gradient_loss | 0.00848     |
|    std                  | 0.0914      |
|    value_loss           | 0.157       |
-----------------------------------------
GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAL
[  0.05231452  -0.12063797 -30.76319193]
[  0.          0.        -29.8484979]
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 7.42e+03    |
|    ep_rew_mean          | 4.43e+03    |
| time/                   |             |
|    fps                  | 1608        |
|    iterations           | 406         |
|    time_elapsed         | 4136        |
|    total_timesteps      | 6651904     |
| train/                  |             |
|    approx_kl            | 0.023244567 |
|    clip_fraction        | 0.185       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.95        |
|    explained_variance   | 0.952       |
|    learning_rate        | 0.000101    |
|    loss                 | 0.0884      |
|    n_updates            | 2025        |
|    policy_gradient_loss | 0.00777     |
|    std                  | 0.0911      |
|    value_loss           | 0.173       |
-----------------------------------------

'''