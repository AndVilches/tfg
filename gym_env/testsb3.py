import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import my_robotenv



def make_env(rank):
    def _init():
        env = gym.make("my_robotenv-v1")
        env.reset(seed=rank)
        return env
    return _init

def train_dqn():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Vectorización con 32 entornos
    env = SubprocVecEnv([make_env(i) for i in range(8)])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda"
    )

    TIMESTEPS = 100000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/ppo_{TIMESTEPS*iters}")


def test_dqn():
    model_dir = "models"
    env = gym.make('my_robotenv-v1', render_mode="human")  # usa render si tu entorno lo soporta

    # Buscar el último modelo guardado
    model_files = sorted([f for f in os.listdir(model_dir) if f.startswith("ppo_")])
    if not model_files:
        print("No se encontraron modelos entrenados.")
        return
    last_model_path = os.path.join(model_dir, "ppo_3000000.zip")
    print(f"Cargando modelo: {last_model_path}")

    model = PPO.load(last_model_path, env=env, device="cuda")

    obs, _ = env.reset()
    done = False
    total_reward = 0
    # action = [1,1]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
    
        print("Accion tomada: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if hasattr(env, "render"):
            env.render()

    print(f"Recompensa total del episodio: {total_reward}")
    env.close()

if __name__=="__main__":
    # train_dqn()
    test_dqn()