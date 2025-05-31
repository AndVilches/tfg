import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import my_robotenv  # Asegúrate de que importa tu entorno personalizado

# Ruta al modelo entrenado
MODEL_PATH = "mario_models/model_110000.zip"

# Crear el entorno
def make_env():
    env = gym.make("my_robotenv-v1", render_mode="human")  # Asegúrate de que tienes render_mode si lo usas
    return env

# Cargar el modelo
model = PPO.load(MODEL_PATH)

# Envolver en DummyVecEnv si es necesario
env = DummyVecEnv([make_env])

# Número de episodios de test
num_episodes = 10

for ep in range(num_episodes):
    obs = env.reset()  # reset() devuelve una tupla (obs, info)
    done = False
    total_reward = 0
    trajectory = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = env.step(action)
        print(terminated)
        done = terminated
        total_reward += reward
        if hasattr(env, "render"):
            env.render()
    
    print("SALE DEL BUCLE")
        # trajectory.append(obs[0][:2])  # Guardar posición x, y para graficar (ajusta si quieres más)

    # print(f"Episodio {ep+1} - Recompensa total: {total_reward}")

    # Opcional: graficar la trayectoria del robot
    # trajectory = np.array(trajectory)
    # plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    # plt.title(f"Trayectoria episodio {ep+1}")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

env.close()
