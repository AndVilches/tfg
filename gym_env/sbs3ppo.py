import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
from agent import AgentActions

register(
    id='my_robotenv-v0',
    entry_point='my_robotenv:RLContinuousEnv' 
)

class RLContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Definir los límites del espacio de estados
        self.xmin = np.array([-10, -10, -180])
        self.xmax = np.array([10, 10, 180])
        
        # Espacio de observaciones continuo
        self.observation_space = spaces.Box(low=self.xmin, high=self.xmax, dtype=np.float32)
        
       # Límites reales de las acciones
        self.max_velocity = 1.0
        self.max_angle = 60.0
        
        # Espacio de acciones normalizado (-1 a 1)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # Inicializar estado y meta
        self.state = []
        self.goal = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Posición aleatoria del robot
        self.state = self.np_random.uniform(self.xmin, self.xmax).astype(np.float32)

        # self.state = np.clip(self.state, self.xmin, self.xmax)

        # Definir una meta aleatoria dentro del espacio
        self.goal = self.np_random.uniform(self.xmin, self.xmax).astype(np.float32)
        
        if self.render_mode == "human":
            self.render()
        
        # self.render()
        obs = self.state
        return obs, {}

    def step(self, action):
        """ Ejecuta una acción y actualiza el estado. """

        velocity = action[0] * self.max_velocity
        angle = action[1] * self.max_angle

        prev_state = self.state.copy()  # Guardar el estado anterior
        pos_final, tiempo = self._simulate_motion(self.state, velocity, angle)
        
        # Guardar estado previo y actualizar estado
        self.state = pos_final

        # Calcular recompensa basada en distancia y orientación
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        distance_compare = distance_to_goal
        # Vector desde el estado previo hasta el nuevo estado
        movement_vector = self.state[:2] - prev_state[:2]
        
        # Vector desde el nuevo estado hasta la meta
        goal_vector = self.goal[:2] - self.state[:2]
        angle_diff = self.goal[2] - self.state[2]
        # Normalizar vectores
        if np.linalg.norm(movement_vector) > 0:
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
        if np.linalg.norm(goal_vector) > 0:
            goal_vector = goal_vector / np.linalg.norm(goal_vector)

        # Penalización basada en la proyección vectorial (alineación con la meta)
        alignment_penalty = np.dot(movement_vector, goal_vector)
        distance_to_goal = distance_to_goal*0.01
        # Recompensa total
        reward = -distance_to_goal + alignment_penalty * 0.05 - tiempo
        if distance_compare < 0.1 and abs(angle_diff) < 5:
            done = True
            reward = 10000
            print("GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAL")
            print(self.state)
            print(self.goal)
        elif np.any(self.state < self.xmin) or np.any(self.state > self.xmax):
            done = True
            reward = -50
        else:
            done = False
        
        return self.state, reward, done, False, {}

    def _simulate_motion(self, state, velocity, angle):
        """ Simula el movimiento basado en la velocidad y el ángulo de giro. """
        dt = 0.1
        new_state = state.copy()
        
        new_state[0] += dt * velocity * np.cos(np.radians(new_state[2]))
        new_state[1] += dt * velocity * np.sin(np.radians(new_state[2]))
        new_state[2] += angle * dt
        
        # Asegurar que la orientación se mantiene en [-180, 180]
        if new_state[2] > 180:
            new_state[2] -= 360
        elif new_state[2] < -180:
            new_state[2] += 360
        
        return new_state, dt

    def render(self, mode="human"):
        print(f"Estado actual: x={self.state[0]}, y={self.state[1]}, theta={self.state[2]}")
        print(f"Meta: x={self.goal[0]}, y={self.goal[1]}, theta={self.goal[2]}")
    
    
    def get_observations(self):
        return self.observation_space

# '''

# if __name__ == "__main__":
#     env = gym.make('my_robotenv-v0', render_mode='human')
#     print("Check environment begin")
#     check_env(env.unwrapped)
#     print("Check environment end")
#     state, _ = env.reset()
#     done = False
#     step_count = 0
#     env.get_observations()
#     while not done and step_count < 100:
#         action = env.action_space.sample()  # Elegir una acción aleatoria
#         next_state, reward, done, _, _ = env.step(action)
        
#         print(f"Step {step_count}:")
#         print(f"Action: {action}")
#         print(f"State: {next_state}")
#         print(f"Reward: {reward}")
#         print(f"Done: {done}")
#         print("-------------------------")
        
#         step_count += 1
# '''
    
        
#         print(step_count)
#         if step_count > 1 and step_count < 6:
#             if step_count == 2:
#                 print('direccion x+ y+')
#             if step_count == 3:
#                 print('direccion x- y+')
#             if step_count == 4:
#                 print('direccion x+ y-')
#             if step_count == 5:
#                 print('direccion x- y-')
#         elif step_count > 1:
#             if step_count % 5 == 0:
#                 print('direccion x- y-')
#             elif step_count % 4 == 0:
#                 print('direccion x+ y-')
#             elif step_count % 3 == 0:
#                 print('direccion x- y+')
#             if step_count % 2 == 0:
#                 print('direccion x+ y+')


#alvaro.belmonte@ua.es

