import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
from agent import AgentActions

register(
    id='my_robotenv-v0',
    entry_point='my_robotenv:RLPartitionEnv' 
)
class RLPartitionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # Definición del espacio de estados discreto
        self.nc = np.array([21, 21, 21])  # Número de celdas en cada dimensión
        self.xmin = np.array([-6.15, -6.15, -180])
        self.xmax = np.array([6.15, 6.15, 180])
        self.h = (self.xmax - self.xmin) / self.nc  # Tamaño de celda
        
        # Espacio de observación discreto
        self.observation_space = spaces.Discrete(self.nc[0]*self.nc[1]*self.nc[2] + 1)
        # Espacio de acciones discretas (66 posibles movimientos)
        self.action_space = spaces.Discrete(66)
        
        # Inicializar agente
        self.agent = AgentActions()
        
        # Definir la meta
        self.goal = np.array([0, 0, 0])
        self.goal_state = self.x2z(self.goal)
        self.state = None
        self.previous_state = None
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """ Reinicia el estado del robot en una posición aleatoria. """
        self.state = self.np_random.uniform(self.xmin, self.xmax).astype(float)
        self.previous_state = self.state.copy()
        if(self.render_mode == 'human'):
            self.render()
        info = {}
        obs = self.x2z(self.state)
        return obs, info
    
    def step(self, action):
        """ Ejecuta una acción y actualiza el estado. """
        base_action = action % 22  # Índice de acción en AgentActions
        distance_idx = action // 22 + 1  # Factor de distancia (1, 2 o 3)
        pos_final, tiempo = self.agent.calculate_pos_odometry(21, [-6.15, 6.15], action)
        
        # Guardar estado previo
        self.previous_state = self.state.copy()
        
        # Actualizar estado continuo
        self.state[0] += pos_final['x'].astype(float)
        self.state[1] += pos_final['y'].astype(float)
        self.state[2] += pos_final['theta'].astype(float)
        if self.state[2] > self.xmax[2]:
            self.state[2] -= 360
        elif self.state[2] < self.xmin[2]:
            self.state[2] += 360
        
        # Convertimos a discreto
        discrete_state = self.x2z(self.state)
        
        # Calcular recompensa
        if discrete_state == 0:  # SINK
            reward = -100
            done = True
        elif discrete_state == self.goal_state:  # GOAL
            reward = 1000
            print("GOOOOOOOOOOOOOOOOOOOOOOOOOAL")
            done = True
        else:
            reward = 0
            
            # Calcular distancia al objetivo
            distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
            
            # Vector dirección hacia el objetivo
            dir_vec = self.goal[:2] - self.state[:2]
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 0:
                dir_vec /= norm_dir
            
            # Vector de movimiento del robot
            robot_vec = self.state[:2] - self.previous_state[:2]
            norm_robot = np.linalg.norm(robot_vec)
            if norm_robot > 0:
                robot_vec /= norm_robot
            
            # Producto escalar para medir alineación
            alignment = np.dot(dir_vec, robot_vec)  # Rango [-1,1]
            
            # Recompensa por alineación
            alignment_reward = 0.5 * alignment
            
            # Penalización por distancia
            distance_penalty = -0.1 * distance_to_goal
            
            # Recompensa total ajustada por tiempo
            # print("Proyeccion de vector: ", alignment_reward)
            # print(distance_penalty)
            # print(tiempo)
            reward += distance_penalty + alignment_reward - tiempo*0.1
            done = False
        
        info = {}
        return discrete_state, reward, done, False, info
    
    def x2z(self, xp):
        """ Convierte un estado continuo en su correspondiente celda discreta. """
        if np.any(xp < self.xmin) or np.any(xp >= self.xmax):
            return 0  # Estado inválido o fuera de límites
        
        zv = ((xp - self.xmin) / self.h).astype(int)
        index = np.ravel_multi_index(zv, self.nc) + 1  # Ajuste para empezar en 1
        return index
    
    def z2x(self, z):
        """ Convierte un índice discreto en su correspondiente punto continuo. """
        if z <= 0 or z > np.prod(self.nc):
            return None  # Estado inválido
        
        z -= 1  # Ajuste por el índice desde 1
        zv = np.unravel_index(z, self.nc)
        xp = (self.xmin + self.h * np.array(zv) + self.h / 2.0).astype(float)  # Centro de la celda
        print(xp)
        return xp
    
    def render(self, mode="human"):
        """ Muestra la posición actual del robot y la meta. """
        print(f"Estado actual: x={self.state[0]}, y={self.state[1]}, theta={self.state[2]}")
        print(f"Meta: x={self.goal[0]}, y={self.goal[1]}, theta={self.goal[2]}")

'''

if __name__ == "__main__":
    env = gym.make('my_robotenv-v0', render_mode='human')
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")
    state = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < 1000:
        
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
        action = env.action_space.sample()	  # Seleccionar acción aleatoria
        state, reward, done, _, _= env.step(action)
        step_count += 1'
'''