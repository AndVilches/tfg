import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np


register(
    id='my_robotenv-v1',
    entry_point='my_robotenv:RLContinuousEnv' 
)

class RLContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = 400  # Número máximo de pasos por episodio
        self.current_step = 0
        
        # Definir los límites del espacio de estados
        self.xmin = np.array([-10, -10, -180])
        self.xmax = np.array([10, 10, 180])
        
        # Espacio de observaciones continuo
        self.observation_space = spaces.Dict(
        {
            "agent": spaces.Box(low=self.xmin, high=self.xmax, dtype=np.float64),
            "target": spaces.Box(low=self.xmin, high=self.xmax, dtype=np.float64),
            "prev_vel": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "prev_steering": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "distance": spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float64)

        })
        
       # Límites reales de las acciones
        self.max_velocity = 8.0
        self.max_angle = 60.0
        
        # Espacio de acciones normalizado (-1 a 1)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)
        self.previous_action = np.array([0.0,0.0]).astype(np.float64)
        # Inicializar estado y meta
        self.state = []
        self.goal = None
        self.last_distance_to_goal  = None
        self.scale = 30  # Escala para visualizar (1 unidad del mundo = 30 píxeles)
        self.window_size = 600  # Tamaño de la ventana cuadrada
        self.screen = None
        self.clock = None
    
    def _get_obs(self):
        return {
            "agent": self.state.astype(np.float64),
            "target": self.goal.astype(np.float64),
            "prev_vel": np.array([self.previous_action[0]], dtype=np.float64),
            "prev_steering": np.array([self.previous_action[1]], dtype=np.float64),
            "distance": np.array([self.last_distance_to_goal], dtype=np.float64)
        }    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Posición aleatoria del robot
        # self.state = self.np_random.uniform(self.xmin, self.xmax).astype(np.float64)
        self.state = np.array([0,0,0]).astype(np.float64)
        # self.state = np.array([1.9245576, -7.2271757, -178.94264]).astype(np.float64)
        if np.isnan(self.state).any():
            print("SE HA CREADO MAL EL ESTADO")
        # Definir una meta aleatoria dentro del espacio
        self.goal = self.np_random.uniform(self.xmin, self.xmax).astype(np.float64)
        
        if self.render_mode == "human":
            self.render()
        self.previous_action = np.array([0.0,0.0]).astype(np.float64)
        # self.render()
        self.last_distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """ Ejecuta una acción y actualiza el estado. """
        self.current_step += 1
        if action[0] > 0:
            velocity = 8
        elif action[0] < 0:
            velocity = -8
        else:
            velocity = 0
        angle = action[1] * self.max_angle
        if np.isnan(self.state).any():
            print("Accion actual: ", action)

        prev_state = self.state.copy()  # Guardar el estado anterior
        pos_final, tiempo = self._simulate_motion(self.state, velocity, angle)
        if np.isnan(pos_final).any():
            print("Accion actual: ", action)
        # Guardar estado previo y actualizar estado
        
        
        
        self.state = pos_final

        # Calcular recompensa basada en distancia y orientación
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        self.last_distance_to_goal = distance_to_goal
        distance_compare = distance_to_goal
        near_threshold = 2
        if np.isnan(distance_to_goal):
            print("Estado (self.state):", self.state)
            print("Meta (self.goal):", self.goal)
        
        movement_vector = self.state[:2] - prev_state[:2]
        goal_vector = self.goal[:2] - self.state[:2]

    # Normalizar vectores si su norma es mayor que 0
        if np.linalg.norm(movement_vector) > 0:
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
        if np.linalg.norm(goal_vector) > 0:
            goal_vector = goal_vector / np.linalg.norm(goal_vector)
        alignment_penalty = 0
        angle_diff = self.goal[2] - self.state[2]
        angle_diff = (angle_diff + 180) % 360 - 180
        alignment_penalty = np.dot(movement_vector, goal_vector)  # Máximo +1 si se mueve hacia la meta
        # Evitar división por cero
        norm_movement = np.linalg.norm(movement_vector)
        norm_goal = np.linalg.norm(goal_vector)

        if norm_movement > 0 and norm_goal > 0:
            cos_angle = np.clip(np.dot(movement_vector, goal_vector) / (norm_movement * norm_goal), -1.0, 1.0)
            angle_between_rad = np.arccos(cos_angle)  # en radianes
            angle_between_deg = np.degrees(angle_between_rad)  # 🔁 en grados
        else:
            angle_between_deg = np.pi  # Máximo desalineado si algún vector es nulo

        alignment_penalty = - (angle_between_deg / 180.0)  # de 0 a -0.01
        distance_to_goal = distance_to_goal
        # alignment_penalty = alignment_penalty*0.01
        # Recompensa total

        # print(alignment_penalty)
        reward = -distance_to_goal + alignment_penalty - tiempo*5 # tiempo tiene un valor constante de 0.1
        if distance_compare < 0.1 and abs(angle_diff) < 5:
            done = True
            reward += 10
            print("GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAL")
            print(self.state)
            print(self.goal)
        elif np.any(self.state < self.xmin) or np.any(self.state > self.xmax):
            done = True
            self.state = prev_state
            reward += -10
        elif self.current_step > self.max_steps:
            done = True
            reward += -10
        else:
            done = False
        self.previous_action = action
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def _simulate_motion(self, state, velocity, angle):
        """ Simula el movimiento basado en la velocidad y el ángulo de giro. """
        dt = 0.05
        axis_distance = 1
        theta_in = np.radians(state[2])
        new_state = state.copy()
        new_state[0] = state[0] + (dt * velocity * np.cos(np.radians(new_state[2])) * np.cos(np.radians(angle)))
        new_state[1] = state[1] + (dt * velocity * np.sin(np.radians(new_state[2])) * np.cos(np.radians(angle)))
       
       
        theta_f = theta_in + (dt * velocity * np.sin(np.radians(angle)) / axis_distance)
        new_state[2] = np.degrees(theta_f)
        # Asegurar que la orientación se mantiene en [-180, 180]
        if new_state[2] > 180:
            new_state[2] -= 360
        elif new_state[2] < -180:
            new_state[2] += 360
        
        return new_state, dt

    # def render(self, mode="human"):
    #     print(f"Estado actual: x={self.state[0]}, y={self.state[1]}, theta={self.state[2]}")
    #     print(f"Meta: x={self.goal[0]}, y={self.goal[1]}, theta={self.goal[2]}")
    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("RL Robot Environment")
            self.clock = pygame.time.Clock()

        # Manejo de eventos para que no se congele la ventana
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))  # Fondo blanco

        def to_pixel_coords(pos):
            x = int((pos[0] + 10) / 20 * self.window_size)
            y = int((pos[1] + 10) / 20 * self.window_size)
            return x, self.window_size - y

        agent_pos = to_pixel_coords(self.state)
        goal_pos = to_pixel_coords(self.goal)
        print("Poiscion del robot: ", self.state)
        print("Poiscion de la meta: ", self.goal)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_pos, 6)

        angle_rad = np.radians(self.state[2])
        arrow_length = 15
        dx = int(np.cos(angle_rad) * arrow_length)
        dy = int(np.sin(angle_rad) * arrow_length)
        end_pos = (agent_pos[0] + dx, agent_pos[1] - dy)
        pygame.draw.line(self.screen, (0, 0, 150), agent_pos, end_pos, 2)

        pygame.draw.circle(self.screen, (0, 255, 0), goal_pos, 6)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    
    # def get_observations(self):
    #     obs = np.array()
    #     extras = {}
    #     # print(obs)
    #     # extras ???  pytorch??  tensorflow?
    #     return obs, extras


# if __name__ == "__main__":
#     env = gym.make('my_robotenv-v1', render_mode='human')
#     obs, _ = env.reset()
#     done = False
#     step_count = 0

#     while not done and step_count < 10000:
#         action = [1,1]
#         obs, reward, done, _, _ = env.step(action)
#         print(obs)
#         print("Accion tomada: ", action )
#         env.render()
#         print(step_count)
#         step_count += 1

#     pygame.quit()


# if __name__ == "__main__":
#     env = gym.make('my_robotenv-v1', render_mode='human')
#     env = env.unwrapped
#     print(env.observation_space)
#     print("Check environment begin")
#     check_env(env.unwrapped)
#     print("Check environment end")
#     obs, _ = env.reset()
#     done = False
#     step_count = 0

#     while step_count < 100000000:
#         action = env.action_space.sample()  # Elegir una acción aleatoria
#         next_state, reward, done, _, _ = env.step(action)
        
#         # print(f"Step {step_count}:")
#         # print(f"Action: {action}")
#         # print(f"State: {next_state}")
#         # print(f"Reward: {reward}")
#         # print(f"Done: {done}")
#         # print("-------------------------")
        
#         step_count += 1

    
        
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
