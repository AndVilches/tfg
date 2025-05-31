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
        self.previous_velocity = 0
        self.change_cont = 0
    
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
        self.state = self.np_random.uniform(self.xmin, self.xmax).astype(np.float64)
        # self.state = np.array([0,0,0]).astype(np.float64)
        # self.state = np.array([1.9245576, -7.2271757, -178.94264]).astype(np.float64)
        if np.isnan(self.state).any():
            print("SE HA CREADO MAL EL ESTADO")
        # Definir una meta aleatoria dentro del espacio
        # self.goal = self.np_random.uniform(self.xmin, self.xmax).astype(np.float64)
        x = 0.0
        y = 0.0
        theta = self.np_random.uniform(self.xmin[2], self.xmax[2])
        self.goal = np.array([x, y, theta], dtype=np.float64)
        self.change_cont = 0
        self.changed_direction = False
        if self.render_mode == "human":
            self.render()
        self.previous_action = np.array([0.0,0.0]).astype(np.float64)
        # self.render()
        self.last_distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """ Ejecuta una acción y actualiza el estado. """
        reward = -0.1
        self.current_step += 1

        # Interpretar acción
        if action[0] > 0:
            velocity = 8
        elif action[0] < 0:
            velocity = -8
        else:
            velocity = 0
        angle = action[1] * self.max_angle

        # Inicializar velocidad previa si no existe
        if not hasattr(self, "previous_velocity"):
            self.previous_velocity = 0

        # Detectar cambio de dirección
        self.changed_direction = (np.sign(velocity) != np.sign(self.previous_velocity)) and (velocity != 0 and self.previous_velocity != 0)
        if (self.changed_direction):
            reward-=2
                
        # Guardar el estado previo
        prev_state = self.state.copy()

        # Simular movimiento
        pos_final, tiempo = self._simulate_motion(self.state, velocity, angle)

        if np.isnan(pos_final).any():
            print("Accion actual: ", action)

        self.state = pos_final

        # Guardar nueva velocidad para el siguiente paso
        self.previous_velocity = velocity

        # Calcular recompensa basada en la distancia y orientación
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        if distance_to_goal < self.last_distance_to_goal:
            reward += 1
        self.last_distance_to_goal = distance_to_goal

        distance_compare = distance_to_goal
        near_threshold = 2

        if np.isnan(distance_to_goal):
            print("Estado (self.state):", self.state)
            print("Meta (self.goal):", self.goal)

        angle_diff = self.goal[2] - self.state[2]
        angle_diff = (angle_diff + 180) % 360 - 180

        if distance_compare < 0.2 and abs(angle_diff) < 5:
            done = True
            reward += 50
            print("GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAL")
            print(self.state)
            print(self.goal)
        elif np.any(self.state < self.xmin) or np.any(self.state > self.xmax):
            done = True
            self.state = prev_state
            reward += -10
        else:
            done = False

        self.previous_action = action
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def _simulate_motion(self, state, velocity, angle):
        """Simula el movimiento basado en la velocidad y el ángulo de giro, con ruido de SLAM y ruedas locas."""
        dt = 0.05
        axis_distance = 1
        theta_in = np.radians(state[2])
        new_state = state.copy()

        # Movimiento sin ruido
        new_state[0] = state[0] + (dt * velocity * np.cos(theta_in) * np.cos(np.radians(angle)))
        new_state[1] = state[1] + (dt * velocity * np.sin(theta_in) * np.cos(np.radians(angle)))
        
        # Actualización de orientación
        theta_f = theta_in + (dt * velocity * np.sin(np.radians(angle)) / axis_distance)
        new_theta = np.degrees(theta_f)

        # --- Ruido SLAM en posición (x, y) ---
        pos_noise_std = 0.01  # 1 cm
        new_state[0] += self.np_random.normal(0, pos_noise_std)
        new_state[1] += self.np_random.normal(0, pos_noise_std)

        # --- Ruido por ruedas locas en orientación ---
        if self.changed_direction:
            angle_noise = self.np_random.normal(0, 1)  # más ruido si cambió de dirección
        else:
            angle_noise = self.np_random.normal(0, 0.3)  # poco ruido si no

        new_state[2] = new_theta + angle_noise

        # Normalizar orientación a [-180, 180]
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
#         action = env.action_space.sample()
#         obs, reward, done, _, _ = env.step(action)
#         print(done)
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
