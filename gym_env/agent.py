import numpy as np


'''
ESTA CLASE FUE CREADA PARA UN MODELO CONTINUO, ACTUALMENTE NO TIENE APLICACION      
'''

class AgentActions:
    def __init__(self, max_turn_angle=60, distances=[1, 2, 3]):
        """
        Clase que define las acciones discretas del agente.
        :param max_turn_angle: Ángulo máximo de giro en grados.
        :param distances: Lista de distancias posibles.
        """
        self.max_turn_angle = max_turn_angle  # En grados
        self.distances = distances  # Distancias posibles
        self.num_actions = 22 * len(distances)  # Acciones con diferentes distancias
        
        # Generamos la lista de acciones posibles
        self.actions = self._generate_actions()
    
    def _generate_actions(self):
        """ Genera la lista de acciones posibles con distintas distancias. """
        base_actions = [(1, 0), (-1, 0)]  # Acción 0 y 1 (recto hacia adelante y atrás)
        aux = (22 - 2) // 4
        value = self.max_turn_angle / aux
        value_aux = value
        velocity = 0.8
        
        for i in range(aux):
            base_actions.append((velocity, value)) # alante izquierda 2
            base_actions.append((-velocity, value)) #atras izquierda 3
            base_actions.append((velocity, -value)) # alante derecha 4
            base_actions.append((-velocity, -value)) #atras derecha 5
            value += value_aux
        
        # Expandir acciones con distancias
        actions = []
        for d in self.distances:
            actions.extend([(v, a, d) for v, a in base_actions])
        return actions
    
    def get_action(self, action_idx):
        """ Devuelve la acción correspondiente a un índice dado. """
        if 0 <= action_idx < self.num_actions:
            return self.actions[action_idx]
        else:
            raise ValueError("Índice de acción fuera de rango.")
    
    def SimModel(self, robot_angle):
        """
        Calcula el ángulo de odometría basado en el ángulo de giro del robot.
        :param robot_angle: Ángulo de giro del robot.
        :return: Ángulo de odometría en radianes.
        """
        if robot_angle != 0:
            if robot_angle < 0:
                robot_angle = abs(robot_angle)
                odometry_angle = 0.0044 * (robot_angle) ** 2 + 0.5196 * robot_angle + 0.3179
                odometry_angle = -odometry_angle
            else:
                odometry_angle = 0.0044 * (robot_angle) ** 2 + 0.5196 * robot_angle + 0.3179
        else:
            odometry_angle = 0
        
        return np.radians(odometry_angle)
    
    def calculate_pos_odometry(self, cells, length_size, action):
        """
        Calcula la posición final basada en la acción tomada y la distancia deseada.
        :param cells: Número de celdas en la cuadrícula.
        :param length_size: Lista con los límites del espacio.
        :param action: Índice de la acción tomada.
        :return: Posición final (x, y, theta) y tiempo empleado.
        """
        punto_final = {'x': 0, 'y': 0, 'theta': 0}
        dt = 0.001
        velocity, robot_angle, distancia = self.actions[action]
        cell_distance = distancia * ((length_size[1] - length_size[0]) / cells)
        aux_point = [0, 0, 0]
        
        odometry_angle = self.SimModel(robot_angle)
        parameter = [odometry_angle, velocity]
        cont_time = 0
        
        while abs(aux_point[0]) < cell_distance and abs(aux_point[1]) < cell_distance:
            self.modelT(aux_point, parameter, dt)
            cont_time += dt
        
        punto_final['x'], punto_final['y'], punto_final['theta'] = aux_point
        return punto_final, cont_time
    
    def modelT(self, x, u, dt):
        """
        Modelo de movimiento basado en la velocidad y el ángulo de giro.
        :param x: Estado actual [x, y, theta].
        :param u: Entrada [ángulo de giro, velocidad].
        :param dt: Pequeño paso de tiempo.
        """
        if len(x) < 3 or len(u) < 2:
            raise ValueError("Tamaño insuficiente de vectores 'x' o 'u'.")

        DistanciaEjes = 1.0  # Distancia entre ejes
        x[0] += dt * u[1] * np.cos(x[2]) * np.cos(u[0])
        x[1] += dt * u[1] * np.sin(x[2]) * np.cos(u[0])
        x[2] += dt * u[1] * np.sin(u[0]) / DistanciaEjes

# if __name__ == "__main__":
#     actions = AgentActions()
    
#     print("Lista de acciones posibles:")
#     for i, action in enumerate(actions.actions):
#         print(f"Acción {i}: Velocidad = {action[0]}, Ángulo = {action[1]}, Distancia = {action[2]}")
    
#     test_idx = 5
#     print(f"\nProbando acción con índice {test_idx}: {actions.get_action(test_idx)}")
    
#     pos, tiempo = actions.calculate_pos_odometry(300, [-30, 30], test_idx)
#     print(f"\nPosición final: {pos}, Tiempo empleado: {tiempo:.4f} segundos")
