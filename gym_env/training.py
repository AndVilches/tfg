import gymnasium as gym
import my_robotenv
import numpy as np
import torch
import torch.nn as nn
import os
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed



class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

def train_skrl():
    
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # Crear el entorno env y adaptarlo para RSL-RL
    env = gym.make_vec("my_robotenv-v1", num_envs=30, vectorization_mode="sync")

    # env = gym.make('my_robotenv-v1')
    env = wrap_env(env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Usar la GPU 0
    print(device)
    
    memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
    models["value"] = MLP(env.observation_space, env.action_space, device)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 2048  # memory_size
    cfg["mini_batches"] = 32
    cfg["grad_norm_clip"] = 0.2
    cfg["experiment"] = {
        "directory": "logs",              # Carpeta donde guarda los logs de TensorBoard
        "experiment_name": "ppo_myrobot5", # Subcarpeta para este experimento
        "write_interval": 10000,           # Cada cuántos timesteps se escriben logs (puede ser "auto" también)
        "checkpoint_interval": 100000,
        "store_separately": False,
        "wandb": False
}
    agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    cfg_trainer = {"timesteps": 1000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()

    agent.save(path="models/ppo_myrobot_final3.pt")

def test_skrl():

    env = gym.make("my_robotenv-v1", render_mode="human")
    env = wrap_env(env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Usar la GPU 0

    memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
    models["value"] = MLP(env.observation_space, env.action_space, device)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 2048  # memory_size
    cfg["mini_batches"] = 32
    cfg["grad_norm_clip"] = 0.2
    cfg["experiment"] = {
        "directory": "logs",              # Carpeta donde guarda los logs de TensorBoard
        "experiment_name": "ppo_myrobot3", # Subcarpeta para este experimento
        "write_interval": 10000,           # Cada cuántos timesteps se escriben logs (puede ser "auto" también)
        "checkpoint_interval": 100000,
        "store_separately": False,
        "wandb": False
}
    agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

    agent.load("models/ppo_myrobot_final3.pt")

    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    max_steps = 100000
    for t in range(max_steps):
        action, _ , _ = agent.act(obs, timestep=t, timesteps=max_steps)
        # action = action.detach().cpu().numpy()
        # action = np.array(action[0]).astype(np.float32)
        # action = np.array([action[0], action[1]]).astype(np.float32)
        
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        
        print("Accion tomada:", action)
        env.render()
        
        if terminated:
            print("Validacion termminada")
            break

    env.close()

if __name__ == "__main__":
    # train_skrl()
    test_skrl()