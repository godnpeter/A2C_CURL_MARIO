import gym
import atari_wrappers
import random
import storage
import torch
import agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import parallel_env
import policy
import model
import os
import atari_py
"""
Where we actually run the whole process of learning from the environment

We'll have 
1. the agent perform real actions within the environment
2. store the performed SARSA in the replay buffer
3. At every update step, perform optimization
4. 
eps_start=1.0
eps_decay=.999985
eps_min=0.02
"""
def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_global_log_levels(level):
    gym.logger.set_level(level)

if __name__=='__main__':

    seed = random.randint(0,9999)
    set_global_seeds(seed)
    set_global_log_levels(40)

    # Call the Environment
    env_name = 'PongNoFrameskip-v4'

    # Get cuda device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    num_workers = 16
    env = gym.make(env_name)
    """
    def create_env(env_name,  n_envs):
        env = gym.make(env_name)
        # Wrap atari-environment with the deepmind-style wrapper
        atari_env_list = atari_py.list_games()
        for atari_env in atari_env_list:
            if atari_env in env_name.lower():
                env = atari_wrappers.wrap_deepmind(env)
                env = atari_wrappers.TransposeFrame(env)
                break
        # Parallelize the environment
        env = parallel_env.ParallelEnv(n_envs, env)
        return env
    """
    # Thanks to the wrapper, we now get 84*84*4 observation each step
    env = atari_wrappers.wrap_deepmind(env)
    env = atari_wrappers.TransposeFrame(env)

    #env = create_env(env_name, num_workers)
    #Parallelize the environment
    env = parallel_env.ParallelEnv(num_processes=num_workers, env=env)



    # Call the storage
    observation = env.reset()

    num_steps = 32
    #storage = storage2.Storage(env.observation_space.shape, num_steps,num_workers ,device = device)
    storage = storage.Storage(env.observation_space.shape,  num_workers, num_steps, device=device)


    # Call Tensorboard logger
    writer = SummaryWriter('runs/' + env_name + '-A2C_POLICY-test'+time.strftime("%d-%m-%Y_%H-%M-%S"))


    # Call Value-Policy network
    embedder = model.Nature_CNN_Embedder(env.observation_space.shape, env.action_space.n).to(device)
    policy_net = policy.Value_Policy_Network(embedder).to(device)

    #embedder = create_embedder(env)
    #policy_net = create_policy(env, embedder)
    #policy_net = policy_net.to(device)


    env_name='test'


    # Call the Agent
    agent = agent.A2C(env, env_name, num_workers,storage, gamma=0.99, num_actions = env.action_space.n, device = device, writer=writer, update_interval = num_steps, network = policy_net)

    # Train the Agent
    agent.train(1000000000)

    # I guess this will be it?



