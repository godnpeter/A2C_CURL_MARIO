import gym
from mario_wrappers_v2 import wrap_environment
import random
import storage
import torch
from agent_CURL import A2C
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import gym
from mario_wrappers_v2 import wrap_environment
import gym_super_mario_bros
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
import ikov_vec_envs
import os
import atari_py
from subproc_vec_env import SubprocVecEnv
from mario_wrappers_v2 import wrap_environment

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

if __name__=='__main__':


    # Call the Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = wrap_environment(env)
    #env = make_env(env)
    # Thanks to the wrapper, we now get 84*84*4 observation each step

    log_name = 'evaluate'


    # Get cuda device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Call the storage
    observation = env.reset()
    storage = storage.Storage(env.observation_space.shape,  n_envs=1, num_steps=1, device=device)

    # Call Tensorboard logger
    writer = SummaryWriter('runs/' + log_name +time.strftime("%d-%m-%Y_%H-%M-%S"))

    # Call Value-Policy network
    embedder = model.Nature_CNN_Embedder(env.observation_space.shape, env.action_space.n).to(device)
    policy_net = policy.Value_Policy_Network(embedder).to(device)

    # Call the Agent
    #agent = DQN_agent.Agent(env,log_name, storage, explore = 0.02, lmbda=0.99, num_actions = env.action_space.n, device = device, explore_timesteps = 200000, writer = writer)
    agent = A2C(env, log_name, 1,storage, gamma=0.9, num_actions = env.action_space.n, device = device, writer=writer, update_interval = 1, network = policy_net, target_momentum=0.999)

    # Train the Agent
    agent.evaluate()

    # I guess this will be it?



