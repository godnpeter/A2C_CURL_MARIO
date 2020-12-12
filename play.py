import gym
import atari_wrappers
import random
import storage
import torch
import agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
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
    env_name = 'PongNoFrameskip-v4'
    env = gym.make(env_name)

    # Thanks to the wrapper, we now get 84*84*4 observation each step
    env = atari_wrappers.wrap_deepmind(env)

    # Get cuda device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Call Tensorboard logger
    writer = SummaryWriter('runs/' + env_name + time.strftime("%d-%m-%Y_%H-%M-%S"))

    # Call the Agent
    agent = agent.Agent(env, env_name,storage, explore = 0.02, lmbda=0.99, num_actions = env.action_space.n, device = device, explore_timesteps = 200000, writer = writer)

    # Evaluate the Agent
    agent.evaluate()

    # I guess this will be it?



