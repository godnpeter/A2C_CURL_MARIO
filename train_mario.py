import gym
from mario_wrappers_v2 import wrap_environment
import gym_super_mario_bros
import random
import storage
import storage_curl
import torch
import agent
import agent_CURL
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


    # Get cuda device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    num_workers = 6
    # Thanks to the wrapper, we now get 84*84*4 observation each step
    #env = gym_super_mario_bros.make(env_name)
    #env = wrap_environment(env)

    #env = create_env(env_name, num_workers)
    #Parallelize the environment
    #env = parallel_env.ParallelEnv(num_processes=num_workers, env=env)

    #Todo gamma 이거 검증 필요!
    #gamma = 0.99

    #env = ikov_vec_envs.make_vec_envs(seed, num_workers,gamma, None, device, True)

    def create_env(env_name, n_envs):
        # Different from atari environment, environment could not be pickled
        # Therefore, environment should be initialized separately in subprocess
        class EnvFnWrapper():
            def __init__(self, env_name):
                self.env_name = env_name

            def __call__(self, *args, **kwargs):
                env = gym_super_mario_bros.make(self.env_name)
                env = wrap_environment(env)
                return env

        env_fn = EnvFnWrapper(env_name)
        env = SubprocVecEnv(n_envs, env_fn)
        return env

    print('INITIALIZAING ENVIRONMENTS...')
    env = create_env('SuperMarioBros-v0', num_workers)
    print('DONE!')


    # Call the storage
    observation = env.reset()
    num_steps = 50
    #storage = storage2.Storage(env.observation_space.shape, num_steps,num_workers ,device = device)
    storage = storage_curl.Storage(env.observation_space.shape,  num_workers, num_steps, device=device)

    env_name = 'SuperMarioBros-v0' + '_CURL'
    # Call Tensorboard logger
    writer = SummaryWriter('runs/' + env_name + '-A2C_POLICY-'+time.strftime("%d-%m-%Y_%H-%M-%S"))


    # Call Value-Policy network
    embedder = model.Nature_CNN_Embedder(env.observation_space.shape, env.action_space.n).to(device)
    policy_net = policy.Value_Policy_Network(embedder).to(device)

    #embedder = create_embedder(env)
    #policy_net = create_policy(env, embedder)
    #policy_net = policy_net.to(device)




    # Call the Agent
    agent = agent_CURL.A2C(env, env_name, num_workers,storage, gamma=0.9, num_actions = env.action_space.n, device = device, writer=writer, update_interval = num_steps, network = policy_net, target_momentum=0.999)

    # Train the Agent
    agent.train(1000000000)

    # I guess this will be it?



