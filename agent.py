import model
import policy
import torch.nn.init as init
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import torch
import utils
import copy
import time
from collections import deque
import torch.nn.functional as F
from torch.distributions import Categorical



class A2C:

    def __init__(self, env, env_name,  n_envs,storage,gamma, num_actions, device, writer, update_interval, network):
        torch.set_num_threads(1)
        self.env_name = env_name
        self.action_space = env.action_space
        #self.embedder = model.Nature_CNN_Embedder(env.observation_space.shape, env.action_space.n).to(device)
        #self.policy_net = policy.Value_Policy_Network(self.embedder).to(device)
        self.policy_net = network
        #CURL Momentum encoder
        #self.target_policy = copy.deepcopy(self.policy_net)
        #self.target_momentum = target_momentum

        self.storage = storage
        self.env = env
        self.gamma = gamma
        self.device = device
        self.writer = writer
        self.n_envs = n_envs
        self.loss_array = []

        #TODO : update interval
        self.update_interval = update_interval



        # Needed for reward logging
        self.env_reward_list = []
        for i in range(self.n_envs):
            self.env_reward_list.append([])
        self.num_episodes = 0
        self.temp_rew = 0
        self.mean_reward_deque = deque(maxlen = 200)

        #TODO : implement anneealing learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = 1e-4)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)

        # Attributes to save the best model
        self.best_reward_mean = None
        self.best_model = None

        #loss functions

    def predict(self, obs):
        self.policy_net.eval()
        obs = torch.FloatTensor(obs).to(device=self.device)
        action, _, value = self.policy_net(obs)

        return action.cpu().numpy(), value.cpu().detach().squeeze().numpy()

    def train(self, num_timesteps):
        current_state = self.env.reset()
        #current_state = current_state.numpy()
        timestep = 0
        update_timestep = 0
        while timestep < num_timesteps:
            # Run Policy
            for _ in range(self.update_interval):
                with torch.no_grad():
                    action, value = self.predict(current_state)
                next_state, reward, done, info = self.env.step(action)

                #next_state = next_state.numpy()
                #reward= reward.squeeze().numpy()
                self.storage.store(current_state, action, reward, next_state ,done, value)
                current_state = next_state

            last_state = next_state
            with torch.no_grad():
                _, last_val = self.predict(last_state)
            self.storage.store_last(last_state, last_val)

            # Compute advantage estimates
            self.storage.compute_estimates()
            # Optimize policy & value
            self.optimize(timestep)
            update_timestep += 1

            # Log the training-procedure
            timestep += self.update_interval * self.n_envs
            self.log(self.writer, timestep)

            if update_timestep % 500 == 0:
                print('NETWORK SAVED')
                torch.save(self.policy_net.state_dict(), self.env_name + '-best_model_' + 'A2C'+time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.env.close()
        torch.save({'state_dict': self.policy_net.state_dict()}, self.writer.logdir + '/model.pth')

    def optimize(self, timestep):
        self.policy_net.train()
        # Obtain the train_batch
        batch_generator = self.storage.batch_generator()
        # batch = self.storage.random_batch()
        for batch in batch_generator:
            # Now  we've got the episode_batch
            # I've found that TD(1) doesn't make a good value target approximation... So I'll try implementing the GAE instead

            current_state_batch, action_batch, reward_batch, next_state_batch, done_batch, return_batch, advantage_batch = batch
            action_pred_batch, probs_sampler_batch, value_pred_batch = self.policy_net(current_state_batch)
            # Value loss
            value_loss = F.smooth_l1_loss(value_pred_batch, return_batch.unsqueeze(1).detach())

            # Policy loss
            #advantage = return_batch.unsqueeze(1) - value_pred_batch
            #advantage = advantage.squeeze()
            #advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-5)
            #advantage = advantage.detach()

            policy_loss = - (probs_sampler_batch.log_prob(action_batch)) * (advantage_batch.detach())
            policy_loss = policy_loss.mean()

            # Entropy loss
            entropy_loss = probs_sampler_batch.entropy().mean()

            self.writer.add_scalar('Loss/value loss', value_loss, timestep)
            self.writer.add_scalar('Loss/policy loss', policy_loss, timestep)
            self.writer.add_scalar('Loss/entropy loss', entropy_loss, timestep)
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

        """
        if timestep % 2000:
            for name, param in self.policy_net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), timestep)
                self.writer.add_histogram(name + '.grad', param.grad.clone().cpu().data.numpy(), timestep)
        """


    def log(self, writer, timestep):

        # reward batch랑 done batch를 받은 후에 logger에 올바르게 reward를 올리고싶다.
        # 그럼 우리는 언제나 logger에는 done일 때 tot_reward를 넘겨주는 것이 맞겟지?
        # 그러므로, environment별로 각자의 reward system이 따로 돌아가고 있을거라고 생각해야한다.

        reward_batch, done_batch = self.storage.reward_done_batch()
        steps = reward_batch.shape[1]
        n_envs = reward_batch.shape[0]

        for env in range(n_envs):
            for step in range(steps):
                self.env_reward_list[env].append(reward_batch[env][step])

                if done_batch[env][step]:


                    tot_reward = np.sum(self.env_reward_list[env])

                    self.mean_reward_deque.append(tot_reward)
                    mean_reward = np.mean(self.mean_reward_deque)

                    if self.best_reward_mean is None or mean_reward > self.best_reward_mean:
                        self.best_reward_mean = mean_reward

                    self.num_episodes +=1
                    print("%d:  %d games, mean reward %.3f" % (timestep, self.num_episodes, mean_reward))
                    self.env_reward_list[env] = []
                    self.writer.add_scalar('Timestep/reward', tot_reward, timestep)
                    self.writer.add_scalar('Timestep/mean_reward', mean_reward, timestep)
                    tot_reward = 0


    def evaluate(self):
        """
        used to play the best learned model
        """
        self.policy_net.load_state_dict(torch.load('SuperMarioBros-1-1-v0_numstep50_gamma09lmbda95'+ '-best_model_A2C'))
        mean_reward_list = []
        tot_reward = 0
        current_state = self.env.reset()
        current_state = torch.FloatTensor(current_state)
        num_episodes = 0
        timestep = 0
        while True:
            self.env.render()
            #if random.random() > 0.1:
            with torch.no_grad():
                action, value = self.predict(current_state.unsqueeze(0))

            next_state, reward, done, info = self.env.step(action[0])
            tot_reward += reward
            next_state = torch.FloatTensor(next_state)
            timestep +=1

            if done:
                mean_reward_list.append(tot_reward)
                mean_reward = np.mean(mean_reward_list[-100:])

                print("%d:  %d games, mean reward %.3f" % (timestep, len(mean_reward_list), mean_reward))

                current_state = self.env.reset()
                current_state = torch.FloatTensor(current_state)
                num_episodes += 1
                tot_reward = 0
            else:
                current_state = next_state