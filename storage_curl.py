import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random
import numpy as np
from data_augs import atari_random_shift

class Storage():

    def __init__(self, obs_shape, n_envs, num_steps, device):
        self.current_state_batch = torch.zeros(num_steps+1,n_envs,*obs_shape)
        self.reward_batch = torch.zeros(num_steps, n_envs)

        # Need to take care of the Non-Discrete action space case
        self.action_batch = torch.zeros(num_steps, n_envs)
        self.next_state_batch = torch.zeros(num_steps, n_envs,*obs_shape)
        self.done_batch = torch.zeros(num_steps, n_envs)
        #self.action_log_prob_batch = torch.zeros(num_steps)

        # 호준이형
        self.value_batch = torch.zeros(num_steps+1, n_envs)
        self.return_batch = torch.zeros(num_steps, n_envs)
        self.action_log_probs = torch.zeros(num_steps, n_envs)
        self.advantage_batch = torch.zeros(num_steps, n_envs)
        self.return_batch3 = torch.zeros(num_steps, n_envs)



        self.step = 0
        self.num_steps = num_steps
        self.device = device
        self.flag = 0
        self.num_envs = n_envs
        self.obs_shape = obs_shape

    def to(self, device):
        self.current_state_batch = self.current_state_batch.to(device)
        self.reward_batch = self.reward_batch.to(device)
        self.value_batch = self.value_batch.to(device)
        self.return_batch = self.return_batch.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_batch = self.action_batch.to(device)
        self.done_batch = self.done_batch.to(device)

    def store(self, current_state, action, reward, next_state, done, value):
        """
        Store the given SARS transition objective
        :return:
        """
        self.current_state_batch[self.step] = torch.from_numpy(current_state)
        self.action_batch[self.step] = torch.from_numpy(action)
        self.reward_batch[self.step] = torch.from_numpy(reward)
        self.next_state_batch[self.step] = torch.from_numpy(next_state)
        self.done_batch[self.step] = torch.tensor(done).clone().detach()

        # 호준이형
        self.value_batch[self.step] = torch.from_numpy(value)

        self.step = (self.step + 1) % self.num_steps


    def store_last(self, last_obs,last_value):
        self.current_state_batch[-1] = torch.from_numpy(last_obs)
        self.value_batch[-1] = torch.from_numpy(last_value)

    """
    # 검증 완료!! 3번이나!!
    def GAE(self, gamma=0.99, lmbda = 0.95):

        delta = self.reward_batch + gamma * self.value_batch[1:] * (1-self.done_batch) - self.value_batch[:-1]
        delta = delta.numpy()
        done_batch = self.done_batch.numpy()
        gae = np.zeros((self.num_envs))
        targets = []
        for delta_t, done in zip(delta[::-1], done_batch[::-1]):
            gae = delta_t + gamma * lmbda * gae * (1-done)
            targets.append(gae)
        targets = targets[::-1]
        targets = np.array(targets)
        self.return_batch = torch.from_numpy(targets).type(torch.float32)
        self.return_batch = self.return_batch + self.value_batch[:-1]

        self.advantage_batch = self.return_batch - self.value_batch[:-1]
        self.advantage_batch = (self.advantage_batch - torch.mean(self.advantage_batch)) / (torch.std(self.advantage_batch) + 1e-5)
    """

    def compute_estimates(self, gamma = 0.9, lmbda = 0.95, use_gae = True, normalize_adv = True):


        if use_gae == True:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = self.reward_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i+1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                A = gamma * lmbda * A * (1 - done) + delta
                self.return_batch[i] = A + value


        self.advantage_batch = self.return_batch - self.value_batch[:-1]
        if normalize_adv == True:
            self.advantage_batch = (self.advantage_batch - torch.mean(self.advantage_batch)) / (torch.std(self.advantage_batch) + 1e-5)

    def batch_generator(self):
        """
        Random sampler.
        Must do : Prioritize experience replay
        :return:
        """
        batch_size = self.num_steps * self.num_envs

        flat_obs_batch = self.current_state_batch.reshape(-1, *self.obs_shape).to(self.device)
        self.data_aug1_batch = atari_random_shift(flat_obs_batch).reshape(self.num_steps+1, self.num_envs, *self.obs_shape)
        self.data_aug2_batch = atari_random_shift(flat_obs_batch).reshape(self.num_steps+1, self.num_envs, *self.obs_shape)

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), batch_size, drop_last = True)
        for indices in sampler:

            # Since the storage is currently in the form of [ Batch, env, *obs_shape ] ,
            # we need to reshape it so that is has the form of [Batch * env, -1]

            current_state_batch = torch.FloatTensor(self.current_state_batch).reshape(-1, *self.obs_shape)[indices].to(self.device)
            action_batch = self.action_batch.reshape(-1)[indices].to(self.device)
            reward_batch = torch.FloatTensor(self.reward_batch).reshape(-1)[indices].to(self.device)
            next_state_batch = torch.FloatTensor(self.next_state_batch).reshape(-1, *self.obs_shape)[indices].to(self.device)
            done_batch = self.done_batch.reshape(-1)[indices].to(self.device)
            return_batch = self.return_batch.reshape(-1)[indices].to(self.device)
            advantage_batch = self.advantage_batch.reshape(-1)[indices].to(self.device)

            data_aug1_batch = self.data_aug1_batch.reshape(-1, *self.obs_shape)[indices].to(self.device)
            data_aug2_batch = self.data_aug2_batch.reshape(-1, *self.obs_shape)[indices].to(self.device)


            yield current_state_batch, action_batch, reward_batch, next_state_batch, done_batch, return_batch, advantage_batch, data_aug1_batch, data_aug2_batch

    def reward_done_batch(self):
        #return transposed shape to make each batch in the shape of [n_env, step]
        return self.reward_batch.T.numpy(), self.done_batch.T.numpy()



    def fetch_log_data(self):
        rew_batch = self.reward_batch.numpy()
        done_batch = self.done_batch.numpy()

        return rew_batch, done_batch