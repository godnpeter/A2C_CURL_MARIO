import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import orthogonal_init
import torch



class Value_Policy_Network(nn.Module):

    def __init__(self, embedder):
        super(Value_Policy_Network, self).__init__()
        self.embedder = embedder
        self.value_layer = orthogonal_init(nn.Linear(self.embedder.output_feature_size, 1), gain=1)
        self.policy_layer = orthogonal_init(nn.Linear(self.embedder.output_feature_size, self.embedder.num_actions), gain=0.01)
        self.train()

    def forward(self, x):
        embedding = self.embedder(x)


        # Policy Network
        policy_logits = self.policy_layer(embedding)
        log_probs = F.log_softmax(policy_logits, dim = 1)
        probs_sampler = Categorical(torch.exp(log_probs))

        action = probs_sampler.sample()


        # Value Network
        value = self.value_layer(embedding)

        return action, probs_sampler, value, embedding
