import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils
from utils import orthogonal_init

class Nature_CNN_Embedder(nn.Module):

    def __init__(self, observation_shape, num_actions):
        super(Nature_CNN_Embedder, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = observation_shape[0], out_channels=32, kernel_size=8, stride = 4)
        self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size= 4, stride=2)
        self.cnn3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride=1)
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 64*7*7, out_features= 512)
        self.output_feature_size = 512
        self.num_actions = num_actions
        self.apply(utils.orthogonal_init)


    def forward(self, x):
        """
        x =self.feature(x)
        x = nn.Flatten()(x)
        x = self.fc_layer(x)
        """
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)

        return x



