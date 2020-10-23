import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 ):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh

    def forward(self, x):
        for i in range(len(self.fc_layers)-1):
            x = self.fc_layers[i](x)
            x = self.nonlinearity(x)
        x = self.fc_layers[-1](x)
        return x  