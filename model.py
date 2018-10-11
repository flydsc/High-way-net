import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import math


class HighWayLayer(nn.Module):
    def __init__(self, state_size, active_func):
        super().__init__()
        self.state_size = state_size
        self.active_func = active_func
        self.gate = nn.Linear(state_size, state_size)
        self.h = nn.Linear(state_size, state_size)
        stdv = 1. / math.sqrt(self.state_size)

        self.gate.weight.data.uniform_(-stdv, stdv)
        self.gate.bias.data.fill_(-1)
        init.xavier_normal(self.h.weight)

    def forward(self, x):
        gate = F.sigmoid(self.gate(x))
        return torch.mul(self.active_func(self.h(x)), gate) + torch.mul(x, (1 - gate))



class HighWay(nn.Module):
    def __init__(self, layer_n, state_size, active_func):
        super().__init__()
        layerlist = []
        for _ in range(layer_n):
            layerlist.append(HighWayLayer(state_size, active_func))
        self.layers = nn.ModuleList(layerlist)

    def forward(self, x):
        for ly in self.layers:
            x = ly(x)
        return x



