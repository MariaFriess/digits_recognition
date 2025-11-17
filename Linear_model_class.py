import torch
import torch.nn as nn
import torch.utils.data as data


class LinearModel(nn.Module):
    def __init__(self, in_features=28*28, n_inside=32, out_features=10):
        super().__init__()
        self.layer1 = nn.Linear(in_features, n_inside)
        self.layer2 = nn.Linear(n_inside, out_features)

    def forward(self, data):
        s1 = self.layer1(data).relu()
        s2 = self.layer2(s1)
        return s2


