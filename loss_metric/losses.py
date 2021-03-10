import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor([10.]))
        self.weight.requires_grad = True
        self.bias = nn.Parameter(torch.tensor([-5.]))
        self.bias.requires_grad = True

    def _get_centroids(self, dvec):
        return torch.mean(dvec, dim=1)

    def _get_centroid(self, )