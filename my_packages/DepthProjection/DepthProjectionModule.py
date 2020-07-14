import torch
import torch.nn as nn
from .MegaDepth.models.HG_model import HGModel


class DepthProjectionModule(nn.Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.model = HGModel()

    def forward(self, input):

        p = self.model(input)
        p = torch.squeeze(p[0])

        return p