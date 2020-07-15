import torch
import torch.nn as nn
from my_packages.DepthProjection.models.HG_model import HGModel


class DepthProjectionModule(nn.Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.model = HGModel("pretrained/best_generalization_net_G.pth")

    def forward(self, input):
        p = self.model(input)
        p = torch.squeeze(p[0])
        return p