import torch
import torch.nn as nn
from utils.tools import transpose1323
from my_packages.DepthProjection.models.HG_model import HGModel


class DepthProjectionModule(nn.Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.model = HGModel("my_packages/DepthProjection/pretrained/best_generalization_net_G.pth")

    def forward(self, input):
        input = transpose1323(input)
        data1 = self.model(input[0:1])
        data2 = self.model(input[1:2])
        p = torch.mean(torch.stack([data1, data2]), dim=0)
        p = torch.squeeze(p[0])
        return p