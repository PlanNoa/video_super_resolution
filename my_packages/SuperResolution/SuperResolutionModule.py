from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

from ..DepthProjection.DepthProjectionModule import DepthProjectionModule
from ..FlowProjection.FlowProjectionModule import FlowProjectionModule
from .SuperResolutionLayer import SuperResolutionLayer

class SuperResolutionModule(Module):
    def __init__(self):
        super(SuperResolutionModule, self).__init__()

    def forward(self, input1, input2):
        optical_flow = FlowProjectionModule(input1, input2)
        depth_map = DepthProjectionModule(input1, input2)
        concat = torch.cat(optical_flow, depth_map)
        return SuperResolutionLayer.apply(input1, input2, concat)