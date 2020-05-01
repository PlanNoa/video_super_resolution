from torch.nn.modules.module import Module
from .DepthProjectionLayer import DepthProjectionLayer

class DepthProjectionModule(Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()

    def forward(self, input1, input2):
        return DepthProjectionLayer.apply(input1, input2)