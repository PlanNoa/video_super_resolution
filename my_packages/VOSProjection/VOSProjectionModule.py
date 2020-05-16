from torch.nn.modules.module import Module
from .VOSProjectionLayer import VOSProjectionLayer

class VOSProjectionModule(Module):
    def __init__(self):
        super(VOSProjectionModule, self).__init__()

    def forward(self, input1, input2):
        return VOSProjectionLayer.apply(input1, input2)