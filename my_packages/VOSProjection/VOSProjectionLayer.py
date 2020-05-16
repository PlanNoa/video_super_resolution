import torch
from torch.autograd import Function

class VOSProjectionLayer(Function):
    def __init__(self):
        super(VOSProjectionLayer, self).__init__()

    @staticmethod
    def forward(ctx, input1, input2):
        pass

    @staticmethod
    def backward(ctx, gradoutput):
        return gradoutput