import torch
from torch.autograd import Function

class DepthProjectionLayer(Function):
    def __init__(self):
        super(DepthProjectionLayer, self).__init__()

    @staticmethod
    def forward(ctx, input1, input2):
        pass

    @staticmethod
    def backward(ctx, gradoutput):
        return gradoutput