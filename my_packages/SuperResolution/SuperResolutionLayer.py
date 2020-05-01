import torch
from torch.autograd import Function

class SuperResolutionLayer(Function):
    def __init__(self):
        super(SuperResolutionLayer, self).__init__()

    @staticmethod
    def forward(ctx, input1, input2, input3):
        pass

    @staticmethod
    def backward(ctx, gradoutput):
        return gradoutput