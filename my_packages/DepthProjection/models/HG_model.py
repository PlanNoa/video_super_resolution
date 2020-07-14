import torch
from torch import nn
from torch.autograd import Variable
from my_packages.DepthProjection.models import pytorch_DIW_scratch


class HGModel(nn.Module):

    def __init__(self, pretrained=None):
        super(HGModel, self).__init__()

        model = pytorch_DIW_scratch.pytorch_DIW_scratch
        if pretrained is None:
            self.netG = model
        else:
            pretrained_dict = torch.load(pretrained)

            model_dict = model.state_dict()
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            self.netG = model


    def forward(self, input):
        input = Variable(input)
        prediction = self.netG.forward(input)
        return prediction