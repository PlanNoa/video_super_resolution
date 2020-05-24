import torch
import torch.nn as  nn
from my_packages.SuperResolution import SuperResolutionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SuperResolutionModule()
        self.SR_loss = _SR_loss()
        self.Flow_loss = _Flow_loss()
        self.loss4object = _loss4object()

    def _initialize_weights(self):
        # init pretrained model
        pass

    def forward(self):
        pass

    def loss_calculate(self, output, target, outputs):
        genSR_loss = self.SR_loss(output, target)
        objSR_loss = self.SR_loss(self.loss4object(output, target))
        Flow_loss = self.Flow_loss(outputs)
        objFlow_loss = self.Flow_loss(self.loss4object(outputs))
        pass
