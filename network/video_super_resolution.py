import torch
import numpy as np
from my_packages.SuperResolution import SuperResolutionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SuperResolutionModule()
        self.SR_loss = _SR_loss()
        self.Flow_loss = _Flow_loss()
        self.loss4object = _loss4object()

    def forward(self, input1, input2, input3, target, outputs):
        output = self.model(input1, input2, input3)
        loss = self.loss_calculate(output, target, outputs)
        return self.model(input1, input2, input3), loss

    def loss_calculate(self, output, target, outputs):
        genSR_loss = self.SR_loss(output, target)
        objSR_loss = np.mean(self.SR_loss(i, j) for i, j in self.loss4object(output, target))
        Flow_loss = self.Flow_loss(outputs)
        objFlow_loss = self.Flow_loss(self.loss4object(outputs))
        return (genSR_loss + objSR_loss + Flow_loss + objFlow_loss)/4
