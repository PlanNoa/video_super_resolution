import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.tools import up_scaling

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SRProjectionModule().train()
        self.FlowModule = FlowProjectionModule().eval()
        self.DepthModule = DepthProjectionModule().eval()
        self.VOSModule = VOSProjectionModule().eval()
        self.SR_loss = _SR_loss()
        self.Flow_loss = _Flow_loss()
        self.loss4object = _loss4object()

    def forward(self, data, target, high_frames, estimated_image, train=True):

        optical_flow = [self.FlowModule(data[0], data[1]),
                        self.FlowModule(data[1], data[2])]
        depth_map = [self.DepthModule(data[0], data[1]),
                     self.DepthModule(data[1], data[2])]
        optical_flow = torch.tensor(list(map(up_scaling, optical_flow)))
        depth_map = torch.tensor(list(map(up_scaling, depth_map)))
        data = torch.tensor(list(map(up_scaling, data)))
        input = torch.cat((data[0], data[1], data[2],
                           optical_flow[0], optical_flow[1],
                           depth_map[0], depth_map[1],
                           estimated_image), 0)
        output = self.model(input)

        data = [estimated_image, output, data[2]]
        optical_flow = [self.FlowModule(data[0], data[1]),
                        self.FlowModule(data[1], data[2])]
        depth_map = [self.DepthModule(data[0], data[1]),
                     self.DepthModule(data[1], data[2])]
        VOSmask = self.VOSModule(data[0], data[1]) != 0
        estimated_image = np.bitwise_and(input[1], VOSmask)
        input = torch.cat((data[0], data[1], data[2],
                           optical_flow[0], optical_flow[1],
                           depth_map[0], depth_map[1],
                           estimated_image))
        output = self.model(input)

        high_frames[1] = output
        loss = self.loss_calculate(output, target, high_frames) if train else None
        return output, loss

    def loss_calculate(self, output, target, outputs):
        genSR_loss = self.SR_loss(output, target)
        objSR_loss = np.mean(self.SR_loss(i, j) for i, j in self.loss4object(output, target))
        Flow_loss = self.Flow_loss(outputs)
        objFlow_loss = self.Flow_loss(self.loss4object(outputs))
        return [genSR_loss, objSR_loss, Flow_loss, objFlow_loss]
