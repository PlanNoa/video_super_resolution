import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule as Dnet
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule as Fnet
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule as Vnet
from utils.tools import up_scaling

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SRProjectionModule()
        self.FlowModule = Fnet()
        self.DepthModule = Dnet()
        self.VOSModule = Vnet()
        self.SR_loss = _SR_loss()
        self.Flow_loss = _Flow_loss()
        self.loss4object = _loss4object()

    def forward(self, data, target, high_frames, estimated_image, train=True):
        optical_flow = [self.FlowModule(data[0].copy(), data[1].copy()),
                        self.FlowModule(data[1].copy(), data[2].copy())]
        depth_map = [self.DepthModule(data[0].copy(), data[1].copy()),
                     self.DepthModule(data[1].copy(), data[2].copy())]
        data = torch.tensor([up_scaling(img, "data") for img in data])
        optical_flow = torch.tensor([up_scaling(img, "op", shape=data[0].shape) for img in optical_flow])
        depth_map = torch.tensor([up_scaling(img, "dp", shape=data[0].shape) for img in depth_map])
        input = torch.cat((data, optical_flow, depth_map,
                           estimated_image if estimated_image!=None else torch.tensor(data[0:1])), 0)
        # shape:[8, y, x, 3]

        output = self.model(input)

        data = [estimated_image, output, data[2]]
        optical_flow = [self.FlowModule(data[0], data[1]),
                        self.FlowModule(data[1], data[2])]
        depth_map = [self.DepthModule(data[0], data[1]),
                     self.DepthModule(data[1], data[2])]
        data = torch.tensor([up_scaling(img, "data") for img in data])
        optical_flow = torch.tensor([up_scaling(img, "op", shape=data[0].shape) for img in optical_flow])
        depth_map = torch.tensor([up_scaling(img, "dp", shape=data[0].shape) for img in depth_map])
        VOSmask = self.VOSModule(data[0], data[1]) != 0
        estimated_image = torch.tensor(np.bitwise_and(input[1], VOSmask))
        input = torch.cat((data, optical_flow, depth_map,
                           estimated_image), 0)
        # shape:[8, y, x, 3]

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
