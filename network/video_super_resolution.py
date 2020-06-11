import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.tools import up_scailing, down_scailing, get_gpu_usage
from PIL import Image

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SRProjectionModule().cpu()
        self.FlowModule = FlowProjectionModule()
        self.DepthModule = DepthProjectionModule()
        self.VOSModule = VOSProjectionModule()
        self.SR_loss = _SR_loss()
        self.Flow_loss = _Flow_loss()
        self.loss4object = _loss4object()

    def forward(self, data, target, high_frames, estimated_image, train=True):
        with torch.no_grad():
            optical_flow = [self.FlowModule(data[0].copy(), data[1].copy()),
                            self.FlowModule(data[1].copy(), data[2].copy())]
            depth_map = [self.DepthModule(data[0].copy(), data[1].copy()),
                         self.DepthModule(data[1].copy(), data[2].copy())]
            data = torch.tensor(data)
            optical_flow = torch.tensor([up_scailing(img, shape=data[0].shape) for img in optical_flow])
            depth_map = torch.tensor([up_scailing(img, shape=data[0].shape) for img in depth_map])
            estimated_image = down_scailing(estimated_image) if estimated_image != None else torch.tensor(data[0:1])

        input = torch.cat((data, optical_flow, depth_map, estimated_image), 0)
        input = input.transpose(1, 3).transpose(2, 3).type(dtype=torch.float32).cuda()
        # input shape: [8, 3, y/4, x/4]

        output = self.model(input)
        # output shape: [1, 3, y, x]
        output = torch.tensor(output).transpose(1, 3).transpose(1, 2).type(dtype=torch.float32)

        with torch.no_grad():
            data = [estimated_image[0].numpy(), down_scailing(output), data[2].numpy()]
            optical_flow = [self.FlowModule(data[0], data[1]),
                            self.FlowModule(data[1], data[2])]
            depth_map = [self.DepthModule(data[0], data[1]),
                         self.DepthModule(data[1], data[2])]
            data = torch.tensor(data)
            optical_flow = torch.tensor([up_scailing(img, shape=data[0].shape) for img in optical_flow])
            depth_map = torch.tensor([up_scailing(img, shape=data[0].shape) for img in depth_map])
            VOSmask = self.VOSModule(data[0], data[1]) != 0
            VOSmask = np.stack((VOSmask,)*3, axis=-1)

            estimated_image = torch.tensor([np.bitwise_and(data[1].numpy(), np.array(VOSmask))])

        input = torch.cat((data, optical_flow, depth_map, estimated_image), 0)
        input = input.transpose(1, 3).transpose(2, 3).type(dtype=torch.float32).cuda()
        # input shape:[8, 3, y/4, x/4]

        output = self.model(input)
        # output shape: [1, 3, y, x]
        output = torch.tensor(output).transpose(1, 3).transpose(1, 2).type(dtype=torch.float32)

        print(np.array(high_frames).shape)
        high_frames[1][0] = output
        loss = self.loss_calculate(output, target, high_frames) if train else None
        return output, loss

    def loss_calculate(self, output, target, outputs):
        genSR_loss = self.SR_loss(output, target)
        objSR_loss = np.mean(self.SR_loss(i, j) for i, j in self.loss4object(output, target))
        Flow_loss = self.Flow_loss(outputs)
        objFlow_loss = self.Flow_loss(self.loss4object(outputs))
        return [genSR_loss, objSR_loss, Flow_loss, objFlow_loss]
