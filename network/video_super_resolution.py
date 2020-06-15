import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.tools import up_scailing, down_scailing, get_gpu_usage, maskprocess
from PIL import Image


class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SRProjectionModule()
        self.FlowModule = FlowProjectionModule().eval()
        self.DepthModule = DepthProjectionModule().eval()
        self.VOSModule = VOSProjectionModule().eval()
        self.SR_loss = _SR_loss().eval()
        self.Flow_loss = _Flow_loss().eval()
        self.loss4object = _loss4object().eval()

    def forward(self, data, target, high_frames, estimated_image, train=True):
        with torch.no_grad():
            optical_flow = [self.FlowModule(data[0].copy(), data[1].copy()),
                            self.FlowModule(data[1].copy(), data[2].copy())]
            depth_map = [self.DepthModule(data[0].copy(), data[1].copy()),
                         self.DepthModule(data[1].copy(), data[2].copy())]
            data = torch.tensor(data)
            optical_flow = torch.tensor([up_scailing(img, shape=data[0].shape) for img in optical_flow])
            depth_map = torch.tensor([up_scailing(img, shape=data[0].shape) for img in depth_map])
            estimated_image = torch.tensor([down_scailing(estimated_image.detach())]) if type(estimated_image) != type(
                None) else torch.tensor(data[0:1])
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
        optical_flow = torch.tensor([up_scailing(img, shape=data[0].shape) for img in optical_flow],
                                    dtype=torch.float32, requires_grad=True)
        depth_map = torch.tensor([up_scailing(img, shape=data[0].shape) for img in depth_map], dtype=torch.float32,
                                 requires_grad=True)
        VOSmask = maskprocess(self.VOSModule(data[0], data[1]) != 0)
        masked_estimated_image = torch.tensor([np.ma.MaskedArray(data[1].numpy(), VOSmask, fill_value=0).filled()],
                                              dtype=torch.float32, requires_grad=True)
        data = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        input = torch.cat((data, optical_flow, depth_map, masked_estimated_image), 0)
        input = input.transpose(1, 3).transpose(2, 3).cuda()
        # input shape:[8, 3, y/4, x/4]

        output = self.model(input)
        # output shape: [1, 3, y, x]
        output = torch.tensor(output, dtype=torch.float32, requires_grad=True).transpose(1, 3).transpose(1, 2)
        high_frames[1][0] = torch.tensor(output, requires_grad=False)
        loss = self.loss_calculate(torch.tensor([target], dtype=torch.float32),
                                   torch.tensor(high_frames, dtype=torch.float32)) if train else None
        return output, loss

    def loss_calculate(self, target, outputs):
        with torch.no_grad():
            genSR_loss = self.SR_loss(outputs[1], target)
            objSR_loss = torch.mean(
                torch.tensor([self.SR_loss(i, j) for i, j in self.loss4object(outputs[:2], target, SR=True)]))
            Flow_loss = self.Flow_loss(outputs)
            objFlow_loss = torch.mean(torch.tensor([self.Flow_loss(output) for output in self.loss4object(outputs)]))
            return torch.tensor([genSR_loss, objSR_loss, Flow_loss, objFlow_loss])
