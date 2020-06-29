import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import SR_loss, Flow_loss, GetObjectsForOBJLoss
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.tools import maskprocess
from PIL import Image
from torch.nn.functional import interpolate

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        self.model = SRProjectionModule()
        self.FlowModule = FlowProjectionModule().eval()
        self.DepthModule = DepthProjectionModule().eval()
        self.VOSModule = VOSProjectionModule().eval()
        self.SR_loss = SR_loss().eval()
        self.Flow_loss = Flow_loss().eval()
        self.loss4object = GetObjectsForOBJLoss().eval()

    def forward(self, data, target, high_frames, estimated_image, train=True):
        with torch.no_grad():
            optical_flow = torch.stack([self.FlowModule(data[0].clone(), data[1].clone()),
                                        self.FlowModule(data[1].clone(), data[2].clone())])
            # space for new depth map net
            depth_map = torch.stack([self.FlowModule(data[0].clone(), data[1].clone()),
                                     self.FlowModule(data[1].clone(), data[2].clone())])
            # depth_map = [self.DepthModule(data[0].clone(), data[1].clone()),
                        #  self.DepthModule(data[1].clone(), data[2].clone())]
            data_1 = data.clone().transpose(1, 3).transpose(2, 3)
            optical_flow = interpolate(optical_flow.transpose(1, 3).transpose(2, 3), data_1.shape[2:])
            depth_map = interpolate(depth_map.transpose(1, 3).transpose(2, 3), data_1.shape[2:])
            estimated_image = interpolate(estimated_image.transpose(1, 3).transpose(2, 3), data_1.shape[2:]) if type(estimated_image) != type(None) else data_1[0:1]
            input = torch.cat((data_1, optical_flow, depth_map, estimated_image), 0)
            # input shape: [8, 3, y/4, x/4]

            output = self.model(input).cuda()
            # output shape: [1, 3, y, x]

            data_2 = torch.stack([estimated_image[0], interpolate(output, data_1.shape[2:])[0], data_1[2].clone()])
            optical_flow = torch.stack([self.FlowModule(data_2[0].clone().transpose(0, 1).transpose(1, 2),
                                                        data_2[1].clone().transpose(0, 1).transpose(1, 2)),
                                        self.FlowModule(data_2[1].clone().transpose(0, 1).transpose(1, 2),
                                                        data_2[2].clone().transpose(0, 1).transpose(1, 2))])
            # space for new depth map net
            depth_map = torch.stack([self.FlowModule(data_2[0].clone().transpose(0, 1).transpose(1, 2),
                                                     data_2[1].clone().transpose(0, 1).transpose(1, 2)),
                                     self.FlowModule(data_2[1].clone().transpose(0, 1).transpose(1, 2),
                                                     data_2[2].clone().transpose(0, 1).transpose(1, 2))])
            # depth_map = [self.DepthModule(data[0].clone(), data[1].clone()),
                        #  self.DepthModule(data[1].clone(), data[2].clone())]
            optical_flow = interpolate(optical_flow.transpose(1, 3).transpose(2, 3), data_2.shape[2:])
            depth_map = interpolate(depth_map.transpose(1, 3).transpose(2, 3), data_2.shape[2:])
            VOSmask = maskprocess(self.VOSModule(data_2[0].clone().transpose(0, 1).transpose(1, 2), data_2[1].clone().transpose(0, 1).transpose(1, 2)) != 0)

        data = data.transpose(1, 3).transpose(2, 3)
        masked_estimated_image = torch.tensor([np.ma.MaskedArray(data_2[1].cpu().numpy(), VOSmask, fill_value=0).filled()], dtype=torch.float32, requires_grad=False).cuda()
        masked_estimated_image = interpolate(masked_estimated_image, data_2.shape[2:])
        input = torch.cat((data, optical_flow, depth_map, masked_estimated_image), 0)

        output = self.model(input).transpose(1, 3).transpose(1, 2)

        high_frames[1] = torch.tensor(output, requires_grad=False)
        with torch.no_grad(): loss = self.loss_calculate(target, high_frames) if train else None
        return output, loss

    def loss_calculate(self, target, outputs):
        genSR_loss = self.SR_loss(outputs[0:1], target)
        objSR_loss = torch.mean(torch.tensor([self.SR_loss(i, j) for i, j in self.loss4object(outputs[:2], target, SR=True)]))
        genFlow_loss = self.Flow_loss(outputs)
        objFlow_loss = torch.mean(torch.tensor([self.Flow_loss(output) for output in self.loss4object(outputs)]))
        return torch.tensor([genSR_loss, objSR_loss, genFlow_loss, objFlow_loss])
