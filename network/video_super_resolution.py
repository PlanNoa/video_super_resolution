import torch
import numpy as np
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from loss_function import _SR_loss, _Flow_loss, _loss4object
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.tools import up_scailing, down_scailing, get_gpu_usage, maskprocess
from PIL import Image
from torch.nn.functional import interpolate

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
            optical_flow = torch.stack([self.FlowModule(data[0].clone(), data[1].clone()),
                                        self.FlowModule(data[1].clone(), data[2].clone())])
            # space for new depth map net
            depth_map = torch.stack([self.FlowModule(data[0].clone(), data[1].clone()),
                                     self.FlowModule(data[1].clone(), data[2].clone())])
            # depth_map = [self.DepthModule(data[0].clone(), data[1].clone()),
                        #  self.DepthModule(data[1].clone(), data[2].clone())]
            data = data.transpose(1, 3).transpose(2, 3)
            optical_flow = interpolate(optical_flow.transpose(1, 3).transpose(2, 3), data.shape[2:])
            depth_map = interpolate(depth_map.transpose(1, 3).transpose(2, 3), data.shape[2:])
            estimated_image = interpolate(estimated_image.transpose(1, 3).transpose(2, 3), data.shape[2:]) if type(estimated_image) != type(None) else data[0:1]
            input = torch.cat((data, optical_flow, depth_map, estimated_image), 0)
            # input shape: [8, 3, y/4, x/4]

            output = self.model(input).cuda()
            # output shape: [1, 3, y, x]

            data = torch.stack([estimated_image[0], interpolate(output, data.shape[2:])[0], data[2]])
            optical_flow = torch.stack([self.FlowModule(data[0].clone().transpose(0, 1).transpose(1, 2),
                                                        data[1].clone().transpose(0, 1).transpose(1, 2)),
                                        self.FlowModule(data[1].clone().transpose(0, 1).transpose(1, 2),
                                                        data[2].clone().transpose(0, 1).transpose(1, 2))])
            # space for new depth map net
            depth_map = torch.stack([self.FlowModule(data[0].clone().transpose(0, 1).transpose(1, 2),
                                                     data[1].clone().transpose(0, 1).transpose(1, 2)),
                                     self.FlowModule(data[1].clone().transpose(0, 1).transpose(1, 2),
                                                     data[2].clone().transpose(0, 1).transpose(1, 2))])
            # depth_map = [self.DepthModule(data[0].clone(), data[1].clone()),
                        #  self.DepthModule(data[1].clone(), data[2].clone())]
            optical_flow = interpolate(optical_flow.transpose(1, 3).transpose(2, 3), data.shape[2:])
            depth_map = interpolate(depth_map.transpose(1, 3).transpose(2, 3), data.shape[2:])
            VOSmask = maskprocess(self.VOSModule(data[0].transpose(0, 1).transpose(1, 2), data[1].transpose(0, 1).transpose(1, 2)) != 0)

        masked_estimated_image = torch.tensor([np.ma.MaskedArray(data[1].cpu().numpy(), VOSmask, fill_value=0).filled()], dtype=torch.float32, requires_grad=True).cuda()
        masked_estimated_image = interpolate(masked_estimated_image, data.shape[2:])
        data.requires_grad = True
        input = torch.cat((data, optical_flow, depth_map, masked_estimated_image), 0)
        # input shape:[8, 3, y/4, x/4]

        output = self.modmodelel(input)
        # output shape: [1, 3, y, x]

        output = torch.tensor(output, dtype=torch.float32, requires_grad=True).transpose(1, 3).transpose(1, 2).cuda()
        high_frames[1] = torch.tensor(output, requires_grad=False)
        loss = self.loss_calculate(target, high_frames) if train else None
        return output, loss

    def loss_calculate(self, target, outputs):
        with torch.no_grad():
            genSR_loss = self.SR_loss(outputs[0:1], target)
            objSR_loss = torch.mean(torch.tensor([self.SR_loss(i, j) for i, j in self.loss4object(outputs[:2], target, SR=True)]))
            Flow_loss = self.Flow_loss(outputs)
            objFlow_loss = torch.mean(torch.tensor([self.Flow_loss(output) for output in self.loss4object(outputs)]))
            return torch.tensor([genSR_loss, objSR_loss, Flow_loss, objFlow_loss])
