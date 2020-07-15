import torch
import numpy as np
from torch.nn.functional import interpolate
from utils.tools import maskprocess, transpose1323, transpose1223, transpose1312, transpose1201
from loss_function import SR_loss, Flow_loss, GetObjectsForOBJLoss
from my_packages.SRProjection.SRProjectionModule import SRProjectionModule
from my_packages.DepthProjection.DepthProjectionModule import DepthProjectionModule
from my_packages.FlowProjection.FlowProjectionModule import FlowProjectionModule
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule


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
            data_shape = (data.shape[1], data.shape[2])

            data_clone = data.clone()
            optical_flow = torch.stack([self.FlowModule(data_clone[0], data_clone[1]),
                                        self.FlowModule(data_clone[1], data_clone[2])])
            depth_map = torch.stack([maskprocess(self.DepthModule(data_clone[0:2])),
                                     maskprocess(self.DepthModule(data_clone[1:3]))])

            data_clone = transpose1323(data_clone)

            optical_flow = interpolate(transpose1323(optical_flow), data_shape)

            estimated_image = interpolate(transpose1323(estimated_image), data_shape) \
                if not isinstance(estimated_image, type(None)) else data_clone[0:1]

            input = torch.cat((data_clone, optical_flow, depth_map, estimated_image), 0)
            output = self.model(input).cuda()
            # output shape: [1, 3, y, x]


            data_clone = transpose1223(torch.stack([estimated_image[0],
                                                    interpolate(output, data_shape)[0], data_clone[2]]))

            optical_flow = torch.stack([self.FlowModule(data_clone[0], data_clone[1]),
                                        self.FlowModule(data_clone[1], data_clone[2])])

            depth_map = torch.stack([maskprocess(self.DepthModule(data_clone[0:2])),
                                     maskprocess(self.DepthModule(data_clone[1:3]))])

            optical_flow = interpolate(transpose1323(optical_flow), data_shape)

            VOSmask = maskprocess(self.VOSModule(data_clone[0], data_clone[1]))


        data = transpose1323(data)
        masked_estimated_image = torch.tensor([np.ma.MaskedArray(transpose1201(data_clone[1]).cpu().numpy(),
                                                                 VOSmask, fill_value=0).filled()],
                                              dtype=torch.float32, requires_grad=False).cuda()

        input = torch.cat((data, optical_flow, depth_map, masked_estimated_image), 0)

        output = transpose1312(self.model(input))

        high_frames[1] = output
        loss = self.loss_calculate(target, high_frames) if train else None

        return output, loss

    def loss_calculate(self, target, outputs):
        with torch.no_grad():
            genSR_loss = self.SR_loss(outputs[0:1], target)
            masked_object = self.loss4object(outputs[:2], target, SR=True)
            objSR_loss = self.SR_loss(masked_object[0], masked_object[1])
            genFlow_loss = self.Flow_loss(outputs)
            masked_object = self.loss4object(outputs)
            objFlow_loss = self.Flow_loss(masked_object)
            return torch.tensor([genSR_loss, objSR_loss, genFlow_loss, objFlow_loss])