import torch
from torch import nn
import numpy as np
from my_packages.VOSProjection.VOSProjectionModule import VOSProjectionModule
from utils.models import vgg16
from utils.tools import maskprocess, transpose1323, MakeCuda


class SR_loss(nn.Module):
    def __init__(self):
        super(SR_loss, self).__init__()
        vgg = vgg16()
        loss_network = nn.Sequential(*list(vgg.features)[:31])
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        output = MakeCuda(transpose1323(output))
        target = MakeCuda(transpose1323(target))
        p_output = self.loss_network(output)
        p_target = self.loss_network(target)
        perception_loss = self.mse_loss(p_output, p_target)
        image_loss = self.mse_loss(output, target)
        tv_loss = self.tv_loss(output)
        return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Flow_loss(nn.Module):
    def __init__(self):
        super(Flow_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.SR_loss = SR_loss()

    def forward(self, outputs):
        # TODO-need to test which one is better
        flow_loss = torch.mean(
            torch.stack((self.SR_loss(outputs[0:1], outputs[1:2]), self.SR_loss(outputs[1:2], outputs[2:3]))))
        # flow_loss = torch.mean(self.mse_loss(outputs[0:1], outputs[1:2]), self.mse_loss(outputs[1:2], outputs[2:3])
        return 0.005 * flow_loss


class GetObjectsForOBJLoss(nn.Module):
    def __init__(self):
        super(GetObjectsForOBJLoss, self).__init__()
        self.VOS = VOSProjectionModule().eval().cpu()
        self.mask = None

    def forward(self, outputs, target=None, SR=False):
        if isinstance(self.mask, type(None)):
            obj_segmentation = self.VOS(outputs[0], outputs[1])
            self.mask = maskprocess(obj_segmentation == 1)

        if SR:
            masked_outputs = getSRMaskedOutputs(outputs, target, self.mask)
            return masked_outputs
        else:
            masked_outputs = getFlowMaskedOutputs(outputs, self.mask)
            return masked_outputs


def getSRMaskedOutputs(outputs, target, mask):
    masked_output = torch.unsqueeze(
        torch.tensor(np.ma.MaskedArray(np.array(outputs[1].cpu(), dtype=np.uint8), mask, fill_value=0).filled(),
                     dtype=torch.float32), 0).cuda()
    masked_target = torch.tensor(
        np.ma.MaskedArray(np.array(target.cpu(), dtype=np.uint8), mask, fill_value=0).filled(),
        dtype=torch.float32).cuda()
    masked_object = (masked_output, masked_target)
    return masked_object


def getFlowMaskedOutputs(outputs, mask):
    masked_outputs = []
    for output in outputs:
        masked_output = torch.tensor(
            np.ma.MaskedArray(np.array(output.cpu(), dtype=np.uint8), mask, fill_value=0).filled())
        masked_outputs.append(masked_output)
    masked_object = torch.stack(masked_outputs).type(torch.float32).cuda()
    return masked_object
