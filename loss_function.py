import torch
from torch import nn
from torchvision.models.vgg import vgg16

# need to change for polygon shape
class SR_loss(nn.Module):
    def __init__(self):
        super(SR_loss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        perception_loss = self.mse_loss(self.loss_network(output), self.loss_network(target))
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

    def forward(self, output1, output2, output3):
        flow_loss_1 = self.mse_loss(output1, output2)
        flow_loss_2 = self.mse_loss(output2, output3)
        return (flow_loss_1 + flow_loss_2)/2