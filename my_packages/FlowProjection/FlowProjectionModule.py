from torch.nn.modules.module import Module
from utils.tools import StaticCenterCrop
from .models import FlowNet2
import torch
import numpy as np

class FlowProjectionModule(Module):
    def __init__(self, image_size, render_size):
        super(FlowProjectionModule, self).__init__()
        self.net = FlowNet2().cuda()
        self.cropper = StaticCenterCrop(image_size, render_size)
        self.dict = torch.load("pretrained/FlowNet2_checkpoint.pth.tar")
        self.net.load_state_dict(self.dict["state_dict"])

    def forward(self, input1, input2):
        images = [input1, input2]
        images = list(map(self.cropper, images))

        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))
        images = images.unsqueeze(0).cuda()

        result = self.net(images).squeeze()
        return result