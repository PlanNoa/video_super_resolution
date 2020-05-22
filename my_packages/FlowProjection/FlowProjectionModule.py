from torch.nn.modules.module import Module
from utils.tools import StaticCenterCrop
from .models import FlowNet2
import torch
import numpy as np

class FlowProjectionModule(Module):
    def __init__(self, image_size=None, render_size=None):
        super(FlowProjectionModule, self).__init__()
        self.cropper = None
        self.net = FlowNet2().cuda()
        self.dict = torch.load("pretrained/FlowNet2_checkpoint.pth.tar")
        self.net.load_state_dict(self.dict["state_dict"])
        self.image_size = image_size
        self.render_size = render_size
        # if image_size != None or render_size != None:
        #     self.cropper = StaticCenterCrop(image_size, render_size)
        # else:
        #     self.cropper = None

    def forward(self, input1, input2):
        if self.cropper == None or self.image_size != input1.shape[:2] or \
           self.render_size != [((input1.shape[0])//64 )*64, ((input1.shape[1])//64)*64]:
            self.image_size = input1.shape[:2]
            self.render_size = [((input1.shape[0])//64 )*64, ((input1.shape[1])//64)*64]
            self.cropper = StaticCenterCrop(self.image_size, self.render_size)
        images = [input1, input2]
        images = list(map(self.cropper, images))

        images = np.array(images).transpose(3, 0, 1, 2)
        images = torch.from_numpy(images.astype(np.float32))
        images = images.unsqueeze(0).cuda()

        result = self.net(images).squeeze()
        return result