import torch
from torch.nn.modules.module import Module
from utils.tools import StaticCenterCrop
from utils.flow_utils import flow2img
from .models import FlowNet2


class FlowProjectionModule(Module):
    def __init__(self, image_size=None, render_size=None):
        super(FlowProjectionModule, self).__init__()
        self.cropper = None
        self.net = FlowNet2()
        self.dict = torch.load("my_packages/FlowProjection/pretrained/FlowNet2_checkpoint.pth.tar")
        self.net.load_state_dict(self.dict["state_dict"])
        self.image_size = image_size
        self.render_size = render_size

    def forward(self, input1, input2):
        if self.cropper is None or self.image_size != input1.shape[:2] or \
                self.render_size != [((input1.shape[0]) // 64) * 64, ((input1.shape[1]) // 64) * 64]:
            self.image_size = input1.shape[:2]
            self.render_size = [((input1.shape[0]) // 64) * 64, ((input1.shape[1]) // 64) * 64]
            self.cropper = StaticCenterCrop(self.image_size, self.render_size)
        images = [input1, input2]
        images = list(map(self.cropper, images))

        images = torch.stack(images).transpose(0, 1).transpose(0, 2).transpose(0, 3)
        images = images.unsqueeze(0)

        result = self.net(images).squeeze()
        flow = result.transpose(1, 2).transpose(0, 2)
        result = torch.tensor(flow2img(flow.cpu().numpy()), dtype=torch.float32).cuda()
        return result
