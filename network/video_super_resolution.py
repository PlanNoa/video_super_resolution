import torch
import torch.nn as  nn

from my_packages import DepthProjection
from my_packages import FlowProjection

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        # init model

    def _initialize_weights(self):
        # init pretrained model
        pass

    def forward(self):
        pass

    def forward_flow(self):
        # get optical flow map
        pass

    def forward_depthmap(self):
        # get depth map
        pass

    def video_super_resolution(self):
        # main model
        pass
