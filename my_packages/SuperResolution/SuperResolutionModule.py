from torch.nn import Module
from ..DepthProjection.DepthProjectionModule import DepthProjectionModule
from ..FlowProjection.FlowProjectionModule import FlowProjectionModule

# Super Resolution with depth map and optical flow for unnatural objects  in video super resolution

class SuperResolutionModule(Module):
    def __init__(self):
        super(SuperResolutionModule, self).__init__()
        self.FlowModule = FlowProjectionModule()
        self.DepthModule = DepthProjectionModule()

    def forward(self, input):
        optical_flow = [self.FlowModule(input[0], input[1]),
                        self.FlowModule(input[1], input[2])]
        depth_map = [self.DepthModule(input[0], input[1]),
                     self.DepthModule(input[1], input[2])]


        return