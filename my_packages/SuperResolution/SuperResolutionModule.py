from torch.nn import Module
from ..DepthProjection.DepthProjectionModule import DepthProjectionModule
from ..FlowProjection.FlowProjectionModule import FlowProjectionModule

# Super Resolution with depth map and optical flow for unnatural objects  in video super resolution

class SuperResolutionModule(Module):
    def __init__(self):
        super(SuperResolutionModule, self).__init__()
        self.FlowModule = FlowProjectionModule()
        self.DepthModule = DepthProjectionModule()

    def forward(self, input1, input2, input3):
        optical_flow = [self.FlowModule(input1, input2),
                        self.FlowModule(input2, input3)]
        depth_map = [self.DepthModule(input1, input2),
                     self.DepthModule(input2, input3)]
        return