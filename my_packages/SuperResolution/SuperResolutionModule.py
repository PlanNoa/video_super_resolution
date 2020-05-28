from torch.nn import Module
from ..DepthProjection.DepthProjectionModule import DepthProjectionModule
from ..FlowProjection.FlowProjectionModule import FlowProjectionModule

# Super Resolution with depth map and optical flow for unnatural objects  in video super resolution

class SuperResolutionModule(Module):
    def __init__(self):
        super(SuperResolutionModule, self).__init__()
        self.FlowModule = FlowProjectionModule().eval()
        self.DepthModule = DepthProjectionModule().eval()

    def forward(self, input, estimated_image=None):
        optical_flow = [self.FlowModule(input[0], input[1]),
                        self.FlowModule(input[1], input[2])]
        depth_map = [self.DepthModule(input[0], input[1]),
                     self.DepthModule(input[1], input[2])]

        if estimated_image:
            # None 아닐때 optical flow, depth map과 그냥 concat
            pass
        else:
            #None 일때 estimated t-1 자리에 VOS(estimated t-1, estimated t+1)
            pass


        return