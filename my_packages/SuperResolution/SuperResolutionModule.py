from torch.nn import Module

# Super Resolution with depth map and optical flow for unnatural objects  in video super resolution

class SuperResolutionModule(Module):
    def __init__(self):
        super(SuperResolutionModule, self).__init__()

    def forward(self, input):
        return