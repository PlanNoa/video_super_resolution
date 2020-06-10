from my_packages.DepthProjection import MegaDepth


class DepthProjectionModule():
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.depthNet=MegaDepth.__dict__['HourGlass']("video_super_resolution/my_packages/DepthProjection/best_generalization_net_G.pth")


    def forward(self):
        pass

