import MegaDepth

class DepthProjectionModule():
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.depthNet=MegaDepth.__dict__['HourGlass']("best_generalization_net_G.pth")


    def forward(self):
        pass

aa = DepthProjectionModule()