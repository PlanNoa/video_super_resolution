import torch
from my_packages.DepthProjection import MegaDepth


class DepthProjectionModule():
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.depthNet=MegaDepth.__dict__['HourGlass']("video_super_resolution/my_packages/DepthProjection/best_generalization_net_G.pth")


    def forward(self, input):
        cur_filter_input = torch.from_numpy(input)
        temp = self.depthNet(cur_filter_input)
        log_depth = [temp[:cur_filter_input.size(0)], temp[cur_filter_input.size(0):]]

        cur_ctx_output = [
            torch.cat((self.ctxNet(cur_filter_input[:, :3, ...]),
                       log_depth[0].detach()), dim=1),
            torch.cat((self.ctxNet(cur_filter_input[:, 3:, ...]),
                       log_depth[1].detach()), dim=1)
        ]
        temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
        cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                             self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]
        import cv2
        cv2.imshow(cur_filter_output)