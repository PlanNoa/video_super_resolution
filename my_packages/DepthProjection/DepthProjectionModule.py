import torch
from my_packages.DepthProjection import MegaDepth


class DepthProjectionModule():
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.depthNet=MegaDepth.__dict__['HourGlass']("video_super_resolution/my_packages/DepthProjection/best_generalization_net_G.pth")


    def forward(self, input):

        input = torch.from_numpy(input)


        input_0, input_2 = torch.squeeze(input, dim=0)
        cur_input_0 = input_0
        cur_input_2 = input_2
        cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
        cur_filter_input = cur_offset_input

        print(cur_filter_input.size())
        temp = self.depthNet(torch.cat((cur_filter_input[:, :3, ...],
                                        cur_filter_input[:, 3:, ...]), dim=0))
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