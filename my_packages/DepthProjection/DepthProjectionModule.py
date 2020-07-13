import torch
import torch.nn as nn
from my_packages.DepthProjection.MegaDepth.models.models import create_model
from my_packages.DepthProjection.MegaDepth.options.train_options import TrainOptions

opt = TrainOptions().parse()


class DepthProjectionModule(nn.Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()

        self.model = create_model(opt)
        self.model.switch_to_eval()

    def forward(self, input):

        p = self.model.inference(input)

        p = torch.squeeze(p[0])

        import matplotlib.pyplot as plt
        p = p.detach().numpy()
        plt.imshow(p)
        plt.savefig('a.png')
        plt.show()

        input = input.float()
        input_0, input_2 = torch.squeeze(input, dim=0)
        print(input_0.size(), input_2.size())
        cur_input_0 = torch.unsqueeze(input_0, 0)
        cur_input_2 = torch.unsqueeze(input_2, 0)
        cur_offset_input = torch.cat([cur_input_0, cur_input_2], dim=1)
        cur_filter_input = cur_offset_input

        print(cur_filter_input[:, :3, ...].size(), cur_filter_input[:, 3:, ...].size())
        temp = self.depthNet(torch.cat((cur_filter_input[:, :3, ...],
                                        cur_filter_input[:, 3:, ...]), dim=0))

        temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
        cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                             self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]

        cur_filter_output = torch.squeeze(cur_filter_output[0])

    def get_MonoNet5(self, channel_in, channel_out, name):
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32, 16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))

        return (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:

            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers, nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp, stack.pop()),
                                         dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                       padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding)
        )
        return layers

    '''keep this fucntion'''

    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                  padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False)
        ])
        return layers


    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                          padding, kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers


    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                         padding, unpooling_factor):

        layers = nn.Sequential(*[
            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear'),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
        ])
        return layers


class Stack:
    def __init__(self):
        self.stack = []

    def pop(self):
        if self.is_empty():
            return None
        else:
            return self.stack.pop()

    def push(self, val):
        return self.stack.append(val)

    def peak(self):
        if self.is_empty():
            return None
        else:
            return self.stack[-1]

    def size(self):
        return len(self.stack)

    def is_empty(self):
        return self.size() == 0
