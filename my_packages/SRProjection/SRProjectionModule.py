import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift
from utils.tools import transpose030112, transpose031323


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        stride = 4
        padding = 2
        kernel_size = 8
        self.num_groups = num_groups
        self.compress_in = ConvBlock(2 * num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()
        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features * (idx + 1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features * (idx + 1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.num_features = num_features

        self.compress_out = ConvBlock(num_groups * num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)
        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)
        lr_features = []
        hr_features = []
        lr_features.append(x)
        for idx in range(self.num_groups):
            lr_feature_size = list(lr_features[0].shape)
            lr_feature_size[1] *= (idx + 1)
            LD_L = torch.empty(*lr_feature_size)
            for i in range(idx):
                LD_L[:, self.num_features * idx:self.num_features * (idx + 1), :, :] = lr_features[i]

            LD_L = LD_L.cuda()
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)
            hr_features.append(LD_H)

            # LD_L.cpu()
            torch.cuda.empty_cache()

            hr_feature_size = list(hr_features[0].shape)
            hr_feature_size[1] *= (idx + 1)
            LD_H = torch.empty(*hr_feature_size)
            for i in range(idx):
                LD_H[:, self.num_features * idx:self.num_features * (idx + 1), :, :] = hr_features[i]

            LD_H = LD_H.cuda()
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)
            lr_features.append(LD_L)

            # LD_H.cpu()
            torch.cuda.empty_cache()

        del hr_features

        output = torch.cat(tuple(lr_features[1:]), 1)  # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)
        self.last_hidden = output
        return output

    def reset_state(self):
        self.should_reset = True


class SRProjectionModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=32, upscale_factor=4, num_steps=3, num_groups=6,
                 act_type='prelu', norm_type=None):
        super(SRProjectionModule, self).__init__()

        stride = 4
        padding = 2
        kernel_size = 8

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)
        self.block = FeedbackBlock(num_features, num_groups, act_type, norm_type)
        self.out = DeconvBlock(num_features, num_features,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        self.fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, x):
        import time
        self._reset_state()
        x = self.sub_mean(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        x = self.conv_in(x)
        x = self.feat_in(x)
        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            h = torch.add(inter_res, self.conv_out(self.out(h)))
            h = self.add_mean(h)
            outs.append(h)
        outs = outs[-1:]
        outs = torch.stack([transpose031323(self.fc(transpose030112(out))).squeeze() for out in outs], 0)
        return outs

    def _reset_state(self):
        self.block.reset_state()