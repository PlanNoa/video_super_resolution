import torch
from torch.nn.modules.module import Module
from .vgg_osvos import OSVOS
import numpy as np

class VOSProjectionModule(Module):
    def __init__(self):
        super(VOSProjectionModule, self).__init__()
        self.net = OSVOS()
        self.net.load_state_dict(torch.load('my_packages/VOSProjection/pretrained/parent_epoch-239.pth',
                               map_location=lambda storage, loc: storage))
        self.meanval = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)

    def forward(self, input1, input2):
        # (2, y, x, 3)
        imgs = [input1.cpu().numpy(), input2.cpu().numpy()]
        imgs = np.array(
            [np.subtract(img, self.meanval) for img in imgs])
        imgs = torch.tensor(imgs.transpose((0, 3, 1, 2))).cuda()
        # (2, 3, y, x)

        outputs = self.net.forward(imgs)
        preds = np.transpose(outputs[-1].cpu().data.numpy(), (0, 2, 3, 1))
        # (2, y, x, 3)
        preds = [np.squeeze(1 / (1 + np.exp(-pred))) for pred in preds]
        pred = preds[0] + preds[1]
        pred[pred > 0.7] = 1
        pred[pred <= 0.7] = 0
        pred = torch.tensor(pred)
        return pred