import cv2
import torch
import numpy as np
from torch.autograd import Variable


def count_objects(optical_flow):
    # TODO-count moving objects from optical flow
    return optical_flow


def ToCudaVariable(xs):
    if torch.cuda.is_available():
        return [Variable(x.cuda()) for x in xs]
    else:
        return [Variable(x) for x in xs]


def upsample(x, size):
    x = x.numpy()[0]
    dsize = (size[1], size[0])
    x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return torch.unsqueeze(torch.from_numpy(x), dim=0)


def downsample(xs, scale):
    if scale == 1:
        return xs

    # find new size dividable by 32
    h = xs[0].size()[2]
    w = xs[0].size()[3]

    new_h = int(h * scale)
    new_w = int(w * scale)
    new_h = new_h + 32 - new_h % 32
    new_w = new_w + 32 - new_w % 32

    dsize = (new_w, new_h)
    ys = []
    for x in xs:
        x = x.numpy()[0]  # c,h,w
        if x.ndim == 3:
            x = np.transpose(x, [1, 2, 0])
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            x = np.transpose(x, [2, 0, 1])
        else:
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)

        ys.append(torch.unsqueeze(torch.from_numpy(x), dim=0))
    return ys