from torch.nn.modules.module import Module
from .utils import *
from .inference import *
from .model import RGMP
import numpy as np
import torch

class VOSProjectionModule(Module):
    def __init__(self):
        super(VOSProjectionModule, self).__init__()
        self.model = RGMP().cuda()
        self.model.load_state_dict(torch.load('pretrained/weights.pth'))
        self.model.eval()

    def preprocess(self, input1, input2):
        imgs = [input1, input2]

        mask = np.array(input1.convert("P"))
        # need to get moving object number from flownet module
        num_objects = np.max(mask)
        shape = np.shape(mask)

        raw_frames = np.empty((2,)+shape+(3,), dtype=np.float32)
        raw_masks = np.empty((2,)+shape, dtype=np.uint8)

        for i, img in enumerate(imgs):
            raw_frames[i] = np.array(img.convert("RGB"))/255
            raw_masks[i] = np.array(img.convert("P"), dtype=np.uint8)

        oh_masks = np.zeros((2,)+shape+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)

        nf, h, w, _ = oh_masks.shape
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        pad_masks = np.pad(oh_masks, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad_frames = np.pad(raw_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad = ((lh, uh), (lw, uw))

        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (3, 0, 1, 2)).copy()).float(), 0)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).float(), 0)

        return th_frames, th_masks, num_objects, pad

    def forward(self, input1, input2):
        all_F, all_M, num_objects, pad = self.preprocess(input1, input2)
        all_F, all_M = all_F[0], all_M[0]
        all_E = Infer_MO(all_F, all_M, 2, num_objects, self.model, scales=[0.5, 0.75, 1.0])

        for f in range(2):
            E = all_E[0,:,f].numpy()
            E = ToLabel(E)
            (lh, uh), (lw, uw) = pad
            E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        return E