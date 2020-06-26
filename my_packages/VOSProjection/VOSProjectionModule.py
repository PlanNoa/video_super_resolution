from torch.nn.modules.module import Module
from .utils import *
from .inference import *
from .model import RGMP
import numpy as np
import torch
from utils.object_utils import *
from PIL import Image

class VOSProjectionModule(Module):
    def __init__(self):
        super(VOSProjectionModule, self).__init__()
        self.model = RGMP()
        self.dict = torch.load('my_packages/VOSProjection/pretrained/weights.pth')
        self.model.load_state_dict({k[7:] if k[:7] == 'module.' else k:self.dict[k] for k in self.dict})
        self.model.eval()

    def preprocess(self, input1, input2, optical_flow=None):
        imgs = [np.array(img, dtype=np.uint8) for img in [input1.cpu(), input2.cpu()]]
        imgs = list(map(Image.fromarray, imgs))

        mask = np.array(imgs[0].convert("P"))
        num_objects = count_objects(optical_flow) if optical_flow else 10
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
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).long(), 0)

        return th_frames, th_masks, num_objects, pad

    def forward(self, input1, input2):
        all_F, all_M, num_objects, pad = self.preprocess(input1, input2)
        # all_F, all_M = all_F, all_M
        all_E = Infer_MO(all_F, all_M, 2, num_objects, self.model, scales=[0.5, 0.75, 1.0])

        E = all_E[0,:,0].numpy()
        E = ToLabel(E)
        (lh, uh), (lw, uw) = pad
        E = E[lh:-uh, lw:-uw]
        # E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        return E