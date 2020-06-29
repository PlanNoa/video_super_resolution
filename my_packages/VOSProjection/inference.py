from __future__ import division
import torch
from torch.autograd import Variable

import torch.nn.functional as F
from utils.object_utils import ToCudaVariable, upsample, downsample

def Encode_MS(val_F1, val_P1, scales, model):
    ref = {}
    for sc in scales:
        if sc != 1.0:
            msv_F1, msv_P1 = downsample([val_F1, val_P1], sc)
            msv_F1, msv_P1 = ToCudaVariable([msv_F1, msv_P1], volatile=True)
            ref[sc] = model.Encoder(msv_F1, msv_P1)[0]
        else:
            msv_F1, msv_P1 = ToCudaVariable([val_F1, val_P1], volatile=True)
            ref[sc] = model.Encoder(msv_F1, msv_P1)[0]

    return ref

def Propagate_MS(ref, val_F2, val_P2, scales, model):
    h, w = val_F2.size()[2], val_F2.size()[3]
    msv_E2 = {}
    for sc in scales:
        if sc != 1.0:
            msv_F2, msv_P2 = downsample([val_F2, val_P2], sc)
            msv_F2, msv_P2 = ToCudaVariable([msv_F2, msv_P2], volatile=True)
            r5, r4, r3, r2 = model.Encoder(msv_F2, msv_P2)
            e2 = model.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = upsample(F.softmax(e2[0], dim=1)[:, 1].data.cpu(), (h, w))
        else:
            msv_F2, msv_P2 = ToCudaVariable([val_F2, val_P2], volatile=True)
            r5, r4, r3, r2 = model.Encoder(msv_F2, msv_P2)
            e2 = model.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = F.softmax(e2[0], dim=1)[:, 1].data.cpu()

    val_E2 = torch.zeros(val_P2.size())
    for sc in scales:
        val_E2 += msv_E2[sc]
    val_E2 /= len(scales)
    return val_E2

def Infer_SO(all_F, all_M, num_frames, model, scales):
    all_E = torch.zeros(all_M.size())
    all_E[:, :, 0] = all_M[:, :, 0]

    ref = Encode_MS(all_F[:, :, 0], all_E[:, 0, 0], scales, model)
    for f in range(0, num_frames - 1):
        all_E[:, 0, f + 1] = Propagate_MS(ref, all_F[:, :, f + 1], all_E[:, 0, f], scales)

    return all_E

def Infer_MO(all_F, all_M, num_frames, num_objects, model, scales):
    if num_objects == 1:
        obj_E = Infer_SO(all_F, all_M, num_frames, model, scales=scales)  # 1,1,t,h,w
        return torch.cat([1 - obj_E, obj_E], dim=1)

    _, n, t, h, w = all_M.size()
    all_E = torch.zeros((1, n + 1, t, h, w))
    all_E[:, 1:, 0] = all_M[:, :, 0]
    all_E[:, 0, 0] = 1 - torch.sum(all_M[:, :, 0], dim=1)

    ref_bg = Encode_MS(all_F[:, :, 0], torch.sum(all_E[:, 1:, 0], dim=1), scales, model)
    refs = []
    for o in range(num_objects):
        refs.append(Encode_MS(all_F[:, :, 0], all_E[:, o + 1, 0], scales, model))

    for f in range(0, num_frames - 1):
        all_E[:, 0, f + 1] = 1 - Propagate_MS(ref_bg, all_F[:, :, f + 1], torch.sum(all_E[:, 1:, f], dim=1), scales, model)
        for o in range(num_objects):
            all_E[:, o + 1, f + 1] = Propagate_MS(refs[o], all_F[:, :, f + 1], all_E[:, o + 1, f], scales, model)

        all_E[:, :, f + 1] = torch.clamp(all_E[:, :, f + 1], 1e-7, 1 - 1e-7)
        all_E[:, :, f + 1] = torch.log((all_E[:, :, f + 1] / (1 - all_E[:, :, f + 1])))
        all_E[:, :, f + 1] = F.softmax(Variable(all_E[:, :, f + 1]), dim=1).data

    return all_E