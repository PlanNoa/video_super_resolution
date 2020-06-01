from torch.nn.modules.module import Module
from .model import MVDepthNet
from numpy.linalg import inv
import cv2
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
from .visualize import np2Depth

class DepthProjectionModule(Module):
    def __init__(self):
        super(DepthProjectionModule, self).__init__()
        self.net = MVDepthNet()
        self.dict = torch.load("my_packages/DepthProjection/pretrained/opensource_model.pth.tar")
        self.net.load_state_dict(self.dict['state_dict'])

    def getVolume(self, left_image, right_image, KRKiUV_T, KT_T):
        idepth_base = 1.0 / 50.0
        idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0

        costvolume = Variable(
            torch.cuda.FloatTensor(left_image.shape[0], 64,
                                   left_image.shape[2], left_image.shape[3]))
        image_height = 256
        image_width = 320
        batch_number = left_image.shape[0]

        normalize_base = torch.cuda.FloatTensor([image_width / 2.0, image_height / 2.0])
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)

        for depth_i in range(64):
            this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
            transformed = KRKiUV_T * this_depth + KT_T
            demon = transformed[:, 2, :].unsqueeze(1)
            warp_uv = transformed[:, 0: 2, :] / (demon + 1e-6)
            warp_uv = (warp_uv - normalize_base) / normalize_base
            warp_uv = warp_uv.view(batch_number, 2, image_width, image_height)

            warp_uv = Variable(warp_uv.permute(0, 3, 2, 1))
            warped = F.grid_sample(right_image, warp_uv)

            costvolume[:, depth_i, :, :] = torch.sum(torch.abs(warped - left_image), dim=1)
        return costvolume

    def getValue(self, left_image, right_image):
        pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        pixel_coordinate = np.concatenate(
            (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        left_pose = np.asarray([
            [0.07543147, 0.61393189, -0.78574661, 1.3405],
            [0.9970987, -0.03837025, 0.06574118, 0.6266],
            [0.01021131, -0.78842588, -0.61504501, 1.6575],
            [0, 0, 0, 1]])

        right_pose = np.asarray(
            [[6.40527011e-02, 6.40832173e-01, -7.65004168e-01, 1.3160],
             [9.97946496e-01, -4.09736058e-02, 4.92336713e-02, 0.6254],
             [2.05541383e-04, -7.66586779e-01, -6.42140692e-01, 1.6196],
             [0, 0, 0, 1]])

        camera_k = np.asarray([[525.0, 0, 319.5],
                               [0, 525.0, 239.5],
                               [0, 0, 1]])

        left2right = np.dot(inv(right_pose), left_pose)
        test_point = np.asarray([left_image.shape[1] / 2, left_image.shape[0] / 2, 1])
        far_point = np.dot(inv(camera_k), test_point) * 50.0
        far_point = np.append(far_point, 1)
        far_point = np.dot(left2right, far_point)
        far_pixel = np.dot(camera_k, far_point[0:3])
        far_pixel = (far_pixel / far_pixel[2])[0:2]
        near_point = np.dot(inv(camera_k), test_point) * 0.1
        near_point = np.append(near_point, 1)
        near_point = np.dot(left2right, near_point)
        near_pixel = np.dot(camera_k, near_point[0:3])
        near_pixel = (near_pixel / near_pixel[2])[0:2]
        cv2.line(right_image,
                 (int(far_pixel[0] + 0.5), int(far_pixel[1] + 0.5)),
                 (int(near_pixel[0] + 0.5), int(near_pixel[1] + 0.5)), [0, 0, 255], 4)
        cv2.circle(left_image, (int(test_point[0]), int(test_point[1])), 4, [0, 0, 255], -1)

        original_width = left_image.shape[1]
        original_height = left_image.shape[0]
        factor_x = 320.0 / original_width
        factor_y = 256.0 / original_height

        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        camera_k[0, :] *= factor_x
        camera_k[1, :] *= factor_y

        torch_left_image = np.moveaxis(left_image, -1, 0)
        torch_left_image = np.expand_dims(torch_left_image, 0)
        torch_left_image = (torch_left_image - 81.0) / 35.0
        torch_right_image = np.moveaxis(right_image, -1, 0)
        torch_right_image = np.expand_dims(torch_right_image, 0)
        torch_right_image = (torch_right_image - 81.0) / 35.0

        left_image_cuda = Tensor(torch_left_image).cuda()
        left_image_cuda = Variable(left_image_cuda, volatile=True)

        right_image_cuda = Tensor(torch_right_image).cuda()
        right_image_cuda = Variable(right_image_cuda, volatile=True)

        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K = camera_k
        K_inverse = inv(K)
        KRK_i = K.dot(left_in_right_R.dot(K_inverse))
        KRKiUV = KRK_i.dot(pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)
        KRKiUV_cuda_T = Tensor(KRKiUV).cuda()
        KT_cuda_T = Tensor(KT).cuda()

        return left_image_cuda, right_image_cuda, KRKiUV_cuda_T, KT_cuda_T

    def forward(self, input1, input2):
        input1, input2, KRKiUV_T, KT_T = self.getValue(input1, input2)
        plane_sweep_volume = self.getVolume(input1, input2, KRKiUV_T, KT_T)
        input = torch.cat((input1, plane_sweep_volume), 1)
        result = self.net(input)
        result = np.squeeze(result[0].cpu().data.numpy())
        result = np2Depth(result, np.zeros(result.shape, dtype=bool))
        return result