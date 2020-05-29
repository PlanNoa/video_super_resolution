import torch
import numpy as np
from .models import FlowNet2
from utils.frame_utils import read_gen
from utils.tools import StaticCenterCrop

if __name__ == '__main__':
    net = FlowNet2().cuda()
    dict = torch.load("pretrained/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    img1 = read_gen("data/MPI-Sintel-testing/test/clean/testfortwo/frame_0001.png")
    img2 = read_gen("data/MPI-Sintel-testing/test/clean/testfortwo/frame_0002.png")

    images = [img1, img2]
    image_size = img1.shape[:2]

    frame_size = img1.shape
    render_size = [((frame_size[0])//64 )*64, ((frame_size[1])//64)*64]

    cropper = StaticCenterCrop(image_size, render_size)
    images = list(map(cropper, images))

    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))
    images = images.unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(images).squeeze()

    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("resultfortwo/a.flo", data)