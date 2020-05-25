## Video Super Resolution with depth map and optical flow for unnatural object flow

### Related Repositories
- [Flownet2](https://github.com/NVIDIA/flownet2-pytorch)
- [MVDepthmap](https://github.com/HKUST-Aerial-Robotics/MVDepthNet)
- [RGMP](https://github.com/seoungwugoh/RGMP)

### Main Goal
1. optical flow net(flownet 2.0) Module(05.05 Done)
2. video object segmentation net(RGMP) Module(05.20 Done)
3. depth map net(MVDepthNet) Module(05.20 Start)
4. super resolution net
5. loss function with etc utils(05.23 Start)
6. training
7. testing

### TO-DO
- Inference function on main.py
- count_object function on object_utils.py
- low_resolution function on frame_utils.py
- define new way to calculate flow-loss
- MVDepthmap module programming
- Main VSR model programming