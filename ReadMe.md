# Video Super Resolution with depth map and optical flow for unnatural object flow

### Related Repositories
- [Flownet2](https://github.com/NVIDIA/flownet2-pytorch)
- ~[MVDepthNet](https://github.com/HKUST-Aerial-Robotics/MVDepthNet)~ [MegaDepth](https://github.com/baowenbo/DAIN/tree/master/MegaDepth)
- [OSVOS](https://github.com/kmaninis/OSVOS-PyTorch)
- [SRFBN](https://github.com/Paper99/SRFBN_CVPR19)

### Main Goal
1. optical flow net Module(05.05 Done)
2. video object segmentation net Module(05.20 Done)
3. depth map net Modkoule(05.20 Start)
4. super resolution net(ToDo)
5. loss function with etc utils(05.23 Start)
6. training(ToDo)
7. testing(ToDo)

### TODO
- srfbn memory problem.(srfbn get (n, 3, y/2, x/2) size input, but our model use t-1 estimated image with other inputs. need to downsample t-1 estimated image without data loss. or we can change srfbn output) 
- #####  define new depthnet module
- #####  define new way to calculate flow-loss
- Inference function on main.py
- etc..

#### Copyright

- The code and development ideas for this project are belonging to PlanNoa and Yudonggeun.
