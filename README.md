# Deep Video Inpainting - TF
This is the Tensorflow implementation for "Deep Video Inpainting" (CVPR 2019) (NOT official)

### Installation
---
The code is tested under Python 3.5.2, Tensorflow 1.8.0, CUDA 10 and cuDNN 7.1.4.

Run `make all` to compile correlation ops.

Read [this](./weights/README.md) to download pretrained .pth weights and convert them to .ckpt.

### Test
---
Run `python demo_vi.py --gpuid=0`, and example results will be saved in **./examples/results**.

### Acknowledgement
---
**correlation ops** is copied from <a href="https://github.com/sampepose/flownet2-tf" target="_blank">flownet2-tf</a>.

Knowledge and mind are power. Thank you.
