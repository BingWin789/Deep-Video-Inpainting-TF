<a href="https://github.com/mcahny/Deep-Video-Inpainting" target="_blank">This page</a> is official **Deep video inpainting** pytorch version. From there, you can know how to configure environment and download pretrained pth weights.

---

Firstly, download pretrained `.pth` file to `./weights/pytorch`;

Secondly, run 

`python pth_to_ckpt.py --gpuid=0 --pthpath=./weights/pytorch/.pth --ckptpath=./weights/tensorflow/.ckpt`

then you can get pretrained tensorflow weights from pth.
