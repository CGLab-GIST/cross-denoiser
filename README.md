# Online Neural Denoising with Cross-Regression for Interactive Rednering

This is PyTorch version of the implementation of paper "Online Neural Denoising with Cross-Regression for Interactive Rednering". As an author, I made this version for my own research purpose. Compared to the original [Tensorflow version](https://github.com/CGLab-GIST/cross-denoiser), it is not tested thoroughly and may have some differences from the original implementation. Definitely, it generates different bandwidths and denoising results.

## Known problems
1. Default memory layout of PyTorch is NCHW, which is not suitable for image processing. This may lead to performance degradation, compared to the TensorFlow with NHWC layout.
2. `torch.compile` is not applied due to the custom operations. It may possible to indicate the operations not to be compiled though.