import os
import time
import torch
from torch.utils.cpp_extension import load

my_env = os.environ.copy()
my_env["PATH"] = "/opt/conda/bin:" + my_env["PATH"]
my_env["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
os.environ.update(my_env)


def init():
    global ops
    # Count time for loading custom ops
    start = time.time()
    print("Loading custom ops... (takes several minutes for the first run)")
    ops = load(name="custom_ops", sources=[
        "ops/binding.cpp",
        "ops/regression.cu",
        "ops/reproject.cu",
        ], verbose=False)
    end = time.time()
    print(f"\t Took {end - start:.1f} s")

class CrossRegression(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color, color2, albedo, normal, win_size, sparsity=1, reconScale=1):
        out_imga, out_imgb, out_vara, out_varb = ops.cross_regression(color, color2, albedo, normal, win_size, sparsity, reconScale)
        return out_imga, out_imgb, out_vara, out_varb

class Reproject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mvec, linearZ, prevLinearZ, normal, prevNormal, pnFwidth, opacity, prevOpacity):
        success, *output = ops.reproject(input, mvec, linearZ, prevLinearZ, normal, prevNormal, pnFwidth, opacity, prevOpacity)
        return success, output

class TemporalScreening(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current1, current2, prev, prev1, prev2, reproj_success, win_curr, win_prev):
        out, out1, out2, success = ops.temporal_screening(current1, current2, prev, prev1, prev2, reproj_success, win_curr, win_prev)
        return out, out1, out2, success

class Erosion2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, win_size):
        output = ops.erosion2d(input, win_size)
        return output

class SpatialFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, imgA, imgB, albedo, normal, band, win_size):
        out_acca, out_accb, out_wgta, out_wgtb, grad_banda, grad_bandb = ops.spatial_filter_forward(imgA, imgB, albedo, normal, band, win_size)
        ctx.save_for_backward(band, grad_banda, grad_bandb)
        ctx.win_size = win_size
        return out_acca, out_accb, out_wgta, out_wgtb

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_acca, grad_accb, grad_wgta, grad_wgtb):
        band, grad_banda, grad_bandb = ctx.saved_tensors
        grad_output = ops.spatial_filter_backward(grad_acca, grad_accb, grad_wgta, grad_wgtb, band, grad_banda, grad_bandb, ctx.win_size)
        return None, None, None, None, grad_output, None
