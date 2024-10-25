"""
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import tensorflow as tf

####################################################################################################
_module_regression = tf.load_op_library('./ops/regression.so')
spatiotemporal_filter = _module_regression.spatiotemporal_filter
spatiotemporal_filter_grad = _module_regression.spatiotemporal_filter_grad
cross_regression = _module_regression.cross_regression

@tf.RegisterGradient("SpatiotemporalFilter")
def _spatiotemporal_filter_grad(op, grad_acc_imga, grad_acc_imgb, grad_wgt_suma, grad_wgt_sumb):    
    imga = op.inputs[0]  
    imgb = op.inputs[1]  
    albedo = op.inputs[2]
    normal = op.inputs[3]
    band = op.inputs[4]  
    winSize = op.get_attr('win_size')        
    grad_band = spatiotemporal_filter_grad(grad_acc_imga, grad_acc_imgb, grad_wgt_suma, grad_wgt_sumb, imga, imgb, albedo, normal, band, win_size=winSize)        
    return [None, None, None, None, grad_band]

####################################################################################################
_module_temporal = tf.load_op_library('./ops/temporal.so')
temporal_screening = _module_temporal.temporal_screening

####################################################################################################
_module = tf.load_op_library('./ops/reproject.so')
Reproject = _module.reproject