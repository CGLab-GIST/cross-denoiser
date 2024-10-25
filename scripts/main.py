"""
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import exr
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import argparse

# Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 for all, 3 for error
import tensorflow as tf
keras = tf.keras
from keras import Model, layers
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found")
    exit(-1)

# Custom ops
from ops import *

# MISC.
from timing import Timing
from utils import *


# =============================================================================
# Argument parsing

argsparser = argparse.ArgumentParser()
argsparser.add_argument("--input_dir", type=str, default="/dataset")
argsparser.add_argument("--scene", type=str, required=True)
argsparser.add_argument("--frames", type=int, default=101)
argsparser.add_argument("--tmp_dir", type=str, default="./tmp_scenes")
argsparser.add_argument("--no_npy", action="store_true")
argsparser.add_argument("--out_dir", type=str, default="./results")
argsparser.add_argument("--experiment", type=str, default=None)
argsparser.add_argument('--save_img', type=bool, default=True)
args = argsparser.parse_args()

# =============================================================================
# Dataset loader

read_types = ['ref'] # Reference
read_types += ['color', 'color2'] # Colors
read_types += ['albedo', 'normal'] # G-buffer
read_types += ['mvec', 'linearZ', 'pnFwidth', 'opacity'] # For reprojection
read_types += ['emissive', 'envLight'] # Emissive
read_types += ['ref_emissive', 'ref_envLight'] # Emissive
loader = DataLoader(os.path.join(args.input_dir, args.scene), read_types, max_num_frames=args.frames, scene_dir=args.tmp_dir, no_npy=args.no_npy)

# =============================================================================
# Configurations

REGRESSION_WIDTH = 17
FILTER_WIDTH = 11
SPARSITY = 4
LEARNING_RATE = 1e-3

# =============================================================================
# Initialize network

tf.keras.utils.set_random_seed(1234) # From keras documentation to make (nearly-)reproducible results

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

def unet_simple(input):
    c1_1 = layers.Conv2D(filters=8, kernel_size=3, activation='relu', use_bias=True, padding='same')(input)
    c1_2 = layers.Conv2D(filters=8, kernel_size=3, activation='relu', use_bias=True, padding='same')(c1_1)
    c1_3 = layers.MaxPooling2D(2)(c1_2)
    c2_1 = layers.Conv2D(filters=16, kernel_size=3, activation='relu', use_bias=True, padding='same')(c1_3)
    c2_2 = layers.Conv2D(filters=16, kernel_size=3, activation='relu', use_bias=True, padding='same')(c2_1)
    c2_3 = layers.MaxPooling2D(2)(c2_2)
    c3_1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', use_bias=True, padding='same')(c2_3)
    c3_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu', use_bias=True, padding='same')(c3_1)
    c3_3 = layers.Conv2DTranspose(filters=32, kernel_size=1, strides=(2, 2), padding='same')(c3_2)
    c4 = layers.Concatenate(axis=-1)([c3_3, c2_2])
    c4_1 = layers.Conv2D(filters=16, kernel_size=3, activation='relu', use_bias=True, padding='same')(c4)
    c4_2 = layers.Conv2D(filters=16, kernel_size=3, activation='relu', use_bias=True, padding='same')(c4_1)
    c4_3 = layers.Conv2DTranspose(filters=16, kernel_size=1, strides=(2, 2), padding='same')(c4_2)
    c5 = layers.Concatenate(axis=-1)([c4_3, c1_2])
    c5_1 = layers.Conv2D(filters=8, kernel_size=3, activation='relu', use_bias=True, padding='same')(c5)
    return c5_1

def make_network():
    ## Bandwidth prediction
    netColor1, netColor2, netPrev, netAlbedo, netNormal = make_inputs([3, 3, 3, 3, 3])
    net_input_all = [netColor1, netColor2, netPrev, netAlbedo, netNormal]
    net_input = layers.Concatenate(axis=-1)(net_input_all)

    dims = [5, 1]

    net_out = unet_simple(net_input)
    param = layers.Conv2D(filters=sum(dims), kernel_size=1, use_bias=True, padding='same', dtype='float32')(net_out)
    band, alpha = keras.ops.split(param, dims[:-1], axis=-1)
    band = keras.ops.reciprocal(keras.ops.square(band) + 1e-4)
    alpha = keras.activations.sigmoid(alpha)

    model = Model(name="MainNet", inputs=net_input_all, outputs=[band, alpha])
    model.compile(jit_compile=True)
    print(f"Number of parameters in the model: {model.count_params():,}")
    return model

filter_net = make_network()

# =============================================================================
# Misc.

timing = Timing()

imgs = {}
def append_to_write(tensor, name, frame, channels=None):
    if name not in imgs:
        imgs[name] = []
    # Check if the same frame is already added
    if len(imgs[name]) == 0 or imgs[name][-1][0] != frame:
        # Remove batch
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        # Do not add non-image shape
        if len(tensor.shape) < 3:
            imgs.pop(name, None)
            return

        if isinstance(tensor, tf.Tensor):
            imgs[name].append((frame, tensor.numpy(), channels))
        elif isinstance(tensor, np.ndarray):
            imgs[name].append((frame, tensor, channels))

def process_write(item):
    path, img, channels = item

    # non-zero
    keywords = ['out', 'ols', 'ref']
    if any(keyword in path for keyword in keywords):
        img = np.maximum(0.0, img)

    # Write
    try:
        if channels is not None:
            exr.write(path, img, channel_names=channels, compression=exr.ZIP_COMPRESSION)
        else:
            if img.shape[-1] <= 3:
                exr.write(path, img, compression=exr.ZIP_COMPRESSION)
            elif img.shape[-1] > 3:
                exr.write(path, img, channel_names=[f"ch{i}" for i in range(img.shape[-1])], compression=exr.ZIP_COMPRESSION)
            else:
                raise NotImplementedError
    except Exception as e:
        print(e)
        raise e

def write_images():
    global imgs

    OUT_DIR = os.path.join(args.out_dir, args.scene) if args.experiment is None else os.path.join(args.out_dir, f"{args.scene}_{args.experiment}")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Merge write images into one list with name
    write_imgs = []
    for name, imgs in imgs.items():
        write_imgs += [(name, i, img, channels) for i, img, channels in imgs]
    name_imgs = [(f"{OUT_DIR}/{name}_{i:04d}.exr", img, channels) for name, i, img, channels in write_imgs]

    # Sort by numbers in descending order
    name_imgs = sorted(name_imgs, key=lambda item: int(item[0].split('_')[-1].split('.')[0]), reverse=True)

    # Use multiprocessing to write images in parallel
    num_workers = mp.cpu_count() // 2  # Use half of available CPU cores
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_write, name_imgs), total=len(name_imgs)))

def RelL2(y_pred, y_true):
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    true_mean = np.mean(y_true, axis=-1, keepdims=True)
    return np.square(y_true - y_pred) / (np.square(true_mean) + 1e-2)

filter_erosion3_arr = [[[1.0], [1.0], [1.0]], 
                  [[1.0], [1.0], [1.0]], 
                  [[1.0], [1.0], [1.0]]]
filter_erosion3 = tf.constant(filter_erosion3_arr, dtype=tf.float32)

# =============================================================================
# Functions

@tf.function(jit_compile=True)
def prepare_input(y1, y2, albedo, normal, opacity):
    ## Set input
    y1 = keras.ops.log1p(y1) # log
    y2 = keras.ops.log1p(y2) # log
    return y1, y2, albedo, normal, opacity

@tf.function(jit_compile=True)
def aggregation_loss(sum1, sumw1, sum2, sumw2, alpha, success, success_prev, prev_out, opacity, label1, label2, prev_label1, prev_label2):
    ## Spatial filtering
    out1 = sum1 / (sumw1 + 1e-4)
    out2 = sum2 / (sumw2 + 1e-4)
    out = (sum1 + sum2) / (sumw1 + sumw2 + 1e-4)

    ## Temporal filtering
    opacity = tf.where(opacity > 0, 1.0, 0.0)
    success = success * success_prev
    alpha = success * alpha + (1 - success)
    out = alpha * out + (1 - alpha) * prev_out
    out *= opacity

    ## Calculate for loss
    out1 = alpha * out1 + (1 - alpha) * prev_out
    out2 = alpha * out2 + (1 - alpha) * prev_out

    ## Spatial loss
    loss1 = tf.square(out1 - label1) / (tf.square(label1) + 1e-2)
    loss2 = tf.square(out2 - label2) / (tf.square(label2) + 1e-2)
    spatial_loss = (loss1 + loss2) * 0.5

    ## Temporal loss
    tloss1 = tf.square(out1 - prev_label1) / (tf.square(prev_label1) + 1e-2)
    tloss2 = tf.square(out2 - prev_label2) / (tf.square(prev_label2) + 1e-2)
    temporal_loss = (tloss1 + tloss2) * 0.5
    
    ## Final loss
    loss = spatial_loss * 0.5 + temporal_loss * 0.5
    loss *= opacity

    avgLoss = tf.reduce_mean(loss)

    return out, tf.maximum(0.0, tf.math.expm1(out)), loss, avgLoss

@tf.function
def train_step(d, prev_d):
    with tf.xla.experimental.jit_scope(compile_ops=True):
        item = dict_to_namedtuple(d)
        prev_item = dict_to_namedtuple(prev_d)

        ## Prepare input
        y1, y2, albedo, normal, opacity = prepare_input(item.color, item.color2, item.albedo, item.normal, item.opacity)

        with tf.name_scope("cross-regression"):
            with tf.xla.experimental.jit_scope(compile_ops=False):
                ols1, ols2, var1, var2 = cross_regression(y1, y2, albedo, normal, win_size=REGRESSION_WIDTH, sparsity=SPARSITY)

        with tf.name_scope("reproject"):
            ## Reprojection
            with tf.xla.experimental.jit_scope(compile_ops=False):
                success, reproj_out = Reproject(item.mvec, [prev_item.out_log, prev_item.ols1, prev_item.ols2], item.linearZ, prev_item.linearZ, item.normal, prev_item.normal, item.pnFwidth, item.ones1, item.ones1)
                prev_out_old, prev_ols1, prev_ols2 = reproj_out

            # Handle bilinear interpolation artifacts in reprojection
            success = tf.cast(success, tf.float32)
            success = tf.nn.erosion2d(value=success,
                                    filters=filter_erosion3, 
                                    strides=[1, 1, 1, 1], 
                                    padding='SAME', 
                                    data_format='NHWC',
                                    dilations=[1, 1, 1, 1]) + 1

            ## Additional temporal screening for temporal label/prev_out
            with tf.xla.experimental.jit_scope(compile_ops=False):
                prev_out, prev_label1, prev_label2, success_prev = temporal_screening(ols1, ols2, prev_out_old, prev_ols1, prev_ols2, success, win_curr=7, win_prev=1)

        ## Set label
        label1 = ols2
        label2 = ols1

        with tf.name_scope("spatiotemporal"):
            with tf.GradientTape() as g:
                ## Network prediction
                band, alpha = filter_net([ols1, ols2, prev_out, albedo, normal], training=True)

                with tf.xla.experimental.jit_scope(compile_ops=False):
                    ## Spatial filtering
                    sum1, sum2, sumw1, sumw2 = spatiotemporal_filter(ols1, ols2, albedo, normal, band, win_size=FILTER_WIDTH)

                ## Spatiotemporal + loss calculation
                out_log, out, loss, avgLoss = aggregation_loss(sum1, sumw1, sum2, sumw2, alpha, success, success_prev, prev_out, opacity, label1, label2, prev_label1, prev_label2)

        ## Gradient update
        with tf.name_scope("backpropagation"):
            dL_dk = g.gradient(avgLoss, filter_net.trainable_variables)
            optimizer.apply_gradients(zip(dL_dk, filter_net.trainable_variables))

        dicts = {
            'out': out,
            # For next frame
            'out_log': out_log,
            'ols1': ols1,
            'ols2': ols2,
        }

    return loss, dicts

# =============================================================================
# Main function

if __name__ == '__main__':
    for item, _ in loader:
        index = item.index

        # =====================================================================
        if loader.counter == 0:
            prev_item = Var()
            prev_item.out_log = item.zeros3
            prev_item.ols1 = item.zeros3
            prev_item.ols2 = item.zeros3
            prev_item.linearZ = item.zeros3
            prev_item.normal = item.zeros3
            loss = 0
        
        item_dict = item.to_dict()
        prev_item_dict = prev_item.to_dict()

        timing.start('train')
        loss_train, output = train_step(item_dict, prev_item_dict)
        # Accumulated -> history
        prev_item = item
        prev_item.out_log = output['out_log']
        prev_item.ols1 = output['ols1']
        prev_item.ols2 = output['ols2']
        timing.end('train', loader.counter <= 20)

        # =====================================================================
        out = output['out'] + item.emissive + item.envLight
        ref = item.ref + item.ref_emissive + item.ref_envLight
        error = np.mean(RelL2(out, ref))
        print(f"[Frame {index:3d}] relL2: {error:.4f} ({timing.to_string(total_only=True)})")

        # =====================================================================
        if args.save_img:
            ols1 = tf.maximum(0.0, tf.math.expm1(output['ols1']))
            ols2 = tf.maximum(0.0, tf.math.expm1(output['ols2']))
            olsavg = (ols1 + ols2) * 0.5 + item.emissive + item.envLight
            append_to_write(olsavg, 'olsavg', index)
            append_to_write(out, 'out', index)
            append_to_write(ref, 'ref', index)
            path = (item.color + item.color2) * 0.5 + item.emissive + item.envLight
            append_to_write(path, 'path', index)

    # =============================================================================
    print("Writing images...", end="")
    write_images()
    print('Done!')