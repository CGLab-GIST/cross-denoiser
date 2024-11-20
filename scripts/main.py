import exr
from utils import *
import ops
import timing
import argparse

import torch
import torch.nn as nn

torch.set_printoptions(linewidth=200)
torch.set_printoptions(sci_mode=False)

imgs = {}
def add_write_img(tensor, name, frame, channels=None):
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

        if isinstance(tensor, torch.Tensor):
            img = tensor.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)

        imgs[name].append((frame, img, channels))


# Write images
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

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        if submodule.bias is not None:
            submodule.bias.data.fill_(0.01)

REGRESSION_WIDTH = 17
FILTER_WIDTH = 11
SPARSITY = 4
LEARNING_RATE = 1e-3

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class UNetSimple(nn.Module):
    def __init__(self):
        super(UNetSimple, self).__init__()
        
        # Encoding path
        self.c1_1 = ConvBlock(15, 8)
        self.c1_2 = ConvBlock(8, 8)
        self.pool1 = nn.MaxPool2d(2)
        
        self.c2_1 = ConvBlock(8, 16)
        self.c2_2 = ConvBlock(16, 16)
        self.pool2 = nn.MaxPool2d(2)
        
        self.c3_1 = ConvBlock(16, 32)
        self.c3_2 = ConvBlock(32, 32)
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=1, stride=2, padding=0, output_padding=1)
        
        # Decoding path
        self.c4_1 = ConvBlock(32 + 16, 16)
        self.c4_2 = ConvBlock(16, 16)
        self.upconv4 = nn.ConvTranspose2d(16, 16, kernel_size=1, stride=2, padding=0, output_padding=1)
        
        self.c5_1 = ConvBlock(16 + 8, 8)
        
    def forward(self, x):
        c1_1 = self.c1_1(x)
        c1_2 = self.c1_2(c1_1)
        c1_3 = self.pool1(c1_2)
        
        c2_1 = self.c2_1(c1_3)
        c2_2 = self.c2_2(c2_1)
        c2_3 = self.pool2(c2_2)
        
        c3_1 = self.c3_1(c2_3)
        c3_2 = self.c3_2(c3_1)
        c3_3 = self.upconv3(c3_2)
        
        # Concatenate along channel dimension
        c4 = torch.cat([c3_3, c2_2], dim=1)
        c4_1 = self.c4_1(c4)
        c4_2 = self.c4_2(c4_1)
        c4_3 = self.upconv4(c4_2)
        
        c5 = torch.cat([c4_3, c1_2], dim=1)
        c5_1 = self.c5_1(c5)
        
        return c5_1

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        
        # Define input channels
        self.unet = UNetSimple()
        self.param_conv = nn.Conv2d(8, 6, kernel_size=1)
        
    def forward(self, netColor1, netColor2, netPrev, netAlbedo, netNormal):
        # Concatenate inputs along channel dimension
        net_input = torch.cat([netColor1, netColor2, netPrev, netAlbedo, netNormal], dim=1)
        
        # Pass through UNet
        net_out = self.unet(net_input)
        
        # Get final output layers
        param = self.param_conv(net_out)
        band, alpha = torch.split(param, [5, 1], dim=1)
        
        band = 1 / (torch.square(band) + 1e-4)
        alpha = torch.sigmoid(alpha)
        
        return band, alpha

def train_step(item, prev_item):
    color = torch.log1p(item.color)
    color2 = torch.log1p(item.color2)
    albedo = item.albedo
    normal = item.normal

    ols1, ols2, vara, varb = ops.CrossRegression.apply(color, color2, albedo, normal, REGRESSION_WIDTH, SPARSITY)

    success, out = ops.Reproject.apply([prev_item.out_log, prev_item.ols1, prev_item.ols2], item.mvec, item.linearZ, prev_item.linearZ, item.normal, prev_item.normal, item.pnFwidth, item.ones1, item.ones1)
    prev_out_old, prev_ols1_old, prev_ols2_old = out

    success = success.float()
    success = ops.Erosion2d.apply(success, 3)

    prev_out, prev_ols1, prev_ols2, success_prev = ops.TemporalScreening.apply(ols1, ols2, prev_out_old, prev_ols1_old, prev_ols2_old, success, 7, 1)

    band, alpha = model(color, color2, prev_out, albedo, normal)

    sum1, sum2, sumw1, sumw2 = ops.SpatialFilter.apply(ols1, ols2, albedo, normal, band, FILTER_WIDTH)

    out1 = sum1 / (sumw1 + 1e-4)
    out2 = sum2 / (sumw2 + 1e-4)
    out = (sum1 + sum2) / (sumw1 + sumw2 + 1e-4)

    opacity = torch.where(item.opacity > 0, 1.0, 0.0)
    succes = success * success_prev
    alpha = success * alpha + (1 - success)
    out_log = alpha * out + (1 - alpha) * prev_out
    out_log *= opacity
    out = torch.maximum(item.zeros3, torch.expm1(out_log))

    label1 = ols2
    label2 = ols1
    prev_label1 = prev_ols2
    prev_label2 = prev_ols1

    loss1 = torch.square(out1 - label1) / (torch.square(label1) + 1e-2)
    loss2 = torch.square(out2 - label2) / (torch.square(label2) + 1e-2)
    spatial_loss = (loss1 + loss2) * 0.5

    tloss1 = torch.square(out1 - prev_label1) / (torch.square(prev_label1) + 1e-2)
    tloss2 = torch.square(out2 - prev_label2) / (torch.square(prev_label2) + 1e-2)
    temporal_loss = (tloss1 + tloss2) * 0.5
    
    loss = (spatial_loss + temporal_loss) * 0.5
    loss *= opacity

    avg_loss = torch.mean(loss)

    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()
    
    ret_dict = {
        'out': out,
        'out_log': out_log,
        'ols1': ols1,
        'ols2': ols2,
        'band': band,
        'alpha': alpha,
    }

    return loss, ret_dict


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--input_dir", type=str, default="/dataset")
    argsparser.add_argument("--scene", type=str, required=True)
    argsparser.add_argument("--frames", type=int, default=101)
    argsparser.add_argument("--tmp_dir", type=str, default="./tmp_scenes")
    argsparser.add_argument("--no_npy", action="store_true")
    argsparser.add_argument("--out_dir", type=str, default="./results")
    argsparser.add_argument("--experiment", type=str, default=None)
    args = argsparser.parse_args()

    # Initialize custom ops
    ops.init()

    input_dir = "/nas/dataset_small"
    SCENE = "BistroExterior2"
    read_types = ['ref']
    read_types += ['color', 'color2']
    read_types += ['mvec', 'linearZ', 'pnFwidth', 'opacity']
    read_types += ['albedo', 'normal']
    read_types += ['emissive', 'envLight']
    read_types += ['ref_emissive', 'ref_envLight']
    loader = NpyDataset(os.path.join(args.input_dir, args.scene), read_types, max_num_frames=args.frames, scene_dir=args.tmp_dir, no_npy=args.no_npy)

    # Network initialization
    # Example model instantiation and printing of parameter count
    model = MainNet().cuda()
    model.apply(weight_init_xavier_uniform)
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-7) # eps same with TF
    optimizer.zero_grad()

    time = timing.Timing()

    # Initialize
    for item in loader:
        break
    width, height = item.color.shape[3], item.color.shape[2]
    
    # train_opt = torch.compile(train)
    # model_opt = torch.compile(model)
    for item in loader:
        if item.index == 0:
            prev_item = Var()
            prev_item.out_log = item.zeros3
            prev_item.ols1 = item.zeros3
            prev_item.ols2 = item.zeros3
            prev_item.linearZ = item.zeros3
            prev_item.normal = item.zeros3
        
        time.start("train_step")
        loss, output = train_step(item, prev_item)
        prev_item = item
        prev_item.out_log = output['out_log']
        prev_item.ols1 = output['ols1']
        prev_item.ols2 = output['ols2']
        time.end("train_step", item.index < 20)
        
        print(f"[Frame {item.index:03d}]", end=" ")
        # print(f"Loss: {loss.item():.6f}", end=" ")
        print(f"({time.to_string()})")

        if True:
            ols1 = torch.expm1(output['ols1'])
            ols2 = torch.expm1(output['ols2'])
            olsavg = (ols1 + ols2) * 0.5 + item.emissive + item.envLight
            out = output['out'] + item.emissive + item.envLight
            ref = item.ref + item.ref_emissive + item.ref_envLight
            add_write_img(ols1, "ols1", item.index)
            add_write_img(ols2, "ols2", item.index)
            add_write_img(olsavg, "olsavg", item.index)
            add_write_img(out, "out", item.index)
            add_write_img(ref, "ref", item.index)
            add_write_img(output['band'], "band", item.index)
            add_write_img(output['alpha'], "alpha", item.index)
    
    out_dir = os.path.join(args.out_dir, args.scene) if args.experiment is None else os.path.join(args.out_dir, f"{args.scene}_{args.experiment}")
    os.makedirs(out_dir, exist_ok=True)

    print("Writing to file...", end="")
    # Merge write images into one list with name
    write_imgs = []
    for name, imgs in imgs.items():
        write_imgs += [(name, i, img, channels) for i, img, channels in imgs]
    name_imgs = [(f"{out_dir}/{name}_{i:04d}.exr", img, channels) for name, i, img, channels in write_imgs]

    # Sort by numbers in descending order
    name_imgs = sorted(name_imgs, key=lambda item: int(item[0].split('_')[-1].split('.')[0]), reverse=True)

    # Use multiprocessing to write images in parallel
    num_workers = mp.cpu_count() // 2  # Use half of available CPU cores
    with mp.Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_write, name_imgs), total=len(name_imgs)))
    print("Done!")
