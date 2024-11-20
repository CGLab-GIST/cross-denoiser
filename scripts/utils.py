import os
import numpy as np
import exr
import glob
import multiprocessing as mp
import pathlib
from pathlib import Path
import tqdm
from collections import namedtuple

import torch
from torch.utils.data import IterableDataset


# An empty wrapper class for images
class Var:
    # Iterate item with its key
    def __iter__(self):
        for key, value in self.__dict__.items():
            if key == "index": continue
            yield key, value

    def to_gpu(self):
        for key, value in self.__dict__.items():
            if key == "index": continue
            self.__dict__[key] = torch.from_numpy(value).detach().cuda().contiguous()
    
    def to_dict(self):
        d = self.__dict__.copy()
        if "index" in d and not isinstance(d["index"], torch.tensor):
            d['index'] = torch.tensor(d['index'])
        return d

def load_exr(path):
    numpy_img = exr.read_all(path)["default"]
    # Add batch dim
    numpy_img = np.expand_dims(numpy_img, axis=0)
    return numpy_img

def save_as_npy(path):
    npy_path = path.replace('.exr', '.npy')
    if os.path.exists(npy_path):
        return
    numpy_img = load_exr(path)

    # Invalid value handling
    if np.isnan(numpy_img).any():
        print('There is NaN in', npy_path, 'Set it to zero for training.')
        numpy_img = np.nan_to_num(numpy_img, copy=False)
    if np.isposinf(numpy_img).any() or np.isneginf(numpy_img).any():
        print("There is INF in", npy_path, 'Set it to zero for training.')
        numpy_img[numpy_img == np.inf] = 0
        numpy_img[numpy_img == -np.inf] = 0

    np.save(npy_path, numpy_img)

# Make symolic link to files in orig_dir in new_dir using exr_dict
def make_symbolic(orig_dir, new_dir, exr_dict):
    # Make symbolic links
    for key, files in exr_dict.items():
        for file in files:
            basename = os.path.basename(file)
            orig_path = os.path.join(orig_dir, file)
            new_file = os.path.join(new_dir, basename)
            # Make symbolic link
            if os.path.exists(new_file):
                os.remove(new_file)
            os.symlink(orig_path, new_file)

def extract_name_frame(filename):
    # Split the path into parts using the appropriate separator
    if os.path.sep == "\\":
        parts = filename.split("\\")
    else:
        parts = filename.split("/")
    # Extract the name from the last part
    name_and_number, ext = parts[-1].split(".", 1)
    name, frame = name_and_number.rsplit("_", 1)
    return name, int(frame)

def extract_filenames(filenames):
    # Extract the names from the file paths
    names = []
    # Dicts for each type
    files_dict = {}
    for filename in filenames:
        name, frame = extract_name_frame(filename)

        # Add the name to the list if it is not already there
        if name not in names:
            names.append(name)

        # Add the file to the dict
        if name not in files_dict:
            files_dict[name] = []
        files_dict[name].append(filename)
    return names, files_dict

# Function to check if all types of files have the same number of files and equal frame indices
def check_files(files_dict):
    # Check if all files have the same number of files
    num_files = {key: len(files) for key, files in files_dict.items()}
    all_same = len(set(num_files.values())) == 1
    if not all_same:
        raise Exception(f'Error: different number of files\n{num_files}')
    
    # Check if all files have equal frame indices
    frames_dict = {}
    for key, files in files_dict.items():
        frames = set()
        for file in files:
            filename = os.path.basename(file)
            name, frame = os.path.splitext(filename)[0].rsplit('_', 1)
            frames.add(int(frame))
        frames_dict[key] = frames
        if not 'frames_compare' in locals():
            frames_compare = frames
            continue
        if frames != frames_compare:
            raise Exception(f"Error: {key} has different frame indices than other types of files\n\t{frames}\n\t{frames_compare}")

    # Check if frame_dict is empty
    if not frames_dict:
        raise Exception('frames_dict is empty')

    # Check if any type of frames_dict has different frame indices than other types
    for key, frames in frames_dict.items():
        for key2, frames2 in frames_dict.items():
            if key != key2 and frames != frames2:
                raise(f"Error: {key} has different frame indices than {key2}")


class NpyDataset(IterableDataset):
    def __init__(self, directory, types, max_num_frames=101, scene_dir='./scenes', no_npy=False, *args, **kwargs):
        super(NpyDataset, self).__init__(*args, **kwargs)
        self.no_npy = no_npy

        # List files in format "{type}_{frame:04d}.exr"
        img_list = sorted(glob.glob(os.path.join(directory, "*.exr")))
        img_list = [os.path.basename(x) for x in img_list]
        img_list = [x for x in img_list if x.rsplit("_", 1)[0] in types]
        assert len(img_list) > 0, directory # Check emtpy

        # Remove excess frames of all types from the list
        if max_num_frames is not None:
            img_list = [x for x in img_list if int(x.rsplit("_", 1)[1].split(".")[0]) < max_num_frames]

        # Parse file type
        unique_types, exr_dict = extract_filenames(img_list)
        check_files(exr_dict)

        # Check if all types are given
        if set(types) != set(unique_types):
            raise Exception(f"Error: {set(types) - set(unique_types)} is not given in directory: {directory}")

        # Make a new directory
        parent_dir = os.path.dirname(directory)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            # Write parent directory of input to directory.txt
            with open(os.path.join(scene_dir, 'directory.txt'), 'w') as f:
                f.write(parent_dir)

        # Check if given directory is same as the one in directory.txt
        with open(os.path.join(scene_dir, 'directory.txt'), 'r') as f:
            orig_dir = f.read().replace('\n', '')
            if orig_dir != parent_dir:
                raise Exception(f"Error:\n\tloadded: {parent_dir}\n\tstored: {orig_dir}. \nTry again after removing {scene_dir}")

        new_dir = os.path.join(scene_dir, pathlib.PurePath(directory).name)
        # Make a data directory
        if not Path(new_dir).exists():
            print(f'Making a directory: {new_dir}')
            os.makedirs(new_dir)
            
        # Make a symbolic link of files in orig_dir in new_dir if not exist
        make_symbolic(directory, new_dir, exr_dict)

        # Check if the first and last saved npy of all types are same as exr
        npy_paths = [os.path.join(new_dir, f"{t}_{0:04d}.npy") for t in unique_types]
        npy_paths += [os.path.join(new_dir, f"{t}_{max_num_frames-1:04d}.npy") for t in unique_types]
        exr_paths = [os.path.join(directory, f"{t}_{0:04d}.exr") for t in unique_types]
        exr_paths += [os.path.join(directory, f"{t}_{max_num_frames-1:04d}.exr") for t in unique_types]

        for npy_path, exr_path in zip(npy_paths, exr_paths):
            if os.path.exists(npy_path):
                exr_img = load_exr(exr_path)
                exr_img = np.nan_to_num(exr_img, copy=False)
                npy_img = np.load(npy_path)
                assert np.allclose(exr_img, npy_img), f"Error: {exr_path} and {npy_path} are different. \nTry again after removing {new_dir}"

        # Save exr images as npy
        fullpath_list = [os.path.join(new_dir, x) for x in img_list]
        print('Making npy files for faster loading... ')
        num_cores = mp.cpu_count()
        with mp.Pool(num_cores) as p:
            list(tqdm.tqdm(p.imap(save_as_npy, fullpath_list), total=len(fullpath_list)))
        print('Done')
        
        # Change extension to npy
        img_list = [x.replace('.exr', '.npy') for x in img_list]

        # Get unique types
        unique_types = set([x.rsplit("_", 1)[0] for x in img_list])

        # Check each type has the same number of files
        num_frames = len(img_list) // len(unique_types)
        for t in unique_types:
            num_type = len([x for x in img_list if x.rsplit("_", 1)[0] == t])
            assert num_type == num_frames

        # Check each type has the same start number of frame
        start_frame = min([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = min(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == start_frame

        # Check each type has the same max number of frame
        max_frame = max([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = max(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == max_frame

        # Set for later use
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.directory = new_dir
        self.unique_types = unique_types

        print(f"Directory '{directory}' has types: \n\t{unique_types}\nwith {num_frames} frames")
        
    def skip_frames(self, num_frames):
        self.start_frame += num_frames
        self.num_frames -= num_frames
    
    def unpack(self, frame, imgs):
        unpacked = Var()
        unpacked.index = frame
        for i, type in enumerate(self.unique_types):
            unpacked.__dict__[type] = imgs[i]
        unpacked.to_gpu()
        # Make useful constants
        unpacked.ones3 = torch.ones_like(unpacked.mvec[:,0:3,:,:]).contiguous()
        unpacked.ones1 = unpacked.ones3[:,0:1,:,:].contiguous()
        unpacked.ones2 = unpacked.ones3[:,0:2,:,:].contiguous()
        unpacked.zeros3 = torch.zeros_like(unpacked.mvec[:,0:3,:,:]).contiguous()
        unpacked.zeros2 = unpacked.zeros3[:,0:2,:,:].contiguous()
        unpacked.zeros1 = unpacked.zeros3[:,0:1,:,:].contiguous()
        return unpacked

    def load_image(self, filename):
        img = np.load(filename)
        # Transpose to NCHW
        img = np.transpose(img, (0, 3, 1, 2))
        return img

    def __iter__(self):
        # Make filename using directory, unique_types and frame
        for frame in range(self.start_frame, self.start_frame + self.num_frames):
            ext = "exr" if self.no_npy else "npy"
            filenames = [
                os.path.join(self.directory, f"{t}_{frame:04d}.{ext}")
                for t in self.unique_types
            ]

            # Images with full-resolution and batch 1 [1, H, W, C]
            imgs = [self.load_image(f) for f in filenames]
            
            yield self.unpack(frame, imgs)

    def __len__(self):
        return self.num_frames

def dict_to_namedtuple(d):
    return namedtuple('Item', d.keys())(**d)
