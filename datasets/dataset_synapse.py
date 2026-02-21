import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    # Lightweight: only flip, no expensive rotation
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    # Fast: only 90/180/270 degree rotations (no expensive ndimage.rotate)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # support both single-slice (H, W) and stacked multichannel (H, W, C)
        if image.ndim == 2:
            h, w = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                # Use order=1 (bilinear) instead of order=3 (cubic) for FAST processing
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=1)
                label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        elif image.ndim == 3:
            # image shape: H, W, C (channels last)
            h, w, c = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                # Use order=1 (bilinear) instead of order=3 (cubic) for FAST processing
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=1)
                label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
            # to C,H,W
            image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))
        else:
            raise ValueError(f"Unsupported image ndim: {image.ndim}")

        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split in ["train", "val"] or self.sample_list[idx].strip('\n').split(",")[0].endswith(".npz"):
            slice_name = self.sample_list[idx].strip('\n').split(",")[0]
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
            # If this is a slice file (caseXXX_sliceYYY.npz) we will load neighboring slices
            # and stack them as channels (H, W, C)
            # attempt to detect slice pattern
            basename = os.path.basename(data_path)
            if "_slice" in basename:
                # parse case and index
                parts = basename.split("_slice")
                case = parts[0]
                idx_part = parts[1].split('.')[0]
                try:
                    center_idx = int(idx_part)
                except:
                    center_idx = 0

                def load_slice(case_name, i):
                    name = f"{case_name}_slice{int(i):03d}.npz"
                    p = os.path.join(self.data_dir, name)
                    if not os.path.exists(p):
                        return None
                    d = np.load(p)
                    try:
                        im = d['image']
                        lb = d['label']
                    except:
                        im = d['data']
                        lb = d['seg']
                    return im, lb

                prev_loaded = load_slice(case, max(0, center_idx - 1))
                center_loaded = load_slice(case, center_idx)
                next_loaded = load_slice(case, center_idx + 1)

                # fallback to center if neighbor missing
                if center_loaded is None:
                    # try direct load of provided path
                    data = np.load(data_path)
                    try:
                        image, label = data['image'], data['label']
                    except:
                        image, label = data['data'], data['seg']
                else:
                    prev_im, prev_lb = prev_loaded if prev_loaded is not None else (center_loaded[0], center_loaded[1])
                    cen_im, cen_lb = center_loaded
                    next_im, next_lb = next_loaded if next_loaded is not None else (center_loaded[0], center_loaded[1])
                    # stack channels last H,W,C
                    image = np.stack([prev_im, cen_im, next_im], axis=-1)
                    label = cen_lb
            else:
                data = np.load(data_path)
                try:
                    image, label = data['image'], data['label']
                except:
                    image, label = data['data'], data['seg']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            # if volume loaded (D, H, W) take a center 3-slice stack
            if image.ndim == 3:
                d, h, w = image.shape
                mid = d // 2
                prev_idx = max(0, mid - 1)
                next_idx = min(d - 1, mid + 1)
                image = np.stack([image[prev_idx], image[mid], image[next_idx]], axis=-1)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
