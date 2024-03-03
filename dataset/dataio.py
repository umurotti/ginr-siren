import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
#import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def to_uint8(x):
    return (255. * x).astype(np.uint8)

def to_numpy(x):
    return x.detach().cpu().numpy()

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data.view(-1, self.dataset.channels)
        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
            data = self.data[coord_idx, :]
            coords = self.mgrid[coord_idx, :]
        else:
            coords = self.mgrid
            data = self.data

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class ImageGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, test_sparsity=None, train_sparsity_range=(10, 200), generalization_mode=None):
        self.dataset = dataset
        self.sidelength = dataset.sidelength
        self.mgrid = dataset.mgrid
        self.test_sparsity = test_sparsity
        self.train_sparsity_range = train_sparsity_range
        self.generalization_mode = generalization_mode

    def __len__(self):
        return len(self.dataset)

    # update the sparsity of the images used in testing
    def update_test_sparsity(self, test_sparsity):
        self.test_sparsity = test_sparsity

    # generate the input dictionary based on the type of model used for generalization
    def get_generalization_in_dict(self, spatial_img, img, idx):
        # case where we use the convolutional encoder for generalization, either testing or training
        if self.generalization_mode == 'conv_cnp' or self.generalization_mode == 'conv_cnp_test':
            if self.test_sparsity == 'full':
                img_sparse = spatial_img
            elif self.test_sparsity == 'half':
                img_sparse = spatial_img
                img_sparse[:, 16:, :] = 0.
            else:
                if self.generalization_mode == 'conv_cnp_test':
                    num_context = int(self.test_sparsity)
                else:
                    num_context = int(
                        torch.empty(1).uniform_(self.train_sparsity_range[0], self.train_sparsity_range[1]).item())
                mask = spatial_img.new_empty(
                    1, spatial_img.size(1), spatial_img.size(2)).bernoulli_(p=num_context / np.prod(self.sidelength))
                img_sparse = mask * spatial_img
            in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sparse': img_sparse}
        # case where we use the set encoder for generalization, either testing or training
        elif self.generalization_mode == 'cnp' or self.generalization_mode == 'cnp_test':
            if self.test_sparsity == 'full':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img, 'coords_sub': self.mgrid}
            elif self.test_sparsity == 'half':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img[:512, :], 'coords_sub': self.mgrid[:512, :]}
            else:
                if self.generalization_mode == 'cnp_test':
                    subsamples = int(self.test_sparsity)
                    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]
                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}
                else:
                    subsamples = np.random.randint(self.train_sparsity_range[0], self.train_sparsity_range[1])
                    rand_idcs = np.random.choice(img.shape[0], size=self.train_sparsity_range[1], replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]

                    rand_idcs_2 = np.random.choice(img_sparse.shape[0], size=subsamples, replace=False)
                    ctxt_mask = torch.zeros(img_sparse.shape[0], 1)
                    ctxt_mask[rand_idcs_2, 0] = 1.

                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub,
                               'ctxt_mask': ctxt_mask}
        else:
            in_dict = {'idx': idx, 'coords': self.mgrid}

        return in_dict

    def __getitem__(self, idx):
        spatial_img, img, gt_dict = self.dataset.get_item_small(idx)
        in_dict = self.get_generalization_in_dict(spatial_img, img, idx)
        return in_dict, gt_dict


