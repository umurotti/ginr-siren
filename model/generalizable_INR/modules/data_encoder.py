import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from ..modules.pointnet2_utils import PointNetSetAbstraction
from .utils import convert_int_to_list
#import pyrender

class Unfold1D(nn.Module):
    """unfold audio (3D tensor), where the channel dimension is one."""

    def __init__(self, patch_size=200, use_padding=True, padding_type="reflect"):
        super().__init__()
        self.patch_size = patch_size
        self.use_padding = use_padding
        self.padding_type = padding_type

    def forward(self, data):
        assert data.ndim == 3  # audio
        if self.use_padding:
            if not data.shape[-1] % self.patch_size == 0:
                len_data = data.shape[-1]
                pad_size = self.patch_size - (len_data % self.patch_size)
            else:
                pad_size = 0
        else:
            pad_size = 0
        data = torch.nn.functional.pad(data, (0, pad_size), mode=self.padding_type)

        num_patches = data.shape[-1] // self.patch_size
        data = torch.reshape(data, (data.shape[0], num_patches, -1))
        return data


class Unfold(nn.Module):
    """Note: only 4D tensors are currently supported by pytorch."""

    def __init__(self, patch_size, padding=0, img_channel=3):
        super().__init__()
        self.patch_size = convert_int_to_list(patch_size, len_list=2)
        self.padding = convert_int_to_list(padding, len_list=2)

        self.unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=self.padding)

    def forward(self, data):
        """
        Args
            data (torch.tensor): data with shape = [batch_size, channel, ...]
        Returns
            unfolded_data (torch.tensor): unfolded data
                shape = [batch_size, channel * patch_size[0] * patch_size[1], L]
        """
        unfolded_data = self.unfold(data)
        return unfolded_data


class Pointnet2(nn.Module):
    def __init__(self,normal_channel=True):
        super(Pointnet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.025, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.05, nsample=64, in_channel=128 + 3, mlp=[128, 128, 285], group_all=False)
        #self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def draw_point_cloud(self,points):
        cloud = pyrender.Mesh.from_points(points)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)

    def forward(self, xyz):  # B,spatial, ?
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        xyz =  xyz.permute(0, 2, 1).contiguous()
        #self.draw_point_cloud(xyz.permute(0, 2, 1).contiguous()[0].cpu().detach())
        l1_xyz, l1_points = self.sa1(xyz, norm)
        #self.draw_point_cloud(l1_xyz.permute(0, 2, 1).contiguous()[0].cpu().detach())
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #self.draw_point_cloud(l2_xyz.permute(0, 2, 1).contiguous()[0].cpu().detach())

        return l2_xyz.permute(0, 2, 1).contiguous(), l2_points.permute(0, 2, 1).contiguous()


class DataEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = config.type
        self.trainable = config.trainable
        self.input_dim = config.n_channel
        self.output_dim = None

        spec = config.encoder_spec

        if self.type == "unfold":
            self.encoder = Unfold(spec.patch_size, spec.padding)
            self.output_dim = self.input_dim * np.product(self.encoder.patch_size)
            self.is_encoder_out_channels_last = False
        elif self.type == "mlp":
            self.encoder = nn.Sequential(nn.Linear(3,256),nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 768))

            self.output_dim = 768
            self.is_encoder_out_channels_last = True

        elif self.type == "PointNet2":
            self.encoder = Pointnet2(normal_channel=False)
            self.output_dim = 256
            self.is_encoder_out_channels_last = True

        else:
            # If necessary, implement additional wrapper for extracting features of data
            raise NotImplementedError

        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, xs, put_channels_last=False):
        xs_coord, xs_embed = self.encoder(xs)
        #xs_embed,_ = torch.max(xs_embed,dim=1)
        #xs_embed = xs_embed[:,None,:]
        if put_channels_last and not self.is_encoder_out_channels_last:
            permute_idx_range = [i for i in range(2, xs_embed.ndim)]
            return xs_embed.permute(0, *permute_idx_range, 1).contiguous()
        else:
            return xs_coord, xs_embed
