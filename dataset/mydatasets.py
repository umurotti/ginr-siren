import os

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ShapeNet(Dataset):
    def __init__(
        self,
        dataset_folder,
        split,
        type,
        clipping_treshold = 0.1
    ):
        self.split = split
        self.data_source = os.path.join(dataset_folder,split)
        #self.filter = os.path.join(dataset_folder,'shape_filter_small.txt')
        self.batch = 5000
        self.clipping_treshold = clipping_treshold
        self.type = type
        files = []
        filter_list=[]
        #with open(self.filter,"r") as text:
        #    for line in text:
        #        file_name = line.strip()
        #        filter_list.append(file_name+".npy")

        for file in os.listdir(self.data_source):
            #if file in filter_list:
            files.append(file)

        self.npyfiles = files

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        if self.type == 'sdf' or self.type =='occ':
            path = os.path.join(self.data_source,self.npyfiles[idx])
            point_cloud = np.load(path)
            self.coords = point_cloud[:, :3]
            #self.occupancies = point_cloud[:, 3].reshape(-1, 1)
            self.sdf = point_cloud[:, 3].reshape(-1, 1)
            self.sdf = np.clip(self.sdf, -self.clipping_treshold, self.clipping_treshold)
            #self.labels = point_cloud[:, 5].reshape(-1, 1)
            self.normals = point_cloud[:,4:7]

            self.labels = np.zeros_like(self.sdf)
            #self.labels[np.where(self.sdf<0.01)[0]] = 1

            positive_indices = np.where(self.sdf > 0)[0]  #stands for inside, I Think
            negative_indices = np.where(self.sdf < 0)[0]

            self.tensor = np.concatenate([self.coords,self.sdf,self.normals,self.labels],axis=-1)
            self.pos_tensor = self.tensor[positive_indices,:]
            self.neg_tensor = self.tensor[negative_indices,:]


            sample= self.create_sdf_sample(idx)
            return {"coords": torch.from_numpy(sample[:,:3]).float(),
                    "sdf": torch.from_numpy(sample[:,3][:,None]),
                    "label":torch.from_numpy(sample[:,6][:,None]),
                    "normal":torch.from_numpy(sample[:,4:6]).float(),
                    "path": path
                    }
            return {"coords": torch.from_numpy(sample[:,:3]).float(),
                    "sdf": torch.from_numpy(sample[:,3][:,None]),
                    "label":torch.from_numpy(sample[:,7][:,None]),
                    "normal":torch.from_numpy(sample[:,4:7]).float()
                    }

        elif self.type == 'siren_sdf':
            path = os.path.join(self.data_source,self.npyfiles[idx])
            #point_cloud = np.genfromtxt(path)
            point_cloud = np.load(path)
            self.coords = point_cloud[:, :3]
            self.normals = point_cloud[:, 3:]

            self.on_surface_points = self.batch // 2


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


    def create_sdf_sample(self,idx):

        pos_num = int(self.batch / 2)
        neg_num = int(self.batch - pos_num)

        #a=a[torch.randperm(a.size()[0])] row shuffling
        pos_tensor_sel = self.pos_tensor[np.random.permutation(self.pos_tensor.shape[0])][:pos_num,:]
        neg_tensor_sel = self.neg_tensor[np.random.permutation(self.neg_tensor.shape[0])][:neg_num,:]

        # TODO: Implement such that you return a pytorch float32 torch tensor of shape (self.num_sample_points, 4)
        # the returned tensor shoud have approximately self.num_sample_points/2 randomly selected samples from pos_tensor
        # and approximately self.num_sample_points/2 randomly selected samples from neg_tensor
        return np.concatenate([pos_tensor_sel,neg_tensor_sel],axis=0)
        pass


class Pointcloud(Dataset):
    def __init__(
        self,
        pc_path,
        split,
        type,
        clipping_treshold = 0.1
    ):
        self.batch =5000
        self.clipping_treshold = clipping_treshold
        
        self.type = type
        if self.type =='sdf'or self.type =='occ':
            point_cloud = np.load(
                pc_path
            )
            self.coords = point_cloud[:, :3]
            #self.occupancies = point_cloud[:, 3].reshape(-1, 1)
            self.sdf = point_cloud[:, 3].reshape(-1, 1)
            self.sdf = np.clip(self.sdf, -self.clipping_treshold, self.clipping_treshold)
            #self.labels = point_cloud[:, 5].reshape(-1, 1)
            self.normals = point_cloud[:,4:7]

            self.labels = np.zeros_like(self.sdf)
            self.labels[np.where(self.sdf<0.01)[0]] = 1

            positive_indices = np.where(self.sdf > 0)[0]  #stands for inside, I Think
            negative_indices = np.where(self.sdf < 0)[0]
            self.tensor = np.concatenate([self.coords,self.sdf,self.normals,self.labels],axis=-1)
            self.pos_tensor = self.tensor[positive_indices,:]
            self.neg_tensor = self.tensor[negative_indices,:]

            self.num_sample_points = self.batch

            # truncate sdf values


        elif self.type =='siren_sdf':
            print("Loading point cloud")
            point_cloud = np.load(pc_path)
            #point_cloud = np.genfromtxt(pc_path)
            print("Finished loading point cloud")

            self.coords = point_cloud[:, :3]
            self.normals = point_cloud[:, 3:]

            self.on_surface_points = self.batch //2

    def __len__(self):
        return int(self.coords.shape[0]//self.batch)


    def __getitem__(self, idx):
        if self.type == 'sdf'or self.type =='occ':

            sample= self.create_sdf_sample(idx)
            return {"coords": torch.from_numpy(sample[:,:3]).float(),
                    "sdf": torch.from_numpy(sample[:,3][:,None]),
                    "label":torch.from_numpy(sample[:,6][:,None]),
                    "normal":torch.from_numpy(sample[:,4:6]).float()
                    }
            return {"coords": torch.from_numpy(sample[:,:3]).float(),
                    "sdf": torch.from_numpy(sample[:,3][:,None]),
                    "label":torch.from_numpy(sample[:,7][:,None]),
                    "normal":torch.from_numpy(sample[:,4:7]).float()
                    }

        elif self.type == 'siren_sdf':
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

    def create_sdf_sample(self,idx):

        pos_num = int(self.num_sample_points / 2)
        neg_num = int(self.num_sample_points - pos_num)

        #a=a[torch.randperm(a.size()[0])] row shuffling
        pos_tensor_sel = self.pos_tensor[np.random.permutation(self.pos_tensor.shape[0])][:pos_num,:]
        neg_tensor_sel = self.neg_tensor[np.random.permutation(self.neg_tensor.shape[0])][:neg_num,:]

        # TODO: Implement such that you return a pytorch float32 torch tensor of shape (self.num_sample_points, 4)
        # the returned tensor shoud have approximately self.num_sample_points/2 randomly selected samples from pos_tensor
        # and approximately self.num_sample_points/2 randomly selected samples from neg_tensor
        return np.concatenate([neg_tensor_sel,pos_tensor_sel],axis=0)


class ImageNette(Dataset):
    """Dataset for ImageNette that contains 10 classes of ImageNet.
    Dataset parses the pathes of images and load the image using PIL loader.

    Args:
        split: "train" or "val"
        transform (sequence or torch.nn.Module): list of transformations
    """

    root = Path(__file__).parent.parent.joinpath("data/imagenette")
    """Dataset for ImageNette that contains 10 classes of ImageNet.
    Dataset parses the pathes of images and load the image using PIL loader.

    Args:
        split: "train" or "val"
        transform (sequence or torch.nn.Module): list of transformations
    """

    def __init__(self, split="train", transform=None):
        assert split in ["train", "val"]
        self.transform = transform

        root_path = os.path.join(ImageNette.root, split)
        class_folders = sorted(os.listdir(root_path))
        self.data_path = []

        for class_folder in class_folders:
            # for testing:
            if class_folder == 'test':
                filenames = sorted(os.listdir(os.path.join(root_path, class_folder)))
                for name in filenames:
                    self.data_path.append(os.path.join(root_path, class_folder, name))


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of data_path
        Returns:
            img (torch.Tensor): (C, H, W) shape of tensor
        """
        img = Image.open(self.data_path[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


