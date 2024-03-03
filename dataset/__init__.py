import os

import torch

from dataset.mydatasets import ShapeNet,Pointcloud,ImageNette
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    if config.dataset.type =="shapenet":
        if config.type == 'overfit':
            dataset_trn = Pointcloud(config.dataset.folder, split="train",type=config.dataset.supervision, clipping_treshold=config.dataset.clipping_treshold)
            dataset_val = Pointcloud(config.dataset.folder, split="val",type=config.dataset.supervision, clipping_treshold=config.dataset.clipping_treshold)
        else:
            dataset_trn = ShapeNet(config.dataset.folder, split="train",type=config.dataset.supervision, clipping_treshold=config.dataset.clipping_treshold)
            dataset_val = ShapeNet(config.dataset.folder, split="val",type=config.dataset.supervision, clipping_treshold=config.dataset.clipping_treshold)
    elif config.dataset.type == "imagenette":
        transforms_trn = create_transforms(config.dataset, split="train", is_eval=is_eval)
        transforms_val = create_transforms(config.dataset, split="val", is_eval=is_eval)
        dataset_trn = ImageNette(split="train", transform=transforms_trn)
        dataset_val = ImageNette(split="val", transform=transforms_val)
    else:
        raise ValueError("%s not supported..." % config.dataset.type)



    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f"#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}")

    return dataset_trn, dataset_val
