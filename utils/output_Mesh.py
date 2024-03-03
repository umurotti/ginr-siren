from ..utils.sdf_meshing import create_mesh
import torch
import os
import sys
from pathlib import Path

from mlp_models import MLP3D


sdf_decoder = SDFDecoder(opt.model_type, opt.checkpoint_path, opt.mode)
name = Path(opt.checkpoint_path).stem
root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

sdf_meshing.create_mesh(
    sdf_decoder, os.path.join(root_path, name), N=opt.resolution
)