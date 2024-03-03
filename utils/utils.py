import os
import logging
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from easydict import EasyDict as edict
from .writer import Writer

import trimesh
import pyrender


def config_setup(args, distenv, result_path):
    if args.eval:
        config = yaml.load(open(os.path.join(args.result_path, "config.yaml")), Loader=yaml.FullLoader)
        config = config_init(config)
        if hasattr(args, "test_batch_size"):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, "seed"):
            config.seed = args.seed
    elif args.resume:
        config = yaml.load(open(os.path.join(os.path.dirname(args.result_path), "config.yaml")), Loader=yaml.FullLoader)
        config = config_init(config)
    else:
        config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
        config = config_init(config)
        config.seed = args.seed

        if hasattr(config.experiment, "total_batch_size"):
            t_batch_size = config.experiment.total_batch_size
            l_batch_size = config.experiment.batch_size
            assert t_batch_size % (l_batch_size * distenv.world_size) == 0
            config.optimizer.grad_accm_steps = int(t_batch_size / (l_batch_size * distenv.world_size))
        else:
            config.experiment.total_batch_size = config.experiment.batch_size * distenv.world_size
            config.optimizer.grad_accm_steps = 1

        if distenv.master:
            config.result_path = result_path
            yaml.dump(config, open(os.path.join(result_path, "config.yaml"), "w"))

    return config

def logger_setup(args):
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank > 0 or args.node_rank > 0:
        return None, None, None

    if args.eval:
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        result_path = os.path.join(args.result_path, "val", now)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "val.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname), logging.StreamHandler()],
        )
    elif args.resume:
        result_path = os.path.dirname(args.result_path)
        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "train.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname, mode="a"), logging.StreamHandler()],
        )
    else:
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_cfg_name = os.path.splitext(args.model_config.split("/")[-1])[0]
        result_path = os.path.join(args.result_path, model_cfg_name, now)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "train.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname), logging.StreamHandler()],
        )
    logger = logging.getLogger(__name__)
    logger.info(args)

    return logger, writer, result_path

def config_init(config):
    config = edict(config)

    def set_default_attr(cfg, attr, default):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    set_default_attr(config.dataset.transforms, "type", None)
    set_default_attr(config.arch.hparams, "loss_type", "mse")
    set_default_attr(config.arch, "ema", None)
    set_default_attr(config.optimizer, "max_gn", None)
    set_default_attr(config.optimizer.warmup, "start_from_zero", True if config.optimizer.warmup.epoch > 0 else False)
    set_default_attr(config.optimizer, "type", "adamW")
    set_default_attr(config.optimizer.warmup, "mode", "linear")
    set_default_attr(config.experiment, "test_freq", 10)
    set_default_attr(config.experiment, "amp", False)

    return config

def set_seed(seed=None):
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def render_meshes(meshes):
    out_imgs = []
    for mesh in meshes:
        img, _ = render_mesh(mesh)
        out_imgs.append(img)
    return out_imgs


def render_mesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        # Handle mesh rendering
        mesh = pyrender.Mesh.from_trimesh(
            obj,
            material=pyrender.MetallicRoughnessMaterial(
                alphaMode="BLEND",
                baseColorFactor=[1, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8,
            ),
        )
    else:
        # Handle point cloud rendering, (converting it into a mesh instance)
        pts = obj
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    eye = np.array([2, 1.4, -2])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    camera_pose = look_at(eye, target, up)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(800, 800)
    color, depth = r.render(scene)
    r.delete()
    return color, depth

# Calculate look-at matrix for rendering
def look_at(eye, target, up):
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    camera_pose = np.eye(4)
    camera_pose[:-1, 0] = right
    camera_pose[:-1, 1] = up
    camera_pose[:-1, 2] = forward
    camera_pose[:-1, 3] = eye
    return camera_pose

def list_and_save_filenames(directory, output_file, search_extension=".obj.npy", add_extension=".obj"):
    """
    Lists all files in a directory with a specific extension and saves their names, with an additional extension, to a file.

    :param directory: The directory to search for files.
    :param output_file: The file where the list of filenames will be saved.
    :param search_extension: The file extension to look for. Default is '.obj.npy'.
    :param add_extension: The extension to add to each filename in the list. Default is '.obj'.
    """

    # List to hold the filenames without extension
    filenames_without_extension = []

    # Loop through the files in the directory
    for filename in os.listdir(directory):
        # Check if the file follows the specific naming convention
        if filename.endswith(search_extension):
            # Strip the search extension and add the new extension
            name_without_extension = filename.rsplit('.', 2)[0] + add_extension
            filenames_without_extension.append(name_without_extension)

    # Write the list to a file
    with open(output_file, 'w') as file:
        for name in filenames_without_extension:
            file.write(name + '\n')

    print(f"List of filenames has been written to {output_file}")