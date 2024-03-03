import argparse
from collections import defaultdict
import math
import numpy as np

import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb

import utils.dist as dist_utils

from model import create_model
from trainer import create_trainer, STAGE_META_INR_ARCH_TYPE
from dataset import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup

import trimesh
from utils.utils import render_meshes, render_mesh

import yaml


def default_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-m", "--model-config", type=str, default="./configs/meta_learning/low_rank_modulated_meta/imagenette178_meta_low_rank.yaml")
    #parser.add_argument("-m", "--model-config", type=str,default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta.yaml")
    parser.add_argument("-m", "--model-config", type=str,default="./config/shapenet_meta_sdf.yaml")
    parser.add_argument("-r", "--result-path", type=str, default="./exp_week9_meta_bias/")
    parser.add_argument("-t", "--task", type=str, default="test_1")

    #parser.add_argument("-l", "--load-path", type=str, default="/home/umaru/praktikum/changed_version/2023_visionpractical/exp_week9_meta_bias/shapenet_meta_sdf/ssdf_1/epoch2000_model.pt")
    parser.add_argument("-l", "--load-path", type=str,default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true",default=False)
    return parser


def add_dist_arguments(parser):
    parser.add_argument("--world_size", default=0, type=int, help="number of nodes for distributed training")
    parser.add_argument("--local_rank", default=1, type=int, help="local rank for distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--nproc_per_node", default=1, type=int)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--timeout", type=int, default=1, help="time limit (s) to wait for other nodes in DDP")
    return parser

def parse_args():
    parser = default_parser()
    parser = add_dist_arguments(parser)
    args, extra_args = parser.parse_known_args()
    return args, extra_args

def parse_wandb_run_id_from_config(config_file_name):
    # Remove the file extension
    filename_without_extension = config_file_name.rsplit('.', 1)[0]

    # Extract the portion after the first underscore, which contains the key-value pairs
    filename_str = filename_without_extension.split('_config_', 1)[1] if '_config_' in filename_without_extension else ''

    return filename_str

if __name__ == "__main__":
    print(torch.cuda.is_available())
    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    
    #init wandb
    run = wandb.init(
        # Set the project where this run will be logged
        project = "ginr_final_generalize_full_val",
        # Identifier for the current run
        name=parse_wandb_run_id_from_config(args.model_config),
        notes = args.task,
        # Track hyperparameters and run metadata
        config = yaml.safe_load(open(args.model_config))
    )

    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)

    model = create_model(config.arch)
    model = model.to(device)

    if distenv.master:
        print(model)
        profiler.get_model_size(model)
        profiler.get_model_size(model, opt="trainable-only")

    # Checkpoint loading
    if not args.load_path == "" or not config.load_path == "":
        if not args.load_path == "":
            load_path = args.load_path
        
        if not config.load_path == "":
            load_path = config.load_path
        
        ckpt = torch.load(load_path, map_location="cpu")    
        model.load_state_dict(ckpt["state_dict"], strict=False)

        if distenv.master:
            logger.info(f"{load_path} model is loaded")
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError("--load-path must be specified in evaluation or resume mode")

    # Optimizer definition
    if args.eval:
        optimizer, scheduler, epoch_st = None, None, None

    else:
        steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
        steps_per_epoch = steps_per_epoch // config.optimizer.grad_accm_steps


        # tmp patch

        if config.type == 'overfit' and config.arch.type == 'low_rank_modulated_transinr':
            model.init_factor_zero()
            loader_trn = torch.utils.data.DataLoader(
                dataset_trn,
                # sampler=self.sampler_trn,
                shuffle=True,
                pin_memory=True,
                batch_size=config.experiment.batch_size,
                # num_workers=num_workers,
            )
            for xt in loader_trn:
                if config.dataset.type == "shapenet":
                    if config.dataset.supervision == 'sdf' or config.dataset.supervision == 'occ':
                        coord_inputs = xt['coords'].to(device)

                    elif config.dataset.supervision == 'siren_sdf':
                        coords, xs = xt
                        coord_inputs = coords['coords'].to(device)
                model.init_factor(coord_inputs)
                break
            
        # log GT on W&B
        if config.type == 'overfit':
            obj_file = str(config.dataset.folder).replace(".npy", "")
            try:
                mesh = trimesh.load(obj_file)
            except:
                print("GT mesh could not be found")
                run.log(
                    {"images/GT_render": "GT mesh could not be found"}
                )
            #rendered_img, _ = render_mesh(mesh)
            #run.log(
            #        {"images/GT_render": wandb.Image(rendered_img, caption=obj_file)}
            #    )



        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer, config.optimizer.warmup, config.optimizer.warmup.step_size, config.experiment.epochs_cos, distenv
        )
        
        if distenv.master:
            print(optimizer)

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            epoch_st = ckpt["epoch"]

            if distenv.master:
                logger.info(f"Optimizer, scheduler, and epoch is resumed")
                logger.info(f"resuming from {epoch_st}..")
        else:
            epoch_st = 0        

    # Usual DDP setting
    static_graph = config.arch.type in STAGE_META_INR_ARCH_TYPE # use static_graph for high-order gradients in meta-learning
    #model = dist_utils.dataparallel_and_sync(distenv, model, static_graph=static_graph)

    trainer = create_trainer(config)
    trainer = trainer(model, dataset_trn, dataset_val, config, writer, device, distenv, wandb=run)

    if distenv.master:
        logger.info(f"Trainer created. type: {trainer.__class__}")

    run.watch(model, log_freq=config.experiment.test_freq, log="all")
    
    if args.eval:
        trainer.config.experiment.subsample_during_eval = False
        trainer.eval(valid=False, verbose=True)
        trainer.eval(valid=True, verbose=True)
    else:
        #alo = iter(dataset_trn)
        
        #cntr=0
        #for item in alo:
        #    if item['coords'].shape[0] != 5000:
        #        print(item['coords'].shape)
        #        print(item['path'])
        #        cntr +=1
        #print("debug counter:", cntr)
    
        trainer.run_epoch(optimizer, scheduler, epoch_st)

    #dist.barrier()

    if distenv.master:
        writer.close()
