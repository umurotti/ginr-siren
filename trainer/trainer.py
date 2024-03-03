import os
import logging
import time
import wandb

import torch

from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import trimesh
from utils.utils import render_meshes, render_mesh

logger = logging.getLogger(__name__)
SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


class TrainerTemplate:
    def __init__(
        self,
        model,
        dataset_trn,
        dataset_val,
        config,
        writer,
        device,
        distenv,
        model_aux=None,
        wandb:wandb=None,
        **kwargs,
    ):
        super().__init__()

        num_workers = 20

        if SMOKE_TEST:
            if not torch.distributed.is_initialized():
                num_workers = 0
            config.experiment.test_freq = 1
            config.experiment.save_ckpt_freq = 1

        self.model = model

        self.model_aux = model_aux

        self.config = config
        self.writer = writer
        self.device = device
        self.distenv = distenv

        self.dataset_trn = dataset_trn
        self.dataset_val = dataset_val

        self.loader_trn = DataLoader(
            self.dataset_trn,
            #sampler=self.sampler_trn,
            shuffle=True,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            #num_workers=num_workers,
        )

        self.loader_val = DataLoader(
            self.dataset_val,
            #sampler=self.sampler_val,
            shuffle=True,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            #num_workers=num_workers,
        )

        self._scaler = None
        
        self.run = wandb

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        raise NotImplementedError

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        raise NotImplementedError

    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = GradScaler(enabled=self.config.experiment.amp)
        return self._scaler

    def run_epoch(self, optimizer=None, scheduler=None, epoch_st=0):
        scaler = self.scaler
        prev_loss = None
        for i in range(epoch_st, self.config.experiment.epochs):
            torch.cuda.empty_cache()
            print("start_training")
            # training process
            summary_trn = self.train(optimizer, scheduler, scaler, epoch=i)


            if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                torch.cuda.empty_cache()

                # do not validate for overfit process
                if self.config.type != 'overfit':
                    summary_val = self.eval(epoch=i)

            if self.distenv.master:
                #logging: write summary and produce shape
                self.logging(summary_trn, scheduler=scheduler, epoch=i + 1, mode="train")
                self.run.log({"train/metrics": summary_trn.metrics}, step = i)

                if self.config.type != 'overfit':
                    if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                        self.logging(summary_val, scheduler=scheduler, epoch=i + 1, mode="valid")
                        self.run.log({"val/metrics": summary_val.metrics}, step = i)
                #if overfit
                #else:
                """                     
                if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                summary_val = self.eval(epoch=i)
                self.logging(summary_val, scheduler=scheduler, epoch=i + 1, mode="valid")
                self.run.log({"val/metrics": summary_val.metrics}, step = i)
                """
                
                # save ckpt
                if (i + 1) % self.config.experiment.save_ckpt_freq == 0:
                    self.save_ckpt(optimizer, scheduler, i + 1)

                if self.config.optimizer.use_patience:
                    if prev_loss != None and \
                        abs(prev_loss -  summary_trn["loss_total"]) < self.config.optimizer.patience_treshold:
                        print(f"Stopping due to patience at epoch={i} w/ prev_loss={prev_loss.item()} and loss_total={summary_trn['loss_total'].item()}")
                        #if self.config.type == 'overfit':
                        #    model = self.model
                        #    model.eval()
                        #    meshes = model.overfit_one_shape(type=self.config.dataset.supervision)
                        #    self.log_images_from_meshes(meshes=meshes, epoch=i)
                        #elif self.config.type == 'generalize':
                        self.logging(summary_trn, scheduler=scheduler, epoch=i + 1, mode="train", override_imlog=True)
                        self.run.log({"train/metrics": summary_trn.metrics}, step = i)
                            #self.log_images_from_meshes(meshes=meshes, epoch=i)
                        self.save_ckpt(optimizer, scheduler, i + 1)
                        break
                    prev_loss = summary_trn["loss_total"]

    def save_ckpt(self, optimizer, scheduler, epoch):
        if self.config.type == "generalize":
            ckpt_path = os.path.join(self.config.result_path, "generalize", "epoch%d_model.pt" % epoch)
        else:
            obj_name = str(self.config.dataset.folder).split(".")[-3].split("/")[-1]
            ckpt_path = os.path.join(self.config.result_path, obj_name, "epoch%d_model.pt" % epoch)
        logger.info("epoch: %d, saving %s", epoch, ckpt_path)
        ckpt = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(ckpt, ckpt_path)

    def log_images_from_meshes(self, meshes, epoch, mode="train"):        
        ply_data_arr, ply_filename_out_arr = self.reconstruct_shape(meshes,epoch,mode=mode)
        wandb_out_imgs = []
        for i, mesh_i in enumerate(meshes):
            trimesh_mesh_i = trimesh.Trimesh(mesh_i["vertices"], mesh_i["faces"])
            out_img_i, _ = render_mesh(trimesh_mesh_i)
            caption_i = ply_filename_out_arr[i].split('/')[-1]
            wandb_out_imgs.append(wandb.Image(out_img_i, caption=caption_i))
        self.run.log(
            {"images/generated_renders": wandb_out_imgs}, step=epoch
        )
