import logging
import time

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import plyfile

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate
from utils.utils import render_meshes, render_mesh
import trimesh
import wandb

from model.loss_functions import KL_regularization_loss

logger = logging.getLogger(__name__)


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        n_inner_step = self.config.arch.n_inner_step
        accm = AccmStageINR(
            scalar_metric_names=("loss_total", "mse", "psnr","onsurface_loss","spatial_loss","grad_loss","normal_loss","div_loss","bce_loss","off_surface_loss","pred_max","pred_min","weight_norm","grad_norm_mean", "KL_regularization_loss"),
            vector_metric_names=("inner_loss", "grad_norm"),
            vector_metric_lengths=(n_inner_step, n_inner_step),
            device=self.device,
        )
        return accm

    @torch.no_grad()

    def reconstruct_shape(self,meshes,epoch,it=0,mode='train'):
        ply_data_arr =  []
        ply_filename_out_arr = []
        
        for k in range(len(meshes)):
            # try writing to the ply file
            verts = meshes[k]['vertices']
            faces = meshes[k]['faces']
            voxel_grid_origin = [-0.5] * 3
            mesh_points = np.zeros_like(verts)
            mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
            mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
            mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

            num_verts = verts.shape[0]
            num_faces = faces.shape[0]

            verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

            for i in range(0, num_verts):
                verts_tuple[i] = tuple(mesh_points[i, :])

            faces_building = []
            for i in range(0, num_faces):
                faces_building.append(((faces[i, :].tolist(),)))
            faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

            el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
            el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

            ply_data = plyfile.PlyData([el_verts, el_faces])
            # logging.debug("saving mesh to %s" % (ply_filename_out))
            ply_filename_out = "./results.tmp/ply/" + str(epoch) + "_" +str(mode)+"_"+ str(it*len(meshes)+k) + "_poly.ply"
            ply_data.write(ply_filename_out)
            
            ply_data_arr.append(ply_data)
            ply_filename_out_arr.append(ply_filename_out)
            
        return ply_data_arr, ply_filename_out_arr

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xt in pbar:
            model.zero_grad()
            if self.config.dataset.type == "shapenet":
                if self.config.dataset.supervision == 'sdf'or self.config.dataset.supervision == 'occ':
                    coord_inputs = xt['coords'].to(self.device)
                    coord_inputs.requires_grad_()

                    xs = xt['sdf'].to(self.device)
                    normals = xt['normal'].to(self.device)
                    labels = xt['label'].to(self.device)
                    xs = torch.concatenate([xs,normals,labels],dim=-1)

                elif self.config.dataset.supervision == 'siren_sdf':
                    coords, xs = xt
                    coord_inputs = coords['coords'].to(self.device)
                    coord_inputs.requires_grad_()


            vis = False
            if self.config.dataset.type == 'shapenet':
                outputs, _, collated_history = model(xs, coord_inputs, is_training=False,vis=vis,type=self.config.dataset.supervision)

           # targets = xs.detach()
            loss = model.compute_loss(outputs, xs,type=self.config.dataset.supervision,coords=coord_inputs,mode='mean')


            metrics = dict(
                loss_total=loss["loss_total"],
                onsurface_loss = loss["onsurface_loss"],
                spatial_loss = loss["spatial_loss"],
                grad_loss=loss["grad_loss"],
                normal_loss=loss["normal_loss"],
                div_loss=loss["div_loss"],
                bce_loss=loss["bce_loss"],
                off_surface_loss = loss["off_surface_loss"]


            )
            accm.update(metrics, count=1,sync=True, distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)

        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%sudo apt install python3.7s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            #self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["xs"] = xt

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        total_step = len(self.loader_trn) * epoch

        accm = self.get_accm()
        #self.distenv.master = 0
        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
        else:
            pbar = enumerate(self.loader_trn)


        model.train()


        for it, xt in pbar:
            model.zero_grad(set_to_none=True)
            if self.config.dataset.type == "shapenet":
                if self.config.dataset.supervision == 'sdf' or self.config.dataset.supervision == 'occ':
                    coord_inputs = xt['coords'].to(self.device)
                    coord_inputs.requires_grad_()

                    xs = xt['sdf'].to(self.device)
                    normals = xt['normal'].to(self.device)
                    labels = xt['label'].to(self.device)
                    xs = torch.concatenate([xs,normals,labels],dim=-1)


                elif self.config.dataset.supervision == 'siren_sdf':
                    coords, xs = xt
                    coord_inputs = coords['coords'].to(self.device)
                    coord_inputs.requires_grad_()


            if  self.config.type == 'overfit':
                outputs = model.overfit_one_shape(coord=coord_inputs)
            else:
                outputs, _, collated_history = model(xs, coord=coord_inputs, is_training=True,type=self.config.dataset.supervision)



            loss = model.compute_loss(outputs, xs,type=self.config.dataset.supervision,coords=coord_inputs,mode='mean')
            
            # KL Regularization
            #if self.config.loss.use_KL_regularization:
            #    loss["KL_regularization_loss"] = self.config.loss.KL_reg_lambda * KL_regularization_loss(model.get_init_modulation_factors_overfit()["linear_wb1"], 0.0, 1.0)
            #    loss["loss_total"] += loss["KL_regularization_loss"]

            epoch_loss =float(loss["loss_total"].item())

            loss["loss_total"].backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optimizer.max_gn)
            optimizer.step()

            if scheduler.mode =='adaptive':

                scheduler.step(epoch_loss)
            else:
                scheduler.step(epoch)


            metrics = dict(
                loss_total=loss["loss_total"],
                onsurface_loss = loss["onsurface_loss"],
                spatial_loss = loss["spatial_loss"],
                grad_loss=loss["grad_loss"],
                normal_loss=loss["normal_loss"],

                div_loss=loss["div_loss"],
                bce_loss=loss["bce_loss"],
                off_surface_loss=loss["off_surface_loss"],
                
                #KL_regularization_loss=loss["KL_regularization_loss"]

            )

            # for visualization of inner norm, grad

            if self.config.type != 'overfit':
                linear_items = {key: value for key, value in collated_history.items() if key.startswith('linear')}
                for key, values in linear_items.items():
                    weight_norm = torch.norm(values[0, :, :, :], dim=[1, 2])
                    metrics['weight_norm'] = weight_norm.mean(dim=0)
                    break

                metrics['inner_loss'] = collated_history['loss_total']/coord_inputs.shape[0]
                metrics['grad_norm'] = collated_history["grad"]
                metrics['pred_max']= collated_history['recons'][:,:,:,1].max()
                metrics['pred_min'] = collated_history['recons'][:, :, :, 1].min()
                metrics['grad_norm_mean'] = collated_history['grad'].mean()
            accm.update(metrics, count=1)
            total_step += 1

            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)



        summary = accm.get_summary()
        summary["xs"] = xt
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train", override_imlog=False):
        if (epoch>=0 and epoch % self.config.experiment.test_imlog_freq == 0) or override_imlog:
            if self.config.dataset.type == 'shapenet':
                if self.config.type !='overfit':
                    model = self.model
                    model.eval()

                    xt = summary["xs"]

                    if self.config.dataset.supervision == 'sdf' or self.config.dataset.supervision == 'occ':
                        coord_inputs = xt['coords'].to(self.device)
                        coord_inputs.requires_grad_()

                        xs = xt['sdf'].to(self.device)
                        normals = xt['normal'].to(self.device)
                        labels = xt['label'].to(self.device)
                        xs = torch.concatenate([xs, normals, labels], dim=-1)

                    elif self.config.dataset.supervision == 'siren_sdf':
                        coords, xs = xt
                        coord_inputs = coords['coords'].to(self.device)
                        coord_inputs.requires_grad_()

                    vis = True
                    _, meshes, _ = model(xs, coord_inputs, is_training=False, vis=vis,type=self.config.dataset.supervision)

                else:
                    model = self.model
                    model.eval()
                    meshes = model.overfit_one_shape(type=self.config.dataset.supervision)
                    #self.reconstruct_shape(meshes,epoch,mode=mode)
                    
                super().log_images_from_meshes(meshes=meshes, epoch=epoch, mode=mode)

        self.writer.add_scalar("loss/loss_total", summary["loss_total"], mode, epoch)
        self.writer.add_scalar("loss/onsurface_loss", summary["onsurface_loss"], mode, epoch)
        self.writer.add_scalar("loss/spatial_loss", summary["spatial_loss"], mode, epoch)
        self.writer.add_scalar("loss/grad_loss", summary["grad_loss"], mode, epoch)
        self.writer.add_scalar("loss/normal_loss", summary["normal_loss"], mode, epoch)
        self.writer.add_scalar("loss/div_loss", summary["div_loss"], mode, epoch)
        self.writer.add_scalar("loss/bce_loss", summary["bce_loss"], mode, epoch)
        self.writer.add_scalar("loss/off_surface_loss", summary["off_surface_loss"], mode, epoch)

        self.writer.add_scalar("output/pred_max", summary["pred_max"], mode, epoch)
        self.writer.add_scalar("output/pred_min", summary["pred_min"], mode, epoch)
        self.writer.add_scalar("output/grad_norm_mean", summary["grad_norm_mean"], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr/outer_lr", scheduler.get_last_lr()[0], mode, epoch)
            self.writer.add_scalar("lr/inner_lr", self.model.get_lr(), mode, epoch)



        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)