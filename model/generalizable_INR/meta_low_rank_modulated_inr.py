import typing

import einops
import numpy as np
import torch
import torch.nn as nn
import os
import copy
from functools import *
from .configs import MetaLowRankModulatedINRConfig
from .modules.hyponet import HypoNet
from .transinr import TransINR
from .modules.sdf_meshing import create_meshes

Tensor = torch.Tensor
TensorDict = typing.Dict[str, Tensor]


def repeat_along_batch_dim(tensor: Tensor, batch_size: int):
    return einops.repeat(tensor, "... -> b ...", b=batch_size)


class ModulatedParamsFactors(nn.Module):
    def __init__(
        self,
        params_shape_dict,
        use_bias_in_hyponet,
        share_bias_in_hyponet,
        initialization_type,
        ranks,
        modulated_layer_idxs=None,
        use_factorization=True,
    ):
        r"""Class to decompose each modulation parameter W into matrix multiplication of shared factor U and
        instance-specific modulation factor V.

        V is adapted via gradient descent during inner loop, while U and init value of V are trained in outer loop.

        Arguments:
            params_shape_dict: Dictionary of hyponet parameter shapes.
            use_bias_in_hyponet: Whether the hyponet uses bias.
            ranks (list of int): Ranks of each factorization (i.e. fan_in of U and fan_out of V).
                Irrelevant if `use_factorization` is True.
            modulated_layer_idxs: List of modulated layer indices.
            use_factorization: If True, W is factorized into U and V. If False, W = V and shared factor U is set `None`.
        """
        super().__init__()

        if len(ranks) == 1:
            ranks = [ranks[0] for _ in range(len(params_shape_dict))]
        else:
            assert len(ranks) == len(params_shape_dict)

        if modulated_layer_idxs is None:
            modulated_layer_idxs = list(range(len(params_shape_dict)))
        else:
            assert len(modulated_layer_idxs) > 0

        self.init_modulation_factors = nn.ParameterDict()
        self.shared_factors = nn.ParameterDict()

        for idx, (name, shape) in enumerate(params_shape_dict.items()):


            if idx not in modulated_layer_idxs:
                continue

            if use_bias_in_hyponet: # base param include bias
                fan_in = (shape[0] - 1)
            else:
                fan_in = shape[0]

            fan_out = shape[1]
            rank = min(ranks[idx], fan_out)



            if use_factorization:
                init_modulation_factor = torch.randn(fan_in, rank)
                init_shared_factor = torch.randn(rank, fan_out) / np.sqrt(rank * fan_in)

                self.init_modulation_factors[name] = nn.Parameter(init_modulation_factor)

                self.shared_factors[name] = nn.Parameter(init_shared_factor)
            else:
                # rank is irrelevant in this case
                #init_modulation_factor = torch.randn(fan_in, fan_out) / np.sqrt(fan_in)


                # kaiming uniform
                if initialization_type.weight_init_type== 'kaiming_uniform':
                    bound = np.sqrt(3.0) / np.sqrt(fan_in)
                    # init_modulation_factor = torch.rand(fan_in, fan_out)*2*bound-bound
                    # KL regularization init trial with n. distr
                    init_modulation_factor = torch.rand(fan_in, fan_out)
                    init_bias = torch.zeros(1,fan_out)

                elif initialization_type.weight_init_type== 'siren':
                    #siren
                    if idx != 0:
                        w0 =30
                        w_std = np.sqrt(6.0 / fan_in) / w0
                        init_modulation_factor = (torch.rand((fan_in, fan_out))*2*w_std-w_std)
                    else:
                        w_std = 1 / fan_in
                        init_modulation_factor = (torch.rand((fan_in, fan_out)) * 2 * w_std - w_std)
                    init_bias = torch.randn(1, fan_out) * 2 * w_std - w_std
                else:
                    print("Incorrect Initialization")


                # initialize bias
                if not share_bias_in_hyponet:
                    init_modulation_factor = torch.concatenate([init_modulation_factor,init_bias],dim=0)
                self.init_modulation_factors[name] = nn.Parameter(init_modulation_factor)
                self.shared_factors[name] = None



    def compute_modulation_params_dict(self, modulation_factors_dict):
        r"""Computes modulation param W by multiplying shared factor U and modulation factor V.
        If shared factor U is None (i.e. `use_factorization` was False), W = V.
        """
        modulation_params_dict = {}

        for name, modulation_factor in modulation_factors_dict.items():
            shared_factor = self.shared_factors[name]
            if shared_factor is not None:
                shared_factor = repeat_along_batch_dim(shared_factor, batch_size=modulation_factor.shape[0])
                modulation_param = torch.bmm(modulation_factor, shared_factor)
            else:
                modulation_param = modulation_factor
            modulation_params_dict[name] = modulation_param

        return modulation_params_dict

    def compute_modulation_params_dict_overfit(self, modulation_factors_dict):
        r"""Computes modulation param W by multiplying shared factor U and modulation factor V.
        If shared factor U is None (i.e. `use_factorization` was False), W = V.
        """

        modulation_params_dict = {}

        for name, modulation_factor in modulation_factors_dict.items():
            shared_factor = self.shared_factors[name]
            if shared_factor is not None:
                modulation_param = torch.matmul(modulation_factor, shared_factor)
            else:
                modulation_param = modulation_factor
            modulation_params_dict[name] = modulation_param

        return modulation_params_dict




    @property
    def modulated_param_names(self):
        return list(self.shared_factors.keys())


class MetaLowRankModulatedINR(TransINR):
    r"""
    `class MetaLowRankModulatedINR` is an optimization-based meta-learner for INR modulation.
    While only one weight matrix is adapted to each data instance for modulating a hyponetwork with a coordinate-based MLP, 
    the remaining weights are trained over data during outer loop.
    Please refer to Algorithm 1 in our paper (https://arxiv.org/abs/2211.13223) for more details.
    """
    Config = MetaLowRankModulatedINRConfig

    def __init__(self, config: MetaLowRankModulatedINRConfig):
        super(TransINR, self).__init__()
        self.config = config
        self.hyponet = HypoNet(config.hyponet)


        # weight factors
        self.factors = ModulatedParamsFactors(
            self.hyponet.params_shape_dict,
            use_bias_in_hyponet=self.hyponet.use_bias,
            share_bias_in_hyponet=self.hyponet.share_bias,
            initialization_type= self.hyponet.init_config,
            ranks=config.rank,
            modulated_layer_idxs=config.modulated_layer_idxs,
            use_factorization=config.use_factorization,
        )


        for name in self.factors.modulated_param_names:
            # We always ignore base param so that each modulated weight W is directly computed
            self.hyponet.ignore_base_param_dict[name] = True


        self.n_inner_step = self.config.n_inner_step
        self.inner_lr = copy.copy(self.config.inner_lr)
        #self.lr = torch.nn.Parameter(torch.Tensor([copy.copy(self.config.inner_lr)]))
        #self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([copy.copy(self.config.inner_lr)]))
        #                            for _ in range(self.config.n_inner_step)])
        #self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([copy.copy(self.config.inner_lr)]))
        #                       for _ in config.modulated_layer_idxs])



    def get_init_modulation_factors_overfit(self):
        modulation_factors_dict = self.factors.init_modulation_factors
        return modulation_factors_dict

    @torch.enable_grad()
    def get_init_modulation_factors(self, xs: Tensor):
        r"""Returns the initial modulation factors."""
        modulation_factors_dict = self.factors.init_modulation_factors
        modulation_factors_dict = {
            name: repeat_along_batch_dim(factor, xs.shape[0]) for name, factor in modulation_factors_dict.items()
        }
        return modulation_factors_dict


    def predict_with_modulation_factors(self, xs, modulation_factors_dict, coord=None):
        r"""Inference function on Hyponet, modulated via given modulation factors."""
        coord = self.sample_coord_input(xs) if coord is None else coord

        # convert modulation factors into modulation params
        modulation_params_dict = self.factors.compute_modulation_params_dict(modulation_factors_dict)

        # predict all pixels of coord after applying the modulation_parms into hyponet
        outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)

        return outputs


    def decode_with_modulation_factors(self, modulation_factors_dict,overfit=False,type='occ'):
        r"""Inference function on Hyponet, modulated via given modulation factors."""
        #coord = self.sample_coord_input(xs) if coord is None else coord

        # convert modulation factors into modulation params

        modulation_params_dict = self.factors.compute_modulation_params_dict(modulation_factors_dict)


        meshes = create_meshes(
                self.hyponet, modulation_params_dict, level=0.0, N=256,overfit=overfit,type=type
                )

        return meshes


    def inner_step(
        self,
        xs: Tensor,
        modulation_factors_dict: TensorDict,
        coord: Tensor,
        inner_lr: Tensor,
        is_training: bool = True,
        type='sdf',
        step=4
    ):
        r"""Single adaptation step of modulation factors via SGD w.r.t. the reconstruction loss for `xs`."""

        with torch.enable_grad():
            # compute reconstruction
            shape = xs[:,:,0] if type=='sdf' or type == 'occ' else xs['sdf'][...,0]
            recons = self.predict_with_modulation_factors(shape, modulation_factors_dict, coord)

            factor_names = list(modulation_factors_dict.keys())
            modulation_factors_list = list(modulation_factors_dict.values())

            # compute the loss
            # reduction should be "sum" here, since we are computing per-sample gradient
            metrics = self.compute_loss(recons,xs,modulation_list=modulation_factors_list,type=type,coords=coord,mode='sum')

            # compute gradient w.r.t. latents
            grads_list = torch.autograd.grad(metrics["loss_total"], modulation_factors_list, create_graph=True)
            # siren sdf may not fit into memory, so we do not create graph
            #grads_list = torch.autograd.grad(metrics["loss_total"], modulation_factors_list, create_graph=False)
            
            # take an SGD step
            new_modulation_factors_dict = {}
            for i, pack in enumerate(zip(factor_names, modulation_factors_list, grads_list)):
                name, factor, grad = pack
                if self.config.hyponet.normalize_weight:
                    lr_scale = factor.norm(dim=[1, 2], keepdim=True).pow(2.0)
                    #lr_scale = 1.0
                    #lr_scale = 1.0 / grad.norm().pow(2.0)
                else:
                    lr_scale = 1.0
                    lr_scale = factor.norm(dim=[1, 2], keepdim=True).pow(2.0)

                #print("factor_norm:"+str(factor.norm(dim=[1, 2], keepdim=True).mean().item()/factor.shape[0]))
                #print("grad_factor:"+str(grad.norm(dim=[1, 2], keepdim=True).mean().item()/grad.shape[0]))

                new_factor = factor - self.inner_lr * lr_scale * grad
                #print("new_factor_norm:" + str(new_factor.norm(dim=[1, 2], keepdim=True).mean().item()/factor.shape[0]))

                #print("-------")
                #print("-------")
                new_modulation_factors_dict[name] = new_factor


        # only for logging
        logs = {
            **{f"{key}_mod": value.detach().clone() for key, value in modulation_factors_dict.items()},
            "recons": recons.detach().clone(),
            "loss_total": metrics["loss_total"].detach().clone(),
            "mse": metrics["mse"].detach().clone(),
            "psnr": metrics["psnr"].detach().clone(),
            "grad": torch.norm(grad,dim=[1,2]).mean().detach().clone(),

        }
        return new_modulation_factors_dict, logs

    def inner_loop(self, xs, coord=None, n_inner_step=1, inner_lr=0.1, is_training=True,type='sdf'):
        r"""A loop of latent adaptation steps, served as the inner loop of meta-learning."""

        # We assume that inner loop uses the coords of shape identical to the spatial shape of xs, while not using
        # coordinate subsampling. For this reason, we compute `coord` from `xs` in the inner loop.
        shape = xs[:,:,0] if type == 'sdf' or type == 'occ' else xs['sdf'][...,0]

        #coord = self.sample_coord_input(xs)if coord is None else coord
        modulation_factors_dict = self.get_init_modulation_factors(shape)

        inner_loop_history = []

        for step_idx in range(n_inner_step):
            modulation_factors_dict, logs = self.inner_step(
                xs, modulation_factors_dict, coord, inner_lr, is_training=is_training,type=type,step=step_idx
            )
            inner_loop_history.append(logs)

        return modulation_factors_dict, inner_loop_history

    def _collate_inner_loop_history(self, inner_loop_history):
        r"""Reorganize `inner_loop_history` which is list of dicts for logging from each inner step.
        Metrics (scalars) are stacked along dim=0, while other tensors (images or modulation factors) are
        stacked along dim=1. Returns the dictionary which looks like:
            {
                ...
                "recons": tensor of shape (batch_size, n_inner_step, 3, H, W)
                "psnr": tensor of shape (n_inner_step,)
                ...
            }
        """
        keys = inner_loop_history[0].keys()
        collated = {}
        for key in keys:
            tensors = [dict_[key] for dict_ in inner_loop_history]
            is_scalar = tensors[0].ndim == 0
            if is_scalar:
                tensors = torch.stack(tensors, dim=0)
            else:
                tensors = torch.stack(tensors, dim=1)
            collated[key] = tensors
        return collated

    def get_lr(self):
        return self.inner_lr

    def forward(self, xs, coord=None, n_inner_step=None, inner_lr=None, is_training=None,vis=False,type='sdf'):
        r"""Infers the signal values at the given coordinates, after an inner loop adapted for `xs`.

        Arguments:
            xs (Tensor): data which is used to adapt latents.
            coord (Tensor, optional): coordinates to infer signal values.
            n_inner_step (int, optional): number of inner steps. (Default: `self.n_inner_step`)
            inner_lr (float, optional): learning rate used in inner steps. (Default: `self.inner_lr`)
            is_training (bool, optional): indicates whether it is in training context. (Default: `self.training`)
        """
        shape = xs[:,:,0] if type == 'sdf' or type == 'occ' else xs['sdf'][...,0]

        coord = self.sample_coord_input(xs) if coord is None else coord

        n_inner_step = self.n_inner_step if n_inner_step is None else n_inner_step
        inner_lr = self.inner_lr if inner_lr is None else inner_lr
        is_training = self.training if is_training is None else is_training

        modulation_factors_dict, inner_loop_history = self.inner_loop(
            xs, coord = coord, n_inner_step=n_inner_step, inner_lr=inner_lr, is_training=is_training,type=type
        )

        outputs = self.predict_with_modulation_factors(shape, modulation_factors_dict, coord)

        collated_history = self._collate_inner_loop_history(inner_loop_history)

        if vis:
            visuals = self.decode_with_modulation_factors(modulation_factors_dict,type=type)
            return outputs, visuals, collated_history

        else:
            return outputs, modulation_factors_dict, collated_history


    def overfit_one_shape(self, coord=None,type='occ'):
        modulation_factors_dict = self.get_init_modulation_factors_overfit()
        # convert modulation factors into modulation params

        if coord is None:
            visuals = self.decode_with_modulation_factors(modulation_factors_dict, overfit=True,type=type)
            return visuals
        else:
            modulation_params_dict = self.factors.compute_modulation_params_dict_overfit(modulation_factors_dict)
            outputs = self.hyponet.forward_overfit(coord, modulation_params_dict=modulation_params_dict)
            return outputs

