import torch
import torch.nn as nn

from .configs import TransINRConfig
from .modules.coord_sampler import CoordSampler
from .modules.data_encoder import DataEncoder
from .modules.hyponet import HypoNet
from .modules.latent_mapping import LatentMapping
from .modules.weight_groups import WeightGroups
from .modules import diff_operators


class TransINR(nn.Module):
    Config = TransINRConfig

    def __init__(self, config: TransINRConfig):
        super().__init__()
        self.config = config = config.copy()  # type: TransINRConfig
        self.hyponet_config = config.hyponet

        self.coord_sampler = CoordSampler(config.coord_sampler)

        self.encoder = DataEncoder(config.data_encoder)  # DataEncoder have to be developed
        self.latent_mapping = LatentMapping(config.latent_mapping, input_dim=self.encoder.output_dim)

        self.hyponet = HypoNet(config.hyponet)

        self.weight_groups = WeightGroups(
            self.hyponet.params_shape_dict,
            num_groups=config.n_weight_groups,
            weight_dim=config.transformer.embed_dim,
            modulated_layer_idxs=config.modulated_layer_idxs,
        )

        self.num_group_total = self.weight_groups.num_group_total
        self.group_modulation_postfc = nn.ModuleDict()  # pass nn.Linear(embed_dim, shape[0]-1)
        for name, shape in self.hyponet.params_shape_dict.items():
            if name not in self.weight_groups.group_idx_dict:
                continue
            postfc_input_dim = self.config.transformer.embed_dim
            postfc_output_dim = shape[0] - 1 if self.hyponet.use_bias else shape[0]

            self.group_modulation_postfc[name] = nn.Sequential(
                nn.LayerNorm(postfc_input_dim), nn.Linear(postfc_input_dim, postfc_output_dim)
            )

    def _init_weights(self, module):
        # other params would be manually initialized.
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif module.bias is not None:
            module.bias.data.zero_()

    def forward(self, xs, coord=None, keep_xs_shape=True):
        """
        Args:
            xs (torch.Tensor): (B, input_dim, *xs_spatial_shape)
            coord (torch.Tensor): (B, *coord_spatial_shape)
            keep_xs_shape (bool): If True, the outputs of hyponet (MLPs) is permuted and reshaped as `xs`
              If False, it returns the outputs of MLPs with channel_last data type (i.e. `outputs.shape == coord.shape`)
        Returns:
            outputs (torch.Tensor): `assert outputs.shape == xs.shape`
        """
        batch_size = xs.shape[0]
        coord = self.sample_coord_input(xs) if coord is None else coord
        xs_emb = self.encode(xs)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        # predict all pixels of coord after applying the modulation_parms into hyponet
        outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)
        if keep_xs_shape:
            permute_idx_range = [i for i in range(1, xs.ndim - 1)]
            outputs = outputs.permute(0, -1, *permute_idx_range)
        return outputs

    def predict_group_modulations(self, group_output, **kwargs):

        modulation_params_dict = dict()
        num_vectors_per_group = self.weight_groups.num_vectors_per_group_dict

        for name in self.hyponet.params_dict.keys():
            if name not in self.wpermuteeight_groups.group_idx_dict:
                continue
            start_idx, end_idx = self.weight_groups.group_idx_dict[name]
            _group_output = group_output[:, start_idx:end_idx]

            # post fc convert the transformer outputs into modulation weights
            _modulation = self.group_modulation_postfc[name](_group_output)
            _modulation = _modulation.transpose(-1, -2)  # (B, fan_in, group_size)
            _modulation = _modulation.repeat(1, 1, num_vectors_per_group[name])  # (B, fan_in, fan_out)
            modulation_params_dict[name] = _modulation
        return modulation_params_dict

    def encode(self, xs, put_channels_last=True):
        return self.encoder(xs, put_channels_last=put_channels_last)

    def encode_latent(self, xs_embed):
        return self.latent_mapping(xs_embed)

    def compute_loss(self, preds, targets,modulation_list=None,label=None,type='occ',coords = None,mode='sum'):
        if type == 'sdf':
            #occ = targets[:,:,0][:,:,None]
            gt_sdf = targets[:,:,0][:,:,None]
            gt_normals = targets[:,:,1:-1]
            gt_labels = targets[:,:,-1]
            gt_occ = (gt_sdf > 0).float()

            pred_sign = preds[:,:,0][:,:,None]
            pred_sdf = preds[:,:,1][:,:,None]
        elif type == 'siren_sdf':
            pred_sign = preds[:,:,0][:,:,None]
            pred_sdf = preds[:,:,1][:,:,None]

            gt_sdf = targets['sdf'].to(pred_sdf.device)
            gt_normals = targets['normals'].to(pred_sdf.device)

        elif type == 'occ':
            gt_sdf = targets[:, :, 0][:, :, None]
            gt_occ = (gt_sdf > 0).float()
            pred_sign = preds[:,:,1][:,:,None]


        batch_size = preds.shape[0]
        sdf_loss = torch.Tensor([0]).squeeze()
        spatial_loss = torch.Tensor([0]).squeeze()
        div_loss = torch.Tensor([0]).squeeze()
        normal_loss = torch.Tensor([0]).squeeze()
        grad_loss = torch.Tensor([0]).squeeze()
        bce_loss = torch.Tensor([0]).squeeze()
        off_surface_loss = torch.Tensor([0]).squeeze()

        if type =='sdf':

            # TODO: truncate predicted sdf between -0.1 and 0.1

            #bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_sign, gt_occ)
            #bce_loss = torch.reshape(bce_loss, (batch_size, -1)).mean(dim=-1)


            #preds = torch.clamp(preds, -0.1, 0.1)
            sdf_loss = torch.nn.functional.l1_loss(pred_sdf,gt_sdf,reduction='none')
            #gradient = diff_operators.gradient(pred_sdf, coords)

            #grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
            #normal_constraint = 1 - torch.nn.functional.cosine_similarity(gradient, gt_normals, dim=-1)

            #normal_constraint = normal_constraint*gt_labels
            #div_constraint = diff_operators.gradient(preds,coords).norm(dim=-1)
            #div_loss = 1*div_constraint.reshape((batch_size, -1)).mean(dim=-1).sum()


            sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1)
            #normal_loss = 1 * normal_constraint.reshape((batch_size, -1)).mean(dim=-1)
            #grad_loss = 1 * grad_constraint.reshape((batch_size, -1)).mean(dim=-1)

            #off_surface_constraint = torch.exp(-20*torch.absolute(pred_sdf[:,:,0]))*(1-gt_labels)
            #off_surface_loss = off_surface_constraint.reshape(batch_size,-1).mean(dim=-1)



            if mode == 'sum':
                sdf_loss = sdf_loss.sum()
                normal_loss=normal_loss.sum()
                grad_loss = grad_loss.sum()
                bce_loss = bce_loss.sum()
                off_surface_loss = off_surface_loss.sum()

            if mode == 'mean':
                sdf_loss = sdf_loss.mean()
                normal_loss = normal_loss.mean()
                grad_loss = grad_loss.mean()
                bce_loss = bce_loss.mean()
                off_surface_loss = off_surface_loss.mean()

                #normal_loss =1 * grad_constraint.reshape((batch_size, -1)).mean(dim=-1).sum()
            #grad_loss = 1 * normal_constraint.reshape((batch_size, -1)).mean(dim=-1).sum()

            total_loss = 1e1*sdf_loss
            psnr = -10 * torch.log10(total_loss)

            pass

        elif type == 'siren_sdf':
            gradient = diff_operators.gradient(pred_sdf, coords)

            # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
            sdf_loss = torch.abs(torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf)))

            off_surface_loss = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf),
                                           torch.exp(-1e2 * torch.abs(pred_sdf)))

            normal_loss = torch.where(gt_sdf != -1,
                                            1 - torch.nn.functional.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                            torch.zeros_like(gradient[..., :1]))

            grad_loss = torch.abs(gradient.norm(dim=-1) - 1)


            sdf_loss = torch.reshape(sdf_loss, (batch_size, -1)).mean(dim=-1)
            off_surface_loss = torch.reshape(off_surface_loss, (batch_size, -1)).mean(dim=-1)
            normal_loss = torch.reshape(normal_loss, (batch_size, -1)).mean(dim=-1)
            grad_loss = torch.reshape(grad_loss, (batch_size, -1)).mean(dim=-1)


            if mode == 'sum':
                sdf_loss = sdf_loss.sum()
                normal_loss=normal_loss.sum()
                grad_loss = grad_loss.sum()
                off_surface_loss=off_surface_loss.sum()

            if mode == 'mean':
                sdf_loss = sdf_loss.mean()
                normal_loss = normal_loss.mean()
                grad_loss = grad_loss.mean()
                off_surface_loss=off_surface_loss.mean()

            total_loss = 3e3*sdf_loss + 1e2*off_surface_loss + 5e1* grad_loss + 1e2 * normal_loss
            psnr = -10 * torch.log10(total_loss)

        elif type == 'occ':
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_sign, gt_occ)
            bce_loss = torch.reshape(bce_loss, (batch_size, -1)).mean(dim=-1)

            if mode == 'sum':
                bce_loss = bce_loss.sum()

            if mode == 'mean':
                bce_loss = bce_loss.mean()

            total_loss = bce_loss
            psnr = -10 * torch.log10(total_loss)

            pass



        return {"loss_total": total_loss, "mse": total_loss, "psnr": psnr,"onsurface_loss":sdf_loss*3e3,"spatial_loss":spatial_loss,"normal_loss":normal_loss*1e2,"grad_loss":grad_loss*5e1,"div_loss":div_loss,"bce_loss":bce_loss,"off_surface_loss":off_surface_loss*3e3}

    def sample_coord_input(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        device = device if device is not None else xs.device
        coord_inputs = self.coord_sampler(xs, coord_range, upsample_ratio, device)
        return coord_inputs

    def predict_modulation_params_dict(self, xs):
        """Computes the modulation parameters for given inputs."""
        batch_size = xs.shape[0]
        xs_emb = self.encode(xs)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        return modulation_params_dict

    def predict_hyponet_params_dict(self, xs):
        """Computes the modulated parameters of hyponet for given inputs."""
        modulation_params_dict = self.predict_modulation_params_dict(xs)
        params_dict = self.hyponet.compute_modulated_params_dict(modulation_params_dict)
        return params_dict

    def forward_with_params(
        self,
        coord,
        keep_xs_shape=True,
        modulation_params_dict=None,
        hyponet_params_dict=None,
    ):
        r"""Computes the output values for coordinates according to INRs specified with either modulation parameters or
        modulated parameters.
        Note: Exactly one of `modulation_params_dict` or `hyponet_params_dict` must be given.

        Args:
            coord (torch.Tensor): Input coordinates in shape (B, ...)
            keep_xs_shape (bool): If True, the outputs of hyponet (MLPs) is permuted and reshaped as `xs`
              If False, it returns the outputs of MLPs with channel_last data type (i.e. `outputs.shape == coord.shape`)
            modulation_params_dict (dict[str, torch.Tensor], optional): Modulation parameters.
            hyponet_params_dict (dict[str, torch.Tensor], optional): Modulated hyponet parameters.
        Returns:
            outputs (torch.Tensor): Evaluated values according to INRs with specified modulation/modulated parameters.
        """
        if (modulation_params_dict is None) and (hyponet_params_dict is None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")
        if (modulation_params_dict is not None) and (hyponet_params_dict is not None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")

        if modulation_params_dict is None:
            assert hyponet_params_dict is not None
            outputs = self.hyponet.forward_with_params(coord, params_dict=hyponet_params_dict)
        else:
            assert hyponet_params_dict is None
            outputs = self.hyponet.forward(coord, modulation_params_dict=modulation_params_dict)

        if keep_xs_shape:
            permute_idx_range = [i for i in range(1, outputs.ndim - 1)]
            outputs = outputs.permute(0, -1, *permute_idx_range)
        return outputs
