#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from .torchdiffeq import odeint, odeint_adjoint

class ode_func_class(torch.nn.Module):
    def __init__(self, decoder, time_dependent=False):
        super(ode_func_class, self).__init__()
        self.decoder = decoder
        self.time_dependent = time_dependent
    def forward(self, t, y, conditioning):
        if self.time_dependent:
            y = torch.cat([conditioning, y, t.repeat(y.shape[0]).view(-1,1)], dim=1)
        else:
            y = torch.cat([conditioning, y], dim=1)
        out = self.decoder.patch_network_forward(y, ode=False)
        return out

class RegularizedODEfunc(torch.nn.Module):
    # based on https://github.com/rtqichen/ffjord/blob/bce4d2def767f2b9a3288ae0b5d43781ad4dc6b1/lib/layers/wrappers/cnf_regularization.py
    def __init__(self, odefunc, regularization_fns):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = regularization_fns
        self.num_regularizations = len(self.regularization_fns)

    def forward(self, t, x, conditioning):
        class SharedContext(object):
            pass

        assert len(conditioning) == 1
        conditioning = conditioning[0]

        with torch.enable_grad():
            [state.requires_grad_(True) for state in x]
            input_states = x[:-self.num_regularizations]
            assert len(input_states) == 1
            input_states = input_states[0]
            y = self.odefunc(t, input_states, conditioning)
            reg_states = tuple(func(input_states, conditioning, y, SharedContext) for name, weight, func in self.regularization_fns)
            return (y,) + reg_states

def _batch_root_mean_squared(tensor):
    tensor = tensor.view(tensor.shape[0], -1)
    return torch.mean(torch.norm(tensor, p=2, dim=1) / tensor.shape[1]**0.5)


def l1_regularzation_fn(input_states, conditioning, y, unused_context):
    del input_states, conditioning
    return torch.mean(torch.abs(y))


def l2_regularzation_fn(input_states, conditioning, y, unused_context):
    del input_states, conditioning
    return _batch_root_mean_squared(y)


def directional_l2_regularization_fn(input_states, conditioning, y, unused_context):
    directional_dx = torch.autograd.grad(y, input_states, y, create_graph=True)[0]
    return _batch_root_mean_squared(directional_dx)


def jacobian_frobenius_regularization_fn(input_states, conditioning, y, context):
    if hasattr(context, "jac"):
        jac = context.jac
    else:
        jac = _get_minibatch_jacobian(y, input_states)
        context.jac = jac
    return _batch_root_mean_squared(jac)

def divergence_approx(input_states, conditioning, y, context, as_loss=True):
    # avoids explicitly computing the Jacobian
    del conditioning
    if hasattr(context, "e_dydx"):
        e = context.e
        e_dydx = context.e_dydx
    else:
        e = torch.randn_like(y)
        e_dydx = torch.autograd.grad(y, input_states, e, create_graph=True)[0]
        context.e = e
        context.e_dydx = e_dydx
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.view(y.shape[0], -1).sum(dim=1)
    if as_loss: # want to push positive and negative divergence to zero
        approx_tr_dydx = torch.abs(approx_tr_dydx)
    return torch.mean(approx_tr_dydx)

def jacobian_frobenius_approx(input_states, conditioning, y, context):
    # avoids explicitly computing the Jacobian. see https://arxiv.org/pdf/2002.02798.pdf "How to train your Neural ODE" for more
    del conditioning
    if hasattr(context, "e_dydx"):
        e = context.e
        e_dydx = context.e_dydx
    else:
        e = torch.randn_like(y)
        e_dydx = torch.autograd.grad(y, input_states, e, create_graph=True)[0]
        context.e = e
        context.e_dydx = e_dydx
    approx_jac_frob = torch.norm(e_dydx, p=2, dim=-1)
    return torch.mean(approx_jac_frob)

def _get_minibatch_jacobian(y, x, create_graph=False):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac

class Decoder(nn.Module):
    def __init__(
        self,
        patch_latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        patch_encoder=None,
        do_code_regularization=True,
        # mixture
        num_patches=1,
        mixture_latent_size=None,
        non_variable_patch_radius=None,
        use_rotations=True,
        pull_patches_to_uncovered_surface=False,
        pull_free_space_patches_to_surface=False,
        loss_on_patches_instead_of_mixture=False,
        align_patch_rotation_with_normal=False,
        weight_threshold=None,
        train_patch_network=True,
        train_object_to_patch=True,
        patch_network_pretrained_path=None,
        results_folder=None,
        script_mode=False,
        mixture_latent_mode="all_implicit",
        posrot_latent_size=None,
        variable_patch_scaling=False,
        keep_scales_small=False,
        scales_low_variance=False,
        mixture_to_patch_parameters=None,
        use_depth_encoder=False,
        use_tiny_patchnet=False,
        positional_encoding=False,
        pretrained_depth_encoder_weights=None,
        use_curriculum_weighting=False,
        minimum_scale=0.0,
        maximum_scale=1000.,
        use_ode=False,
        time_dependent_ode=False,
        device=None
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        self.use_ode = use_ode
        self.time_dependent_ode = time_dependent_ode

        self.positional_encoding = positional_encoding
        self.positional_encoding_frequencies = 3 # this is "L" in Neural Radiance Fields. NeRF uses L = 4.
        self.use_tiny_patchnet = use_tiny_patchnet

        original_coordinate_size = 4 if use_ode and time_dependent_ode else 3
        coordinate_size =  original_coordinate_size*2*self.positional_encoding_frequencies if positional_encoding else original_coordinate_size
        self.dims = dims = [(0 if use_tiny_patchnet else patch_latent_size) + coordinate_size] + dims + [3 if self.use_ode else 1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        self.do_code_regularization = do_code_regularization
        self.use_curriculum_weighting = use_curriculum_weighting

        self.num_patches = num_patches
        self.mixture_latent_mode = mixture_latent_mode
        self.posrot_latent_size = posrot_latent_size
        self.non_variable_patch_radius = non_variable_patch_radius
        self.variable_patch_scaling = variable_patch_scaling
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.keep_scales_small = keep_scales_small
        self.weight_threshold = weight_threshold
        self.use_rotations = use_rotations
        self.pull_patches_to_uncovered_surface = pull_patches_to_uncovered_surface
        self.pull_free_space_patches_to_surface = pull_free_space_patches_to_surface
        self.loss_on_patches_instead_of_mixture = loss_on_patches_instead_of_mixture
        self.align_patch_rotation_with_normal = align_patch_rotation_with_normal
        self.scales_low_variance = scales_low_variance
        self.script_mode = script_mode
        self.results_folder = results_folder

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= coordinate_size
            print(layer, dims[layer], out_dim)
            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "patch_lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "patch_lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "patch_bn" + str(layer), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        self.patch_latent_size = patch_latent_size
        self.mixture_latent_size = mixture_latent_size

        self.use_depth_encoder = use_depth_encoder
        if self.use_depth_encoder:
            from networks.selecsls import Net as depth_encoder
            self.depth_encoder = depth_encoder(nClasses=self.mixture_latent_size)
            weights = pretrained_depth_encoder_weights
            pretrained_weights = torch.load(pretrained_depth_encoder_weights)
            del pretrained_weights["classifier.0.weight"]
            del pretrained_weights["classifier.0.bias"]
            current_weights = self.depth_encoder.state_dict()
            current_weights.update(pretrained_weights)
            self.depth_encoder.load_state_dict(current_weights)


        self.patch_encoder = patch_encoder
        if patch_encoder is not None:
            in_channels = 4 # xyz + SDF
            for i, (layer_type, param, activation) in enumerate(patch_encoder["layers"]):
                
                if layer_type == "FC":
                    out_channels, weight_norm = param
                    if out_channels <= 0: # special case: this is the last layer. it outputs the patch latent vector
                        out_channels = patch_latent_size
                    if weight_norm:
                        setattr(
                            self,
                            "patch_encoder_lin" + str(i),
                            nn.utils.weight_norm(nn.Linear(in_channels, out_channels)),
                        )
                    else:
                        setattr(self, "patch_encoder_lin" + str(i), nn.Linear(in_channels, out_channels))
                    in_channels = out_channels

        if mixture_to_patch_parameters is None or mixture_latent_mode == "all_explicit":
            self.mixture_to_patch_parameters = None
        else:
            self.mixture_to_patch_parameters = self._parse_mixture_to_patch_parameter_string(mixture_to_patch_parameters)

            import numpy as np
            def _initial_patch_center():
                random_point = np.random.normal(size=(3,))
                radius = 0.5
                return radius * random_point / np.linalg.norm(random_point)

            if self.mixture_latent_mode == "all_implicit":
                if self.num_patches > 1:
                    in_channels = self.mixture_latent_size
                    final_channels = self.num_patches * (self.patch_latent_size + 3 + 3 + (1 if self.variable_patch_scaling else 0))
                    final_bias = np.concatenate([np.concatenate([np.zeros(self.patch_latent_size), _initial_patch_center(), np.zeros(3), (np.array([self.non_variable_patch_radius]) if self.variable_patch_scaling else np.zeros(0))]) for _ in range(self.num_patches)])

                    use_precomputed_bias_init = True
                    if use_precomputed_bias_init:
                        # See readme.txt
                        #sdf_filename = "shapenetv1/deepsdf_preprocessed/SdfSamples/ShapeNetV1/02691156/1b7ac690067010e26b7bd17e458d0dcb.obj.npz" # airplane
                        #sdf_filename = "shapenetv1/deepsdf_preprocessed/SdfSamples/ShapeNetV1/04256520/1731d53ab8b8973973800789ccff9705.obj.npz" # sofa
                        patch_latent_size = self.patch_latent_size
                        num_patches = self.num_patches
                        surface_sdf_threshold = 0.02
                        final_scaling_increase_factor = 1.2
                        initialization_file = sdf_filename + "_init_" + str(patch_latent_size) + "_" + str(num_patches) + "_" + str(surface_sdf_threshold) + "_" + str(final_scaling_increase_factor)  + ("_tiny" if self.use_tiny_patchnet else "") + ".npy" # generated by train_deep_sdf.py
                        final_bias = np.load(initialization_file).reshape(final_channels)
                        initial_patch_latent = self._initialize_tiny_patchnet() if self.use_tiny_patchnet else np.zeros(self.patch_latent_size)
                        for i in range(num_patches):
                            metadata_size = 7
                            offset = (patch_latent_size + metadata_size) * i
                            final_bias[offset : offset + patch_latent_size] = initial_patch_latent
                else: # single patch
                    in_channels = self.mixture_latent_size
                    final_channels = self.patch_latent_size
                    final_bias = self._initialize_tiny_patchnet() if self.use_tiny_patchnet else np.zeros(self.patch_latent_size)

            elif self.mixture_latent_mode == "patch_explicit_meta_implicit":
                in_channels = self.posrot_latent_size
                final_channels = self.num_patches * (3 + 3 + (1 if self.variable_patch_scaling else 0))
                final_bias = np.concatenate([np.concatenate([_initial_patch_center(), np.zeros(3), (np.array([0.3]) if self.variable_patch_scaling else np.zeros(0))]) for _ in range(self.num_patches)])
            else:
                raise RuntimeError("mixture_latent_mode and mixture_to_patch_parameters combination not supported")

            for i, (layer_type, (out_channels, weight_norm), activation, dropout) in enumerate(self.mixture_to_patch_parameters["layers"]):
                if layer_type == "FC":

                    is_final_layer = out_channels == -1
                    if is_final_layer:
                        out_channels = final_channels

                    layer = nn.Linear(in_channels, out_channels)

                    if is_final_layer:
                        layer.weight = nn.Parameter(layer.weight * 0.0001)
                        layer.bias = nn.Parameter(torch.tensor(torch.from_numpy(final_bias)).float().clone().detach())

                    if weight_norm:
                        layer = nn.utils.weight_norm(layer)
                    setattr(self, "object_to_patch_FC_" + str(i), layer)

                    if dropout > 0.:
                        setattr(self, "object_to_patch_dropout_" + str(i), nn.Dropout(dropout))

                    in_channels = out_channels
                else:
                    raise RuntimeError("unknown layer type: " + str(layer_type))

        self._init_patch_network_training(train_patch_network, patch_network_pretrained_path, results_folder)

        for name, weight in self.named_parameters():
            if "patch_" in name: # needs to be before object due to object_to_patch
                weight.requires_grad = train_patch_network
            if "object_" in name:
                weight.requires_grad = train_object_to_patch
            if "depth_encoder" in name:
                weight.requires_grad = train_object_to_patch
            print(name, weight.requires_grad)


    def _init_patch_network_training(self, train_patch_network, patch_network_pretrained_path=None, results_folder=None):
        if patch_network_pretrained_path == "":
            patch_network_pretrained_path = None

        if patch_network_pretrained_path is not None:
            import os
            if results_folder is None:
                this_file = os.path.realpath(__file__)
                results_folder = os.path.split(os.path.split(this_file)[0])[0] + "/"
            copy_folder = results_folder + "backup/pretrained_patch_network/"
            if not os.path.exists(copy_folder + "COPY_DONE"):
                import pathlib
                pathlib.Path(copy_folder).mkdir(parents=True, exist_ok=True)

                root_pretrained = os.path.split(os.path.split(patch_network_pretrained_path)[0])[0] + "/"
                weight_path = patch_network_pretrained_path
                network_path = root_pretrained + "backup/networks/deep_sdf_decoder.py"
                json_path = root_pretrained + "backup/specs.json"
                workspace_path = root_pretrained + "backup/deep_sdf/workspace.py"

                import shutil
                shutil.copyfile(weight_path, copy_folder + "weight.pth")
                shutil.copyfile(network_path, copy_folder + "deep_sdf_decoder.py")
                shutil.copyfile(json_path, copy_folder + "specs.json")
                shutil.copyfile(workspace_path, copy_folder + "workspace.py")
                shutil.copytree(root_pretrained + "backup/", copy_folder + "backup/")

                with open(copy_folder + "COPY_DONE", "w"):
                    pass
                
            pretrained_weights = torch.load(copy_folder + "weight.pth")["model_state_dict"]
            current_weights = self.state_dict()
            prefix_len = len("module.")
            pretrained_weights = {k[prefix_len:]:v for k,v in pretrained_weights.items() if k[prefix_len:] in current_weights}
            current_weights.update(pretrained_weights)
            self.load_state_dict(current_weights)


    def _parse_mixture_to_patch_parameter_string(self, parameter_string):

        layers = []
        for layer in parameter_string.split(","):
            out_channels, weight_norm, activation, dropout = layer.strip().split(" ")
            out_channels = int(out_channels)
            if weight_norm != "wn" and weight_norm != "nown":
                raise RuntimeError("invalid parameter string for mixture-to-patch network: " + str(weight_norm))
            weight_norm = weight_norm == "wn"
            dropout = float(dropout)
            activation = activation.strip().lower()
            layers.append(("FC", (out_channels, weight_norm), activation, dropout))

        return {"layers": layers}


    def _initialize_tiny_patchnet(self):
        import numpy as np

        original_coordinate_size = 4 if self.use_ode and self.time_dependent_ode else 3
        coordinate_size = original_coordinate_size * 2 * self.positional_encoding_frequencies if self.positional_encoding else original_coordinate_size

        weights = []
        for layer in range(0, len(self.dims) - 1):
            input_dim = self.dims[layer]
            output_dim = self.dims[layer+1]
            if layer in self.latent_in:
                input_dim += coordinate_size

            matrix = torch.empty((output_dim, input_dim), dtype=torch.float32)
            bias = torch.empty((output_dim), dtype=torch.float32)

            import math
            torch.nn.init.kaiming_uniform_(matrix, a=math.sqrt(5), nonlinearity="relu")
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(matrix)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(bias, -bound, bound)

            if layer == len(self.dims) - 2:
                matrix /= 1000.
                bias /= 1000.

            weights.append(matrix)
            weights.append(bias)

        initial_patch_latent = np.concatenate([weight.detach().reshape(-1).numpy() for weight in weights], axis=0)
        print("the patch latent size for tiny PatchNets is: " + str(initial_patch_latent.size), flush=True)
        return initial_patch_latent


    def _convert_euler_to_matrix(self, angles): 
            # angles: N x 3
            sine = torch.sin(angles)
            cosine = torch.cos(angles)
        
            sin_alpha, sin_beta, sin_gamma = sine[:,0], sine[:,1], sine[:,2]
            cos_alpha, cos_beta, cos_gamma = cosine[:,0], cosine[:,1], cosine[:,2]

            R00 = cos_gamma*cos_beta
            R01 = -sin_gamma*cos_alpha + cos_gamma*sin_beta*sin_alpha
            R02 = sin_gamma*sin_alpha + cos_gamma*sin_beta*cos_alpha
    
            R10 = sin_gamma*cos_beta
            R11 = cos_gamma*cos_alpha + sin_gamma*sin_beta*sin_alpha
            R12 = -cos_gamma*sin_alpha + sin_gamma*sin_beta*cos_alpha
    
            R20 = -sin_beta
            R21 = cos_beta*sin_alpha
            R22 = cos_beta*cos_alpha

            R0 = torch.stack([R00, R01, R02], 1) # first row
            R1 = torch.stack([R10, R11, R12], 1) # second row
            R2 = torch.stack([R20, R21, R22], 1) # third row

            R = torch.stack([R0, R1, R2], 1) # shape: (batch_size, row, column)
            return R

    def patch_network_forward(self, input, ode=None, ode_evaluation_times=[0., 1.], ode_return_pos=False, extra_loss=None):
        original_coordinate_size = 4 if self.use_ode and self.time_dependent_ode else 3

        xyz = input[:, -original_coordinate_size:] # samples x 3
        device = xyz.get_device()

        if ode is None:
            ode = self.use_ode

        if ode:
            atol = 1e-5
            rtol = 1e-5
            max_num_steps = 1000
            total_remaining_forced_step = 10

            ode_func = ode_func_class(self, self.time_dependent_ode)
            conditioning = input[:, :-3] # do not use original_coordinate_size here, since this is input from the outside, not from the ode solver
            xyz = xyz[:, -3:] # do not use original_coordinate_size here

            self.regularize_ode = True
            regularize = self.training and self.regularize_ode
            if regularize:
                regularization_fns = [     ("velocity_L2", 0.001, l2_regularzation_fn), \
                                           ("jacobian_frobenius", 0.01, jacobian_frobenius_approx),\
                                           ("divergence", 0.001, divergence_approx), \
                                           ]
                ode_func = RegularizedODEfunc(ode_func, regularization_fns)
                xyz = (xyz,) + tuple(torch.tensor(0).to(xyz) for _ in range(len(regularization_fns)))
                rtol = [rtol] + [1e20] * len(regularization_fns)
                atol = [atol] + [1e20] * len(regularization_fns)

            use_adjoint = True
            method = "dopri5" # "dopri5" or "rk4"
            if method == "dopri5":
                options = {"total_remaining_forced_step": total_remaining_forced_step, "max_num_steps": max_num_steps}
            elif method == "rk4":
                options = {"step_size": 0.25}
            else:
                options = None
            evaluation_times = torch.tensor(ode_evaluation_times, device=device)
            odeint_func = odeint_adjoint if use_adjoint else odeint
            ode_pos = odeint_func(ode_func, xyz, conditioning, evaluation_times, rtol=rtol, atol=atol, method=method, options=options) # evaluation_times x samples x xyz

            if regularize:
                ode_returned = ode_pos
                ode_regularization_losses = ode_returned[1:]
                assert extra_loss is not None
                for (name, weight, func), loss in zip(regularization_fns, ode_regularization_losses):
                    extra_loss[name] = weight * loss[-1] # take loss at the last timestep, which should be the right thing to do to get the full loss
                ode_pos = ode_returned[0]

            x = ode_pos[-1,:,2].reshape(-1,1) # take z coordinate as distance to flat patch

        else:

            if self.positional_encoding:
                wrap_around = False
                coordinate_size = original_coordinate_size * 2 * self.positional_encoding_frequencies
                if not wrap_around:
                    xyz = xyz / 2. # scale [-1,+1] to [-0.5,+0.5]
                xyz = math.pi * xyz # samples x 3
                xyz = xyz.view(-1, original_coordinate_size, 1).repeat(1, 1, self.positional_encoding_frequencies) # samples x 3 x L
                xyz *= 2**torch.arange(self.positional_encoding_frequencies, device=device).float().view(1, 1, -1) # samples x 3 x L
                xyz = torch.cat([torch.sin(xyz), torch.cos(xyz)], dim=-1) # samples x 3 x 2L
                xyz = xyz.view(-1, original_coordinate_size * 2 * self.positional_encoding_frequencies)  # samples x 3*2L
                input = torch.cat([input[:,:-original_coordinate_size], xyz], dim=1)
            else:
                coordinate_size = original_coordinate_size
                
            if input.shape[1] > coordinate_size and self.latent_dropout:
                latent_vecs = input[:, :-coordinate_size]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

            if self.use_tiny_patchnet:
                # tiny patchnet:
                # latent vector has parameters for len(self.dims)-1 many layers (this is len(dims)+1 when looking at specs.json).
                # each layer has (input+1)*output many weights (including a bias). the order is: matrix, then bias.
                # the coordinate (xyz or positional encoding) is fed into the first and fourth layer.
                x = input[:, -coordinate_size:] # samples x input_dim
                latents = input[:, :-coordinate_size] # samples x latent
                latent_offset = 0
                for layer in range(0, self.num_layers - 1):
                    input_dim = self.dims[layer]
                    output_dim = self.dims[layer+1]

                    if layer in self.latent_in:
                        input_dim += coordinate_size
                        x = torch.cat([x, input[:,-coordinate_size:]], 1)

                    matrix_weights = latents[:, latent_offset : latent_offset + (input_dim * output_dim)] # samples x (input_dim*output_dim)
                    latent_offset += input_dim * output_dim
                    bias_weights = latents[:, latent_offset : latent_offset + output_dim] # samples x output_dim
                    latent_offset += output_dim

                    matrix_weights = matrix_weights.reshape(-1, output_dim, input_dim) # samples x output_dim x input_dim
                    bias_weights = bias_weights.reshape(-1, output_dim, 1) # samples x output_dim x 1

                    # x: samples x input_dim
                    x = torch.matmul(matrix_weights, x.view(-1, input_dim, 1)) + bias_weights # samples x output_dim x 1
                    x = x.squeeze(-1)

                    if layer < self.num_layers - 2:
                        x = self.relu(x)

            else:
                for layer in range(0, self.num_layers - 1):
                    lin = getattr(self, "patch_lin" + str(layer))
                    if layer in self.latent_in:
                        x = torch.cat([x, input], 1)
                    elif layer != 0 and self.xyz_in_all:
                        x = torch.cat([x, xyz], 1)
                    x = lin(x)
                    if layer < self.num_layers - 2:
                        if (
                            self.norm_layers is not None
                            and layer in self.norm_layers
                            and not self.weight_norm
                        ):
                            bn = getattr(self, "patch_bn" + str(layer))
                            x = bn(x)
                        #x = self.softplus(x)
                        x = self.relu(x)
                        #x = self.elu(x)
                        if self.dropout is not None and layer in self.dropout:
                            x = F.dropout(x, p=self.dropout_prob, training=self.training)

            if hasattr(self, "th") and not self.use_ode:
                x = self.th(x)

        if ode and ode_return_pos:
            return x, ode_pos
        else:
            return x

    # input: N x (L+3)
    def forward(self, input):

        if type(input) != dict:
            #print("not giving a dict as input is dangerous, as some of the losses assume proper scene information to work as intended")
            standard_input = input
            input = {}
            input["num_samp_per_scene"] = 1
            input["standard_input"] = standard_input # samples x (mixture_latent_size + 3)
            if self.mixture_latent_mode == "patch_explicit_meta_implicit":
                input["mixture_latent_vectors"] = standard_input[0,:-3].reshape(1, -1)
                input["xyz"] = standard_input[:, -3:].reshape(1, -1, 3)
                input["num_samp_per_scene"] = input["xyz"].shape[1]
            else:
                input["mixture_latent_vectors"] = standard_input[:, :-3].reshape(-1, input["num_samp_per_scene"], self.mixture_latent_size)
                input["xyz"] = standard_input[:, -3:].reshape(-1, input["num_samp_per_scene"], 3)
            input["use_patch_encoder"] = False
            input["mixture_latent_mode"] = self.mixture_latent_mode

        extra_loss = {}
        mixture_latent_mode = input["mixture_latent_mode"]

        device = input["xyz"].get_device()

        if self.use_depth_encoder:

            depth_maps = input["depth_maps"]
            depth_maps = depth_maps.view(-1, 1, 400, 400).repeat(1,3,1,1) # treat as RGB channels
            encoder_output = self.depth_encoder(depth_maps) # num_scenes x mixture_latent_size

            input["mixture_latent_vectors"] = encoder_output


        if self.num_patches > 1:
            mixture_latent_vectors = input["mixture_latent_vectors"] # num_scenes x mixture_latent_size
            xyz = input["xyz"] # num_scenes x samp_per_scene x 3

            patch_metadata_input = mixture_latent_vectors
            
            # transform input latent vectors to patch metadata
            if mixture_latent_mode == "all_explicit":
                patch_metadata = patch_metadata_input  # num_scenes x (num_patches * (patch_latent_size + 3 + 3 + 1)) # patch latent, position, rotation, scaling
            elif mixture_latent_mode == "patch_explicit_meta_implicit":
                patch_metadata = patch_metadata_input  # num_scenes x (posrot_latent_size + num_patches * patch_latent_size) # patch latent, implicit (position, rotation, scaling)
                _patch_latents = patch_metadata[:,self.posrot_latent_size:].view(-1, self.num_patches, self.patch_latent_size)
                patch_metadata = patch_metadata[:,:self.posrot_latent_size] # num_scenes x posrot_latent_size
                posrot_latents = patch_metadata.clone()
                for i, (layer_type, _, activation, dropout) in enumerate(self.mixture_to_patch_parameters["layers"]):
                    if layer_type == "FC": 
                        layer = getattr(self, "object_to_patch_FC_" + str(i))
                        patch_metadata = layer(patch_metadata)
                        if dropout > 0.:
                            dropout = getattr(self, "object_to_patch_dropout_" + str(i))
                            patch_metadata = dropout(patch_metadata)
                        if activation == "relu":
                            patch_metadata = self.relu(patch_metadata)
                        elif activation == "elu":
                            patch_metadata = self.elu(patch_metadata)
                        elif activation != "none":
                            raise RuntimeError("wrong activation:" + activation)
                # patch_metadata: num_scenes x num_patches * (3 + 3 + 1)
                # _patch_latents: num_scenes x num_patches x patch_latent_size
                num_scenes = patch_metadata.shape[0]
                patch_metadata = torch.cat([_patch_latents, patch_metadata.view(-1, self.num_patches, 3+3+1)], dim=-1).view(num_scenes, -1)

            elif mixture_latent_mode == "all_implicit":
                patch_metadata = patch_metadata_input  # num_scenes x object_latent # implicit (patch latent, position, rotation, scaling)
                
                for i, (layer_type, _, activation, dropout) in enumerate(self.mixture_to_patch_parameters["layers"]):
                    if layer_type == "FC": 
                        layer = getattr(self, "object_to_patch_FC_" + str(i))
                        patch_metadata = layer(patch_metadata)
                        if dropout > 0.:
                            dropout = getattr(self, "object_to_patch_dropout_" + str(i))
                            patch_metadata = dropout(patch_metadata)
                        if activation == "relu":
                            patch_metadata = self.relu(patch_metadata)
                        elif activation == "elu":
                            patch_metadata = self.elu(patch_metadata)
                        elif activation != "none":
                            raise RuntimeError("wrong activation:" + activation)

            patch_metadata = patch_metadata.reshape(-1, self.num_patches, self.patch_latent_size + 3 + 3 + (1 if self.variable_patch_scaling else 0)) # num_scenes x num_patches x (patch_latent_size + 3 + 3 + 1)
            global_xyz = xyz.repeat(1, 1, self.num_patches).view(-1, input["num_samp_per_scene"], self.num_patches, 3) # num_scenes x samp_per_scene x num_patches x 3
        
            patch_latent_vectors = patch_metadata[:, :, :self.patch_latent_size] # num_scenes x num_patches x patch_latent_size
            patch_position = patch_metadata[:, :, self.patch_latent_size:self.patch_latent_size+3] # num_scenes x num_patches x 3
            patch_rotation = patch_metadata[:, :, self.patch_latent_size+3:self.patch_latent_size+6] # num_scenes x num_patches x 3
            if self.variable_patch_scaling:
                patch_scaling = patch_metadata[:, :, self.patch_latent_size+6:self.patch_latent_size+7] # num_scenes x num_patches x 1. this is the scaling of the patch size, i.e. a value of 2 means that the patch's radius is twice as big
                if self.minimum_scale > 0.:
                    #minimum_scaling = 0.01
                    patch_scaling = torch.clamp(patch_scaling, min=self.minimum_scale)
                if self.maximum_scale != 1000.:
                    #maximum_scaling = 0.5
                    patch_scaling = torch.clamp(patch_scaling, max=self.maximum_scale)
            else:
                patch_scaling = self.non_variable_patch_radius * torch.ones((patch_position.shape[0], patch_position.shape[1], 1), device=device)

            fix_metadata_regression_by_decreasing_patch_scale = True
            if fix_metadata_regression_by_decreasing_patch_scale:
                if "current_stage" in input and input["current_stage"] == "1":
                    patch_scaling /= 1.3

            patch_xyz = global_xyz.clone()
            patch_xyz -= patch_position.unsqueeze(1) # num_scenes x samp_per_scene x num_patches x 3

            unscaled_center_distances_nonflat = torch.norm(patch_xyz, dim=-1) # num_scenes x samp_per_scene x num_patches
            scaled_center_distances_nonflat = unscaled_center_distances_nonflat / patch_scaling.squeeze(-1).unsqueeze(1)
            scaled_center_distances = scaled_center_distances_nonflat.flatten() # scaled distances to patch center
            unscaled_center_distances = unscaled_center_distances_nonflat.flatten() # unscaled distances to patch center
            patch_weight_type = "gaussian"
            if patch_weight_type == "binary":
                patch_weights = (scaled_center_distances < 1.).to(torch.float).detach() # num_scenes * samp_per_scene * num_patches
            elif patch_weight_type == "gaussian":
                std_cutoff = 3. 
                smooth_patch_seam_weights = True
                patch_weight_std = 1. / std_cutoff
                import numpy as np
                distances_to_use = scaled_center_distances # if self.patch_scaling else unscaled_center_distances
                patch_weights = torch.zeros_like(scaled_center_distances)
                patch_mask = scaled_center_distances < 1. 
                patch_weights[patch_mask] = torch.exp(  -0.5 * (scaled_center_distances[patch_mask]/patch_weight_std)**2   ) - (np.exp(-0.5 * std_cutoff**2) if smooth_patch_seam_weights else 0.) # samples * num_patches
                patch_weights[~patch_mask] = 0.
            else:
                raise RuntimeError("missing patch_weight_type")

            patch_weights = patch_weights.view(-1, self.num_patches) # samples x num_patches
            patch_weight_normalization = torch.sum(patch_weights, 1) # samples
            patch_weight_norm_mask = patch_weight_normalization == 0.
            patch_weights[patch_weight_norm_mask,:] = 0.0
            patch_weights[~patch_weight_norm_mask,:] = patch_weights[~patch_weight_norm_mask,:] / patch_weight_normalization[~patch_weight_norm_mask].unsqueeze(-1)
            patch_weights = patch_weights.view(-1) # samples * num_patches

            if self.use_rotations:
                rotations = self._convert_euler_to_matrix(patch_rotation.reshape(-1, 3)).view(-1, self.num_patches, 3, 3) # num_scenes x num_patches x 3 x 3
                # first argument: num_scenes x 1 x num_patches x 3 x 3
                # second argument: num_scenes x samp_per_scene x num_patches x 3 x 1
                patch_xyz = torch.matmul(torch.transpose(rotations, -2, -1).unsqueeze(1), patch_xyz.unsqueeze(-1)) # num_scenes x samp_per_scene x num_patches x 3 x 1
            
            # patch_scaling: num_scenes x num_patches x 1
            # patch_xyz: num_scenes x samp_per_scene x num_patches x 3 x 1
            patch_xyz /= patch_scaling.unsqueeze(1).unsqueeze(-1)

            if input["use_patch_encoder"]:
                sdf_gt = input["sdf_gt"] # num_scenes x samp_per_scene x 1
                per_patch_sdf_gt = sdf_gt / patch_scaling.unsqueeze(1).squeeze(-1) # num_scenes x samp_per_scene x num_patches
                patch_encoder_input = torch.cat([patch_xyz.squeeze(-1), per_patch_sdf_gt.unsqueeze(-1)], dim=-1) # num_scenes x samp_per_scene x num_patches x 4
                patch_encoder_input = patch_encoder_input.view(-1, 4) # samples * num_patches x 4
                
                optimize_input_of_patch_encoder = True
                if optimize_input_of_patch_encoder:
                    samples_in_patches_mask = patch_weights > self.weight_threshold
                    patch_encoder_input = patch_encoder_input[samples_in_patches_mask,:]

                x = patch_encoder_input
                for i, (layer_type, param, activation) in enumerate(self.patch_encoder["layers"]):

                    if layer_type == "FC":
                        lin = getattr(self, "patch_encoder_lin" + str(i))
                        x = lin(x)
                        if activation == "relu":
                            x = self.relu(x)

                    if layer_type == "max":
                        activation = param
                        
                        features_size = x.shape[-1]
                        x = self.relu(x) # some_samples x features_size. this allows to use "0" as a dummy value in helper_tensor_for_max

                        num_scenes = sdf_gt.shape[0]
                        helper_tensor_for_max = torch.zeros((num_scenes, input["num_samp_per_scene"], self.num_patches, features_size), device=device)
                        original_shape_helper_tensor = helper_tensor_for_max.shape
                        helper_tensor_for_max = helper_tensor_for_max.view(-1, features_size)
                        helper_tensor_for_max[samples_in_patches_mask,:] = x
                        helper_tensor_for_max = helper_tensor_for_max.view(original_shape_helper_tensor)

                        x, _ = torch.max(helper_tensor_for_max, dim=1) # num_scenes x num_patches x features_size
                        x = x.view(-1, features_size) # num_scenes * num_patches x features_size

                        if activation == "relu": # redundant with previous relu
                            x = self.relu(x)

                patch_latent_vectors = x.view(-1, self.num_patches, self.patch_latent_size) # num_scenes x num_patches x patch_latent_size

            patch_xyz = patch_xyz.squeeze(-1)
            repeated_patch_latent_vectors = patch_latent_vectors.repeat(1, input["num_samp_per_scene"], 1).view(-1, self.patch_latent_size) # num_scenes * samp_per_scene * num_patches x patch_latent_size
            input["standard_input"] = torch.cat([repeated_patch_latent_vectors, patch_xyz.view(-1, 3)], 1)  # num_scenes * samp_per_scene * num_patches x (patch_latent_size + 3)

            optimize_input_of_patch_network = True
            if optimize_input_of_patch_network:
                samples_in_patches_mask = patch_weights > self.weight_threshold
                input["standard_input"] = input["standard_input"][samples_in_patches_mask,:]

        else: # single patch

            mixture_latent_vectors = input["mixture_latent_vectors"] # num_scenes x mixture_latent_size

            if mixture_latent_mode == "all_explicit":
                patch_latent_vectors = mixture_latent_vectors

            elif mixture_latent_mode == "all_implicit":

                intermediate = mixture_latent_vectors  # num_scenes x mixture_size # implicit (patch latent, position, rotation, scaling)
                for i, (layer_type, _, activation, dropout) in enumerate(self.mixture_to_patch_parameters["layers"]):
                    if layer_type == "FC": 
                        layer = getattr(self, "object_to_patch_FC_" + str(i))
                        intermediate = layer(intermediate)
                        if dropout > 0.:
                            dropout = getattr(self, "object_to_patch_dropout_" + str(i))
                            intermediate = dropout(intermediate)
                        if activation == "relu":
                            intermediate = self.relu(intermediate)
                        elif activation == "elu":
                            intermediate = self.elu(intermediate)
                patch_latent_vectors = intermediate # num_scenes x patch_size

            # input["mixture_latent_vectors"]: num_scenes x mixture_latent_size
            repeated_patch_latent_vectors = patch_latent_vectors.view(-1, 1, self.patch_latent_size).repeat(1, input["num_samp_per_scene"], 1).view(-1, self.patch_latent_size) # num_scenes * samp_per_scene * num_patches x patch_latent_size
            # input["xyz"]: num_scenes x samp_per_scene x 3
            input["standard_input"] = torch.cat([repeated_patch_latent_vectors, input["xyz"].view(-1, 3)], 1)  # num_scenes * samp_per_scene * num_patches x (patch_latent_size + 3)




        patch_network_input = input["standard_input"] # samples x (patch_latent_size + 3) (might be fewer samples if input is optimized)
        skip_patch_net = patch_network_input.shape[0] == 0 or ("current_stage" in input and input["current_stage"] == "1" and self.num_patches > 1)
        if not skip_patch_net:
            x = self.patch_network_forward(patch_network_input, extra_loss=extra_loss) # samples x 1





        if self.num_patches > 1:

            if skip_patch_net:
                patch_sdfs = 0.
            else:
                # undo subampled/optimized network input & undo scaling
                if optimize_input_of_patch_network:
                    all_samples = torch.zeros((samples_in_patches_mask.shape[0], x.shape[1]), device=device) # samples x 1
                    all_samples[samples_in_patches_mask,:] = x
                    all_samples_original_shape = all_samples.shape
                    all_samples = all_samples.view(patch_scaling.shape[0], input["num_samp_per_scene"], self.num_patches)
                    all_samples *= patch_scaling.unsqueeze(1).squeeze(-1)
                    all_samples = all_samples.view(all_samples_original_shape)
                    all_samples[~samples_in_patches_mask,:] = 1.
                    x = all_samples
                else:
                    raise RuntimeError("check code again")
                    x_original_shape = x.shape
                    x = x.view(patch_scaling.shape[0], input["num_samp_per_scene"], self.num_patches)
                    x *= patch_scaling.unsqueeze(1).squeeze(-1)
                    x = x.view(x_original_shape)

                patch_sdfs = x.view(-1, self.num_patches)
            patch_weights = patch_weights.view(-1, self.num_patches)

            if self.loss_on_patches_instead_of_mixture or self.script_mode:
                if "sdf_gt" in input:

                    if mixture_latent_mode == "all_explicit":
                        direct_patch_weight = 1.
                    else:
                        direct_patch_weight = 10.

                    if self.use_curriculum_weighting:
                        curriculum_deepsdf_lambda = 0.5
                        curriculum_clamping_distance = 0.1
                        clamped_patch_sdfs = torch.clamp(patch_sdfs.detach(), -curriculum_clamping_distance, +curriculum_clamping_distance)
                        clamped_sdf_gt = torch.clamp(input["sdf_gt"].view(-1, 1).detach(), -curriculum_clamping_distance, +curriculum_clamping_distance)
                        first_weight = torch.sign(clamped_sdf_gt) # samples x 1
                        second_weight = torch.sign(clamped_sdf_gt - clamped_patch_sdfs) # samples x num_patches
                        curriculum_weights = 1. + curriculum_deepsdf_lambda * first_weight * second_weight # samples x num_patches

                    patch_mask = patch_weights == 0.
                    patch_recon = patch_sdfs - input["sdf_gt"].view(-1, 1) # broadcast across patches
                    patch_recon[patch_mask] = 0.
                    if self.use_curriculum_weighting:
                        patch_recon *= curriculum_weights
                    patch_recon = patch_recon.view(-1, input["num_samp_per_scene"], self.num_patches)
                    patch_recon = torch.abs(patch_recon)
                    patch_recon = patch_recon / (torch.sum((~patch_mask).view(-1, input["num_samp_per_scene"], self.num_patches), dim=1).unsqueeze(1).float() + 0.000001)
                    direct_patch_loss = torch.sum(patch_recon, dim=1) # num_scenes x num_patches
                    extra_loss["direct_patch"] = direct_patch_weight * torch.mean(direct_patch_loss)
                else:
                    direct_patch_loss = None # dummy value

            mixture_type = "convex"
            if mixture_type == "convex":
                weighted_sdfs = patch_weights * patch_sdfs
                weighted_sdfs = torch.sum(weighted_sdfs, 1)
                normalization = torch.sum(patch_weights, 1)

                # hacky
                default_sdf_value = 1.0
                mask = normalization == 0.
                weighted_sdfs[mask] = default_sdf_value
                weighted_sdfs[~mask] = weighted_sdfs[~mask] / normalization[~mask]
                weighted_sdfs = weighted_sdfs.unsqueeze(1)
                x = weighted_sdfs
            ##elif mixture_type == "closest":
            #    # only works if patch_weights are monotonically falling with distance
            #    patch_assignment = torch.argmax(patch_weights, dim=1)
            #    x = patch_sdfs[:, patch_assignment]
            
            if self.do_code_regularization:
                if mixture_latent_mode == "all_explicit" or mixture_latent_mode == "patch_explicit_meta_implicit":
                    extra_loss["latent_regularization_patch"] = torch.mean(patch_latent_vectors.pow(2))

                if mixture_latent_mode == "all_implicit":
                    extra_loss["latent_regularization_object"] = torch.mean(mixture_latent_vectors.pow(2))

                elif mixture_latent_mode == "patch_explicit_meta_implicit":
                    extra_loss["latent_regularization_posrot"] = torch.mean(posrot_latents.pow(2))
            
            if self.keep_scales_small:

                if fix_metadata_regression_by_decreasing_patch_scale:
                    if "current_stage" in input and input["current_stage"] == "1":
                        small_scales_weight = 20.
                    else:
                        if mixture_latent_mode == "all_implicit":
                            small_scales_weight = 0.2
                        elif mixture_latent_mode == "all_explicit":
                            small_scales_weight = 0.01
                        else:
                            raise RuntimeError()
                else:
                    if mixture_latent_mode == "all_implicit":
                        small_scales_weight = 0.2
                    elif mixture_latent_mode == "all_explicit":
                        small_scales_weight = 0.01
                    else:
                        raise RuntimeError()

                extra_loss["small_scales"] = small_scales_weight * torch.mean(patch_scaling**2)

            if self.scales_low_variance:

                variances = torch.var(patch_scaling.view(-1, self.num_patches), dim=-1, unbiased=False)

                if mixture_latent_mode == "all_implicit":
                    low_variance_weight = 50.
                elif mixture_latent_mode == "all_explicit":
                    low_variance_weight = 0.01
                else:
                    raise RuntimeError()

                extra_loss["low_variance_scales"] = low_variance_weight * torch.mean(variances)


            if self.pull_free_space_patches_to_surface or self.pull_patches_to_uncovered_surface:

                surface_sdf_threshold = 0.02

                if "sdf_gt" in input:
                    sdf_gt = input["sdf_gt"].squeeze(-1).flatten()
                    surface_mask = torch.abs(sdf_gt) <= surface_sdf_threshold

                    if "extra_losses_mask" in input and input["extra_losses_mask"] is not None: # used for hierarchical representation
                        surface_mask = surface_mask[input["extra_losses_mask"]]

            if self.pull_free_space_patches_to_surface:

                free_space_distance_threshold = 0.2 * self.non_variable_patch_radius
                free_space_loss_weight = 5.0

                if "sdf_gt" in input and "num_samp_per_scene" in input:

                    masked_distances = unscaled_center_distances.clone().view(-1, self.num_patches) #distances: samples * num_patches
                    masked_distances[~surface_mask,:] = 10000000.
                    masked_distances = masked_distances.view(-1, input["num_samp_per_scene"], self.num_patches) # num_scenes x samples_per_scene x num_patches
                        
                    closest_surface_distances, closest_surface_indices = torch.min(masked_distances, dim=1) # num_scenes x num_patches

                    free_space_patches = closest_surface_distances > free_space_distance_threshold

                    closest_surface_distances[~free_space_patches] = 0.
                    free_space_scene_normalization = torch.sum(free_space_patches, dim=1) # num_scenes
                    free_space_scenes = free_space_scene_normalization > 0
                    eps = 0.001
                    free_space_scene_losses = torch.sum(closest_surface_distances[free_space_scenes,:], dim=1) / (free_space_scene_normalization[free_space_scenes].float() + eps) # num_scenes
                    free_space_loss = torch.sum(free_space_scene_losses) / (torch.sum(free_space_scenes) + eps)

                    extra_loss["free_space"] = free_space_loss_weight * free_space_loss

            if self.align_patch_rotation_with_normal and "normals" in input:

                # surface normal should be aligned with local z-axis of patch coordinate system

                if not self.pull_free_space_patches_to_surface:
                    masked_distances = unscaled_center_distances.clone().view(-1, self.num_patches) #distances: samples * num_patches
                    masked_distances[~surface_mask,:] = 10000000.
                    masked_distances = masked_distances.view(-1, input["num_samp_per_scene"], self.num_patches) # num_scenes x samples_per_scene x num_patches
                        
                    closest_surface_distances, closest_surface_indices = torch.min(masked_distances, dim=1) # num_scenes x num_patches

                normals = input["normals"] # num_scenes x samp_per_scene x 3
                index_helper = np.repeat(np.arange(normals.shape[0]), repeats=self.num_patches) # e.g. for three patches and four scenes: [0,0,0,1,1,1,2,2,2,3,3,3]
                #closest_surface_indices # shape: num_scenes x num_patches. indexes samples_per_scene
                target_zaxis = normals[index_helper, closest_surface_indices.view(-1), :] # num_scenes * num_patches x 3
                target_zaxis = target_zaxis.view(-1, self.num_patches, 3) # num_scenes x num_patches x 3

                #rotations # num_scenes x num_patches x 3 x 3. local-to-global coordinates
                regressed_zaxis = rotations[:,:,:,2] # num_scenes x num_patches x 3

                dot_product = torch.sum(regressed_zaxis * target_zaxis, dim=-1) # num_scenes x num_patches
                rotation_loss = (1. - dot_product)**2

                rotation_loss_weight = 1.
                extra_loss["rotation_alignment"] = rotation_loss_weight * torch.mean(rotation_loss)

            if self.pull_patches_to_uncovered_surface:

                pull_weight_threshold = 0. # if 0. --> effectively normalization == 0
                pull_std = 0.05
                loss_weight = 200.0
                
                if "sdf_gt" in input and "num_samp_per_scene" in input:
                    sum_weights = normalization

                    weight_mask = sum_weights.detach() <= pull_weight_threshold

                    pull_mask = weight_mask * surface_mask # logical AND. shape: samples
                    pull_mask = pull_mask.unsqueeze(1).repeat(1, self.num_patches).view(-1) # shape: samples * num_patches

                    use_scaled_distances = False
                    if use_scaled_distances:
                        distances_to_use = scaled_center_distances # num_scenes x samp_per_scene x num_patches
                    else:
                        # unscaled_center_distances_nonflat: num_scenes x samp_per_scene x num_patches
                        distances_to_use = unscaled_center_distances_nonflat - patch_scaling.unsqueeze(1).squeeze(-1)# * self.patch_clamping_radius
                        distances_to_use = torch.clamp(distances_to_use, min=0.) # remove negative entries, i.e. samples inside patches
                        distances_to_use = distances_to_use.flatten()
                        
                    eps = 0.0001
                    normalize_weights = True
                    if normalize_weights:

                        pull_distances = torch.zeros_like(distances_to_use) # distances: samples * num_patches
                        pull_distances[pull_mask] = distances_to_use[pull_mask]
                        pull_weights = torch.exp(  -0.5 * (pull_distances/pull_std)**2   ) / pull_std # samples * num_patches
                        pull_weights[~pull_mask] = 0.

                        pull_weights = pull_weights.view(-1, self.num_patches)
                        pull_normalization = torch.sum(pull_weights, 1) # samples
                        norm_mask = pull_normalization == 0.
                        pull_weights[norm_mask,:] = 0.0
                        pull_weights[~norm_mask,:] = pull_weights[~norm_mask,:] / (pull_normalization[~norm_mask].unsqueeze(-1) + eps)
                        pull_weights = pull_weights.view(-1)                    
                    else:
                        pull_distances = distances_to_use[pull_mask] # distances: samples * num_patches
                        pull_weights = torch.exp(  -0.5 * (pull_distances/pull_std)**2   ) / pull_std

                    weighted_pulls = (pull_weights * pull_distances).view(-1, input["num_samp_per_scene"], self.num_patches)
                    weighted_pulls = torch.sum(weighted_pulls, dim=-1) # num_scenes x samples_per_scene
                    norm_mask = norm_mask.view(-1, input["num_samp_per_scene"])
                    norm_mask = torch.sum(norm_mask, dim=1) # num_scenes
                    norm_scenes_mask = norm_mask > 0
                    coverage_scene_losses = torch.sum(weighted_pulls[norm_scenes_mask,:], dim=1) / (norm_mask[norm_scenes_mask].float() + eps) # num_scenes
                    coverage_loss = torch.sum(coverage_scene_losses) / (torch.sum(norm_scenes_mask) + eps)

                    extra_loss["uncovered"] = loss_weight * coverage_loss

        else: # single patch

            if self.do_code_regularization:
                extra_loss["latent_regularization_object"] = torch.mean(input["mixture_latent_vectors"].pow(2))

        if "first_stage_pos" in input and input["first_stage_pos"] is not None:

            target_positions = input["first_stage_pos"] # num_scenes x num_patches x 3
            target_rotations = input["first_stage_rot"] # num_scenes x num_patches x 3
            target_scales = input["first_stage_scale"] # num_scenes x num_patches x 1

            target_rotations = self._convert_euler_to_matrix(target_rotations.view(-1, 3)).view(-1, self.num_patches, 3, 3)
            target_zaxis = target_rotations[:,:,:,2]
            regressed_zaxis = rotations[:,:,:,2]
            dot_product = torch.sum(regressed_zaxis * target_zaxis, dim=-1) # num_scenes x num_patches

            rotation_loss_weight = 0.
            rotation_loss = (1. - dot_product)**2

            scale_weight = 30.
            scale_loss = (target_scales - patch_scaling)**2

            position_weight = 3.
            position_loss = (target_positions - patch_position)**2

            extra_loss["metadata_rotation"] = rotation_loss_weight * torch.mean(rotation_loss)
            extra_loss["metadata_position"] = position_weight * torch.mean(position_loss)
            extra_loss["metadata_scale"] = scale_weight * torch.mean(scale_loss)
           
            
        if self.script_mode and self.num_patches > 1:
            self.patch_network_input = patch_network_input
            self.patch_latent_vectors = patch_latent_vectors
            self.patch_positions = patch_position
            self.patch_rotations = patch_rotation
            self.patch_scalings = patch_scaling
            self.patch_network_sdfs = patch_sdfs
            self.patch_network_mixture_weights = patch_weights
            self.patch_network_mixture_normalization = normalization
            self.direct_patch_loss = direct_patch_loss

        if self.num_patches > 1:
            extra_outputs = {}
            extra_outputs["patch_positions"] = patch_position
            extra_outputs["patch_rotations"] = patch_rotation
            extra_outputs["patch_scalings"] = patch_scaling

        # always return SDF/x as the first result, else adapt deep_sdf.utils.decode_sdf and potentially other code!
        if input["use_patch_encoder"]:
            regressed_mixture_vectors = torch.cat([patch_latent_vectors, patch_position, patch_rotation, patch_scaling], dim=2)
            regressed_mixture_vectors = regressed_mixture_vectors.view(-1, 1, self.num_patches * (self.patch_latent_size + 3 + 3 + 1))
            return x, regressed_mixture_vectors, extra_loss
        elif self.use_depth_encoder:
            return x, extra_loss, extra_outputs, mixture_latent_vectors
        else:
            if "return_extra_outputs" in input and input["return_extra_outputs"]:
                return x, extra_loss, extra_outputs
            else:
                return x, extra_loss
