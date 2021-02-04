#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import deep_sdf
import deep_sdf.workspace as ws
from train_deep_sdf import initialize_mixture_latent_vector


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    latent=None,
    specs=None,
    full_loss=False,
    sdf_filename=None,
    adjust_lr_every=None,
    decrease_lr_by=None,
    mixture_latent_mode=None
):
    if latent is None:
        if specs is None:
            if type(stat) == type(0.1):
                latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
            else:
                latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        else:
            latent = initialize_mixture_latent_vector(specs, sdf_filename=sdf_filename).cuda()

    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decrease_lr_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decrease_lr_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    if decrease_lr_by is None:
        decrease_lr_by = 2
    if adjust_lr_every is None:
        adjust_lr_every = 100 #max(int(num_iterations / 3), 1)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    old_latent_l2 = False

    if not old_latent_l2:
        if l2reg:
            decoder.do_code_regularization = True

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        xyz = xyz.reshape(1, num_samples, 3)
        sdf_gt = sdf_gt.reshape(1, num_samples, 1)
            
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decrease_lr_by, adjust_lr_every)

        optimizer.zero_grad()

        inputs = {}
        inputs["mixture_latent_vectors"] = latent.view(1, -1) # 1 x mixture_latent_size
        inputs["xyz"] = xyz
        inputs["use_patch_encoder"] = False
        inputs["num_samp_per_scene"] = num_samples
        inputs["sdf_gt"] = sdf_gt
        inputs["mixture_latent_mode"] = specs["NetworkSpecs"]["mixture_latent_mode"] if mixture_latent_mode is None else mixture_latent_mode

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        #if e == 0:
        #    pred_sdf = decoder(inputs)

        if type(pred_sdf) == tuple:
            pred_sdf, extra_losses = pred_sdf

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = torch.zeros(1).cuda()[0]
        if full_loss:
            for extra_loss_name, extra_loss_value in extra_losses.items():
                if old_latent_l2:
                    loss += extra_loss_value
                else:
                    if "latent_regularization" in extra_loss_name:
                        loss += 1e-4 * extra_loss_value
                    else:
                        loss += extra_loss_value

            
        main_loss_weight = 1.0 if specs is None or not full_loss else specs["MainLossWeight"]
        loss += main_loss_weight * loss_l1(pred_sdf, sdf_gt)
        if old_latent_l2:
            if l2reg:
                loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def main_function_reconstruction(args):
    
    deep_sdf.configure_logging(args.loglevel, args.logfile)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.results_folder, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    with open(specs_filename) as specs:
        specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
        specs = json.loads(specs)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    patch_latent_size = specs["PatchCodeLength"]
    mixture_latent_size = specs["MixtureCodeLength"]

    decoder = arch.Decoder(patch_latent_size=patch_latent_size, mixture_latent_size=mixture_latent_size, encoder=grid_encoder_param, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.results_folder, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
    npz_filenames = [(npz, os.path.join(args.data_source, ws.sdf_samples_subdir, npz)) for npz in npz_filenames]

    #random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.output_folder, ws.reconstructions_subdir, "videos/", str(saved_model_epoch) + "_" + ("opt" if args.optimize else "noopt")
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )

    print(reconstruction_meshes_dir, flush=True)

    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    with open(reconstruction_meshes_dir + "/MeshFiles", "w") as mesh_files:
        for npz, _ in npz_filenames:
            mesh_files.write(npz[:-4] + ".obj\n")

    for ii, (npz, full_filename) in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".obj")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()

            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                mixture_latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True
            )

            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    max_batch = int(2 ** 18)
                    if hasattr(decoder, "num_patches"):
                        max_batch /= decoder.num_patches + 2
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(max_batch)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="results_folder",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        dest="output_folder",
        default=None,
        help="The output directory where reconstructions will be stored.",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        help="If using the encoder, whether to optimize the latent vector or directly use the encoder output.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.results_folder

    main_function_reconstruction(args)