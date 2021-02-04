#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "output"
reconstruction_meshes_subdir = "meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"


def load_experiment_specifications(results_folder):

    filename = os.path.join(results_folder, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(results_folder)
        )

    return json.load(open(filename))


def load_model_parameters(results_folder, checkpoint, decoder):

    filename = os.path.join(
        results_folder, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def build_decoder(results_folder, experiment_specs):

    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(
    results_folder, experiment_specs, checkpoint, data_parallel=True
):

    decoder = build_decoder(results_folder, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(results_folder, checkpoint, decoder)

    return (decoder, epoch)


def load_latent_vectors(results_folder, checkpoint):

    filename = os.path.join(
        results_folder, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(results_folder, checkpoint)
        )

    data = torch.load(filename)

    num_vecs = data["latent_codes"].size()[0]

    lat_vecs = []
    for i in range(num_vecs):
        lat_vecs.append(data["latent_codes"][i].cuda())

    return lat_vecs


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )

def read_patch_encoder_param(input_string):

    layers = []
    for layer in input_string.split(","):
        layer = layer.strip().split(" ")
        layer_type = layer[0]
        if layer_type == "max":
            activation = layer[1]
            layers.append(("max", None, activation))
        if layer_type == "FC":
            _, out_channels, weight_norm, activation = layer
            layers.append(("FC", (int(out_channels), bool(weight_norm)), activation))

    return {"layers": layers}