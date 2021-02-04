#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import warnings
warnings.filterwarnings("ignore", module=".*tensorflow.*")

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import numpy as np
import json
import time
import shutil
import socket
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import scipy.spatial

import deep_sdf
import deep_sdf.workspace as ws
import reconstruct

from localization.SystemSpecific import get_settings_dictionary, system_specific_cleanup, system_specific_session_config


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))

class StagedLearningRateSchedule(LearningRateSchedule):
    def __init__(self, lengths, stages):
        self.lengths = lengths
        self.stages = get_learning_rate_schedules(stages)

    def get_learning_rate(self, epoch):

        epoch_offset = 0
        current_stage_id = None
        for stage_id, length in enumerate(self.lengths):
            epoch_offset += length
            if epoch <= epoch_offset:
                current_stage_id = stage_id
                epoch_offset -= length
                break
        if current_stage_id is None:
            current_stage_id = len(self.lengths) - 1

        return self.stages[current_stage_id].get_learning_rate(epoch - epoch_offset)
        


def get_learning_rate_schedules(schedule_specs):

    schedules = []

    for schedule_specs in schedule_specs:


        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )

        elif schedule_specs["Type"] == "Staged":
            schedules.append(
                StagedLearningRateSchedule(
                    schedule_specs["Lengths"],
                    schedule_specs["Stages"]
                )
            )

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(results_folder, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(results_folder, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )

    #if filename[-4:] == ".pth":
    #    filename = filename[:-4]
    #torch.save(decoder, os.path.join(model_params_dir, filename + "_full_model.pth"))


def save_optimizer(results_folder, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(results_folder, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(results_folder, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(results_folder), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(results_folder, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(results_folder, True)

    all_latents = torch.zeros(0)
    for l in latent_vec:
        all_latents = torch.cat([all_latents, l.cpu().unsqueeze(0)], 0)

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(results_folder, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(results_folder), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if not len(lat_vecs) == data["latent_codes"].size()[0]:
        raise Exception(
            "num latent codes mismatched: {} vs {}".format(
                len(lat_vecs), data["latent_codes"].size()[2]
            )
        )

    if not lat_vecs[0].size()[1] == data["latent_codes"].size()[2]:
        raise Exception("latent code dimensionality mismatch")

    for i in range(len(lat_vecs)):
        lat_vecs[i] = data["latent_codes"][i].cuda()

    return data["epoch"]


def save_logs(
    results_folder,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(results_folder, ws.logs_filename),
    )


def load_logs(results_folder):

    full_filename = os.path.join(results_folder, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    host_vectors = np.array(
        [vec.detach().cpu().numpy().squeeze() for vec in latent_vectors]
    )
    return np.mean(np.linalg.norm(host_vectors, axis=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def _random_initialize_mixture_latent_vector(specs):

    mixture_latent_size = specs["MixtureCodeLength"]
    patch_latent_size = specs["PatchCodeLength"]

    if specs["NetworkSpecs"]["mixture_latent_mode"] == "all_explicit" or specs["NetworkSpecs"]["mixture_latent_mode"] == "patch_explicit_meta_implicit":
        # num_patches * (patch_latent_size + 3 + 3)
        vec = np.empty((0), dtype=np.float32)

        for patch in range(specs["NetworkSpecs"]["num_patches"]):
            patch_latent = np.random.normal(0., get_spec_with_default(specs, "CodeInitStdDev", 1.0), patch_latent_size)
            if specs["NetworkSpecs"]["mixture_latent_mode"] == "all_explicit":
                patch_pos = np.random.uniform(-0.5, +0.5, 3)
                patch_rot = np.random.uniform(0, 2*np.pi, 3)
                patch_meta = np.concatenate([patch_pos, patch_rot])
                if specs["NetworkSpecs"]["patch_scaling"]:
                    patch_scale = np.array([1.])
                    patch_meta = np.concatenate([patch_meta, patch_scale])
            else:
                patch_meta = np.random.normal(0., get_spec_with_default(specs, "CodeInitStdDev", 1.0), specs["NetworkSpecs"]["meta_latent_size"])
            vec = np.concatenate([vec, patch_latent, patch_meta])

        vec = vec.reshape(1, -1)
        if vec.shape[1] != mixture_latent_size:
            raise RuntimeError("incompatible latent sizes when initializing latent vectors")
        vec = torch.from_numpy(vec.astype(np.float32))

    else:
        vec = (
            torch.ones(1, mixture_latent_size)
            .normal_(0, get_spec_with_default(specs, "CodeInitStdDev", 1.0))
        )
    return vec

def farthest_point_sampling(points, K):
    # greedily sample K farthest points from "points" (N x 3)
    num_points = points.shape[0]
    if num_points < K:
        print("too few points for farthest point sampling. will returned repeated indices.")
        indices = np.tile(np.arange(num_points), int(K // num_points) + 1)
        indices = indices[:K]
        return points[indices,:], None

    # compute all pairwise distances
    import scipy.spatial
    pairwise_distances = scipy.spatial.distance.cdist(points, points, metric="euclidean") # points x points
    farthest_points_mask = np.zeros(num_points).astype(bool)
    farthest_points_mask[0] = True
    index_helper = np.arange(num_points)
    for k in range(K-1):
        relevant_distances = pairwise_distances[np.ix_(index_helper[~farthest_points_mask], index_helper[farthest_points_mask])] # distances from not-yet-sampled points to farthest points
        relevant_minimums = np.min(relevant_distances, axis=1)
        new_farthest_point_index = np.argmax(relevant_minimums) # new_farthest_point_index indexes "1" entries in ~farthest_points_mask, not num_points
        new_farthest_point_index = index_helper[~farthest_points_mask][new_farthest_point_index]
        farthest_points_mask[new_farthest_point_index] = True

    return points[farthest_points_mask,:], farthest_points_mask # numpy array K x 3, Boolean mask of size "num_points" 

def _normalized_vector(a):
    return a / np.linalg.norm(a)

def _angle_between_vectors(a, b):
    an = _normalized_vector(a)
    bn = _normalized_vector(b)
    return np.arccos(np.clip(np.dot(an, bn), -1.0, 1.0))

def _get_euler_angles_from_rotation_matrix(rotation_matrix):
    if np.abs(rotation_matrix[2,0]) != 1.:
        beta = -np.arcsin(rotation_matrix[2,0])
        cosBeta = np.cos(beta)
        return np.array([np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]), beta, np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])])

		# Solution not unique, this is the second one
		#const float beta = PI+asin(R.m31); const float cosBeta = cos(beta);
		#return make_float3(atan2(R.m32/cosBeta, R.m33/cosBeta), beta, atan2(R.m21/cosBeta, R.m11/cosBeta));
    else:
        if rotation_matrix[2,0] == -1.0:
            return np.array([np.arctan2(rotation_matrix[0,1], rotation_matrix[0,2]), np.pi/2., 0.])
        else:
            return np.array([np.arctan2(-rotation_matrix[0,1], -rotation_matrix[0,2]), -np.pi/2., 0.])

def _get_rotation_from_normal(normal):
    # normal: numpy array of size (3,)

    # left-handed coordinate system with x-y plane and z-axis as height (x=thumb, y=middle, z=index). want to rotate the normal such that it aligns with the z-axis. apply that rotation to the patch as a whole
    # first, rotate coordinate system around y-axis such that the normal lies in the y-z plane.
    projected_z_axis = np.array([0., 1.]) # in x-z plane
    projected_normal = np.array([normal[0], normal[2]])
    if np.linalg.norm(projected_normal) < 0.000001:
        y_angle = 0.
    else:
        y_angle = _angle_between_vectors(projected_z_axis, projected_normal)
    if normal[0] > 0.:
        y_angle *= -1.
    y_rotation = scipy.spatial.transform.Rotation.from_euler("y", y_angle)
    rotated_normal = y_rotation.apply(normal)
    # then, rotate around x-axis to align the normal with the z-axis
    z_axis = np.array([0., 0., 1.])
    x_angle = _angle_between_vectors(z_axis, rotated_normal)
    if rotated_normal[1] <= 0.:
        x_angle *= -1.
    x_rotation = scipy.spatial.transform.Rotation.from_euler("x", x_angle)

    # converts global coordinates into local coordinates. the normal is in global coordinates. after multiplication with the rotation matrix, we get the local z-axis [0,0,1].
    rotation = x_rotation * y_rotation
    rotation = rotation.as_matrix() # 3x3 numpy array
    # we use local-to-global rotations in the model. so we need to take the inverse.
    rotation = rotation.transpose()

    euler_angles = _get_euler_angles_from_rotation_matrix(rotation)

    if np.any(np.isnan(euler_angles)):
        euler_angles = np.zeros(3)

    return euler_angles # numpy array of size (3,) 

def initial_metadata_from_sdf_samples(sdf_samples, normals, num_patches, surface_sdf_threshold, num_samples_for_computation=10000, final_scaling_increase_factor=1.3):
    # sdf_samples: numpy array, num_points x 4
    # normals: numpy array, num_points x 3

    # surface samples
    surface_samples_mask = np.abs(sdf_samples[:,3]) < surface_sdf_threshold
    surface_samples = sdf_samples[surface_samples_mask,:]
    normals = normals[surface_samples_mask,:]

    num_surface_samples = surface_samples.shape[0]
    if num_surface_samples < num_patches:
        raise RuntimeError("not enough surface SDF samples found")

    # DeepSDF preprocessing generates ~500k points. Considering all of them is very expensive. Instead, only consider at most num_samples_for_computation many of them.
    if num_surface_samples > num_samples_for_computation:
        indices = np.linspace(0, num_surface_samples, num=num_samples_for_computation, endpoint=False, dtype=int)
        surface_samples = surface_samples[indices,:]
        normals = normals[indices,:]
        num_surface_samples = num_samples_for_computation

    # patch centers
    patch_centers, patch_center_indices = farthest_point_sampling(surface_samples[:,:3], K=num_patches) # patch_centers: num_patches x 3

    # patch rotations
    patch_rotations = np.array([_get_rotation_from_normal(normal) for normal in normals[patch_center_indices,:]]) # num_patches x 3

    # patch scale
    index_helper = np.arange(num_surface_samples)
    distances_to_patches = scipy.spatial.distance.cdist(surface_samples[:,:3], patch_centers, metric="euclidean") # num_surface_samples x num_patches
    closest_patches = np.argmin(distances_to_patches, axis=1) # shape: num_surface_samples
    filtered_distances = np.zeros((num_surface_samples, num_patches), dtype=np.float32)
    filtered_distances[index_helper,closest_patches] = distances_to_patches[index_helper,closest_patches]
    patch_scales = np.max(filtered_distances, axis=0) # shape: num_patches
    patch_scales *= final_scaling_increase_factor

    return patch_centers, patch_rotations, patch_scales # num_patches x 3, num_patches x 3, num_patches 



def initialize_mixture_latent_vector(specs, sdf_samples_with_normals=None, sdf_filename=None, overwrite_init_file=False, use_precomputed_init=None, initial_patch_latent=None):

    mixture_latent_mode = specs["NetworkSpecs"]["mixture_latent_mode"]
    if mixture_latent_mode == "all_implicit":
        return torch.zeros((1, specs["MixtureCodeLength"]))

    patch_latent_size = specs["PatchCodeLength"]

    if mixture_latent_mode == "patch_explicit_meta_implicit":
        return torch.zeros((1, specs["NetworkSpecs"]["posrot_latent_size"] + 30 * patch_latent_size))

    if sdf_filename is None and sdf_samples_with_normals is None:
        raise RuntimeError("using completely random initialization. are you sure?")
        return _random_initialize_mixture_latent_vector(specs)
    
    use_tiny_patchnet = specs["NetworkSpecs"]["use_tiny_patchnet"]
    num_patches = specs["NetworkSpecs"]["num_patches"]
    variable_patch_scaling = specs["NetworkSpecs"]["variable_patch_scaling"]
    surface_sdf_threshold = 0.02
    final_scaling_increase_factor = 1.2
    num_samples_for_computation = 30000
    if sdf_filename is not None:
        initialization_file = sdf_filename + "_init_" + str(patch_latent_size) + "_" + str(num_patches) + "_" + str(surface_sdf_threshold) + "_" + str(final_scaling_increase_factor) + ("_tiny" if use_tiny_patchnet else "") + ".npy"

    if sdf_samples_with_normals is None:
        if use_precomputed_init:
            return torch.from_numpy(np.load(initialization_file))
        else:
            npz = np.load(sdf_filename)
            pos_tensor = deep_sdf.data.remove_nans(torch.from_numpy(np.concatenate([npz["pos"], npz["pos_normals"]], axis=1))).numpy()
            neg_tensor = deep_sdf.data.remove_nans(torch.from_numpy(np.concatenate([npz["neg"], npz["neg_normals"]], axis=1))).numpy()
            sdf_samples_with_normals = np.concatenate([pos_tensor, neg_tensor], axis=0)

    patch_centers, patch_rotations, patch_scales = initial_metadata_from_sdf_samples(sdf_samples_with_normals[:,:4], sdf_samples_with_normals[:,4:], num_patches=num_patches, surface_sdf_threshold=surface_sdf_threshold, num_samples_for_computation=num_samples_for_computation, final_scaling_increase_factor=final_scaling_increase_factor)

    patch_latents = [ np.zeros(patch_latent_size) if initial_patch_latent is None else initial_patch_latent for _ in range(num_patches)]

    per_patch_merged = [ np.concatenate([ latent, center, rotation, np.array([scale]) if variable_patch_scaling else np.empty(0) ]) for latent, center, rotation, scale in zip(patch_latents, patch_centers, patch_rotations, patch_scales) ]
    latent = np.concatenate(per_patch_merged).reshape(1, -1).astype(np.float32)

    if sdf_filename is not None and (not os.path.exists(initialization_file) or overwrite_init_file):
        np.save(initialization_file, latent)
    return torch.from_numpy(latent)


def main_function(specs, settings, results_folder, code_folder, train_split_file, test_split_file=None):

    #torch.autograd.set_detect_anomaly(True) # finds weird gradient behavior for debugging

    logging.debug("running " + results_folder)

    logging.info("Experiment description: \n" + "".join(specs["Description"]))

    if specs["seed"] >= 0:
        np.random.seed(specs["seed"])
        torch.manual_seed(specs["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        #torch.backends.cudnn.benchmark = True # tries to pick good kernel implementations
        pass

    data_source = specs["DataSource"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    mixture_latent_size = specs["MixtureCodeLength"]
    patch_latent_size = specs["PatchCodeLength"]
    batch_split = specs["BatchSplit"]
    staged_training = specs["StagedTraining"]

    patch_encoder = "Encoder" in specs and specs["Encoder"] == "True"
    patch_encoder_param = ws.read_patch_encoder_param(specs["PatchEncoderLayers"]) if "PatchEncoderLayers" in specs and patch_encoder else None

    use_tiny_patchnet = specs["NetworkSpecs"]["use_tiny_patchnet"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs["LearningRateSchedule"])
    if staged_training:
        lr_schedules = lr_schedules[2:]
    else:
        lr_schedules = lr_schedules[:2]

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(results_folder, "latest.pth", decoder, epoch)
        save_optimizer(results_folder, "latest.pth", optimizer_train, epoch)
        save_latent_vectors(results_folder, "latest.pth", train_lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(results_folder, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(results_folder, str(epoch) + ".pth", optimizer_train, epoch)
        save_latent_vectors(results_folder, str(epoch) + ".pth", train_lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def latent_size_regul(latent, indices):
        latent_loss = 0.0
        for ind in indices:
            latent_loss += torch.mean(latent[ind].pow(2))
        return latent_loss / len(indices)

    #def latent_size_regul_with_encoder(latent):
    #    return torch.mean(latent.pow(2))

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    if not (scene_per_batch % batch_split) == 0:
        raise RuntimeError("Unequal batch splitting is not supported.")

    scene_per_subbatch = scene_per_batch // batch_split

    min_vec = torch.from_numpy(np.array([minT])).float().cuda()
    max_vec = torch.from_numpy(np.array([maxT])).float().cuda()

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    if patch_encoder:
        code_bound = None
    else:
        code_bound = get_spec_with_default(specs, "CodeBound", None)

    pull_patches_to_uncovered_surface = specs["NetworkSpecs"]["pull_patches_to_uncovered_surface"]
    pull_free_space_patches_to_surface = specs["NetworkSpecs"]["pull_free_space_patches_to_surface"]
    loss_on_patches_instead_of_mixture = specs["NetworkSpecs"]["loss_on_patches_instead_of_mixture"]
    use_normals = specs["NetworkSpecs"]["align_patch_rotation_with_normal"]
    use_depth = specs["NetworkSpecs"]["use_depth_encoder"]

    decoder = arch.Decoder(patch_latent_size=patch_latent_size, mixture_latent_size=mixture_latent_size, patch_encoder=patch_encoder_param, do_code_regularization=do_code_regularization, results_folder=results_folder, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    def _get_data_loader(list_file, shuffle):
        with open(list_file, "r") as f:
            split = json.load(f)
        sdf_dataset = deep_sdf.data.SDFSamples(
            data_source, split, num_samp_per_scene, load_ram=True, use_normals=use_normals, use_depth=use_depth
        )
        num_scenes = len(sdf_dataset)
        sdf_loader = data_utils.DataLoader(
            sdf_dataset,
            batch_size=scene_per_subbatch,
            shuffle=shuffle,
            num_workers=num_data_loader_threads,
            drop_last=False,
            pin_memory=True
        )
        return sdf_dataset, sdf_loader, num_scenes

    train_sdf_samples, train_sdf_loader, num_train_scenes = _get_data_loader(train_split_file, shuffle=True)
    if test_split_file is not None:
        test_sdf_samples, test_sdf_loader, num_test_scenes = _get_data_loader(test_split_file, shuffle=False)

    # backup npyfilenames used for training
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
        npyfilenames = deep_sdf.data.get_instance_filenames(data_source, train_split)
        with open(results_folder + "backup/special_files/npyfilenames.txt", "w") as fn:
            for filename in npyfilenames:
                fn.write(filename + "\n")

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    logging.debug(decoder)

    def _init_latent_vectors(sdf_samples_object):
        
        logging.info("There are {} scenes".format(len(sdf_samples_object.npyfiles)))

        lat_vecs = []

        if use_tiny_patchnet:
            initial_patch_latent = decoder.module._initialize_tiny_patchnet()
        else:
            initial_patch_latent = None

        for i, npyfile in enumerate(sdf_samples_object.npyfiles):

            if i % 1000 == 0:
                print("initializing: " + str(i) + " / " + str(len(sdf_samples_object.npyfiles)), flush=True)
            
            if specs["DeepSDFMode"]:
                vec = torch.ones(1, specs["MixtureCodeLength"]).normal_(0, 0.01).cuda()
            elif specs["BaselineMode"]:
                vec = torch.zeros(1, specs["MixtureCodeLength"]).cuda()
            else:
                filename = os.path.join(sdf_samples_object.data_source, ws.sdf_samples_subdir, npyfile)
                vec = initialize_mixture_latent_vector(specs, sdf_filename=filename, overwrite_init_file=specs["overwrite_init_files"], use_precomputed_init=specs["use_precomputed_init"], initial_patch_latent=initial_patch_latent).cuda()
            
            vec.requires_grad = True
            lat_vecs.append(vec)

        logging.debug(
            "initialized with mean magnitude {}".format(
                get_mean_latent_vector_magnitude(lat_vecs)
            )
        )

        return lat_vecs

    train_lat_vecs = _init_latent_vectors(train_sdf_samples)
    if test_split_file is not None:
        test_lat_vecs = _init_latent_vectors(test_sdf_samples)

    loss_l1 = torch.nn.L1Loss()

    optimizer_parameters = [
            {
                "params": filter(lambda p: p.requires_grad, decoder.parameters()), # only train weights that have "requires_grad==True"
                "lr": lr_schedules[0].get_learning_rate(0),
            },
        ]

    if not use_depth:
        optimizer_parameters.append({"params": train_lat_vecs, "lr": lr_schedules[1].get_learning_rate(0)})
        if test_split_file is not None:
            optimizer_test = torch.optim.Adam([{"params": test_lat_vecs, "lr": lr_schedules[1].get_learning_rate(0)}])
    optimizer_train = torch.optim.Adam(optimizer_parameters)
    # Note: Keep Adam! autodecoding can lead to a bug (https://github.com/facebookresearch/DeepSDF/issues/51) and Adam allows for a workaround. If a different optimizer is to be used, make sure that it allows for a similar workaround.

    host = socket.gethostname()
    pid = str(os.getpid())

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    losses_log = []
    param_mag_log = {}

    start_epoch = 1

    if specs["Tensorboard"]:
        tensorboard_folder = results_folder + "logging/tensorboard/"
        tensorboard = SummaryWriter(log_dir=tensorboard_folder, max_queue=10, flush_secs=5)

    if specs["ContinueFrom"] != "":

        if test_split_file is not None:
            raise NotImplementedError("take care of test optimizer when continuing")

        continue_from = specs["ContinueFrom"]

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            results_folder, continue_from + ".pth", train_lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            results_folder, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            results_folder, continue_from + ".pth", optimizer_train
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            results_folder
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    epochs = list(range(start_epoch, num_epochs + 1))
    loop_epochs = []
    current_stage = 1 if staged_training else None
    for epoch in epochs:
        loop_epochs.append((train_sdf_loader, train_lat_vecs, optimizer_train, epoch, True, str(current_stage)))
        if staged_training:
            stage_transitions = np.cumsum(specs["LearningRateSchedule"][3]["Lengths"])
            if epoch == stage_transitions[0]:
                loop_epochs.append((train_sdf_loader, train_lat_vecs, optimizer_train, epoch, False, "1-2"))
                current_stage += 1
            if epoch == stage_transitions[1]: # allows to skip second stage
                loop_epochs.append((train_sdf_loader, train_lat_vecs, optimizer_train, epoch, False, "2-3"))
                current_stage += 1
    if test_split_file is not None:
        for test_epoch in epochs:
            loop_epochs.append((test_sdf_loader, test_lat_vecs, optimizer_test, test_epoch, False))

    first_stage_metadata = [None for _ in range(len(train_lat_vecs))]
    current_batch = 0
    for sdf_loader, lat_vecs, optimizer, epoch, is_training, current_stage in loop_epochs:

        print(host + " " + pid + " " + results_folder)

        start = time.time()

        logging.info(("training" if is_training else "test") + " epoch {}...".format(epoch))

        if is_training and not specs["test_time"]:
            decoder.train()  
        else:
            decoder.eval()
        adjust_learning_rate(lr_schedules, optimizer, epoch)

        if current_stage == "1": # only metadata, no recon
            main_loss_weight = 0.
            decoder.module.loss_on_patches_instead_of_mixture = False
            decoder.module.pull_free_space_patches_to_surface = specs["NetworkSpecs"]["pull_free_space_patches_to_surface"]
            decoder.module.pull_patches_to_uncovered_surface = specs["NetworkSpecs"]["pull_patches_to_uncovered_surface"]
            decoder.module.align_patch_rotation_with_normal = specs["NetworkSpecs"]["align_patch_rotation_with_normal"]
            decoder.module.keep_scales_small = specs["NetworkSpecs"]["keep_scales_small"]
            decoder.module.scales_low_variance = specs["NetworkSpecs"]["scales_low_variance"]
            return_extra_outputs = False

        elif current_stage == "1-2":
            return_extra_outputs = specs["NetworkSpecs"]["mixture_latent_mode"] == "all_implicit"
            main_loss_weight = 0.
            decoder.module.loss_on_patches_instead_of_mixture = False
            decoder.module.pull_free_space_patches_to_surface = False
            decoder.module.pull_patches_to_uncovered_surface = False
            decoder.module.align_patch_rotation_with_normal = False
            decoder.module.keep_scales_small = False
            decoder.module.scales_low_variance = False
            reset_optimizer_between_stages = True
            if reset_optimizer_between_stages:
                optimizer.state = defaultdict(dict)

        elif current_stage == "2": # no metadata, only recon
            main_loss_weight = specs["MainLossWeight"]
            decoder.module.loss_on_patches_instead_of_mixture = specs["NetworkSpecs"]["loss_on_patches_instead_of_mixture"]
            decoder.module.pull_free_space_patches_to_surface = False
            decoder.module.pull_patches_to_uncovered_surface = False
            decoder.module.align_patch_rotation_with_normal = False
            decoder.module.keep_scales_small = False
            decoder.module.scales_low_variance = False
            return_extra_outputs = False
            if specs["generate_initialization_for_object_latent"]:
                decoder.module.script_mode = True

        elif current_stage == "2-3":
            if reset_optimizer_between_stages:
                optimizer.state = defaultdict(dict)

        elif current_stage == "3": # metadata and recon
            main_loss_weight = specs["MainLossWeight"]
            decoder.module.loss_on_patches_instead_of_mixture = specs["NetworkSpecs"]["loss_on_patches_instead_of_mixture"]
            decoder.module.pull_free_space_patches_to_surface = specs["NetworkSpecs"]["pull_free_space_patches_to_surface"]
            decoder.module.pull_patches_to_uncovered_surface = specs["NetworkSpecs"]["pull_patches_to_uncovered_surface"]
            decoder.module.align_patch_rotation_with_normal = specs["NetworkSpecs"]["align_patch_rotation_with_normal"]
            decoder.module.keep_scales_small = specs["NetworkSpecs"]["keep_scales_small"]
            decoder.module.scales_low_variance = specs["NetworkSpecs"]["scales_low_variance"]
            #decoder.module.script_mode = False
            return_extra_outputs = False
            if specs["generate_initialization_for_object_latent"]:
                decoder.module.script_mode = True

        else:
            return_extra_outputs = False
            main_loss_weight = specs["MainLossWeight"]
    
        epoch_loss = {}
        def _add(name, value):
            if name not in epoch_loss:
                epoch_loss[name] = (0.0, 0)
            old_value, old_counter = epoch_loss[name] 
            epoch_loss[name] = (old_value + value.item(), old_counter+1)

        _subbatch = 0
        for batch in sdf_loader:

            sdf_data, indices, depth = batch

            if _subbatch == 0:
                batch_loss = 0.0
                batch_sublosses = defaultdict(lambda: 0.)
                batch_indices_to_optimize = []

                optimizer.zero_grad()

            # Process the input datag
            sdf_data.requires_grad = False
            if use_depth:
                depth.requires_grad = False

            if use_normals:
                sdf_data = (sdf_data.cuda()).view(-1, num_samp_per_scene, 7)
                normals = sdf_data[:, :, 4:7].contiguous() # num_scenes x samp_per_scene x 3
            else:
                sdf_data = (sdf_data.cuda()).view(-1, num_samp_per_scene, 4)
            xyz = sdf_data[:, :, 0:3].contiguous()
            sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).contiguous() # num_scenes x samp_per_scene x 1

            latent_inputs = [torch.zeros(0).cuda()]
            for ind in indices.numpy():
                batch_indices_to_optimize.append(ind)
                latent_ind = lat_vecs[ind]
                latent_repeat = latent_ind # 1 x mixture_latent_size
                latent_inputs.append(latent_repeat)
            latent_inputs = torch.cat(latent_inputs, 0)

            if enforce_minmax:
                sdf_gt = deep_sdf.utils.threshold_min_max(sdf_gt.view(-1, 1), min_vec, max_vec).view(-1, num_samp_per_scene, 1)

            if mixture_latent_size == 0:
                inputs = xyz

            inputs = {}
            inputs["mixture_latent_vectors"] = latent_inputs # num_scenes x mixture_latent_size
            inputs["xyz"] = xyz # num_scenes x samp_per_scene x 3
            if use_normals:
                inputs["normals"] = normals
            inputs["use_patch_encoder"] = patch_encoder
            inputs["num_samp_per_scene"] = num_samp_per_scene
            inputs["mixture_latent_mode"] = specs["NetworkSpecs"]["mixture_latent_mode"]
            inputs["current_stage"] = current_stage
            inputs["return_extra_outputs"] = return_extra_outputs
            if use_depth:
                inputs["depth_maps"] = depth

            if current_stage == "2" and specs["NetworkSpecs"]["mixture_latent_mode"] == "all_implicit" and specs["NetworkSpecs"]["num_patches"] > 1:
                current_target_metadata = []
                for ind in indices.numpy():
                    current_target_metadata.append(first_stage_metadata[ind])  
                inputs["first_stage_pos"] = torch.stack([scene["pos"] for scene in current_target_metadata], dim=0) # num_scenes x num_patches x 3
                inputs["first_stage_rot"] = torch.stack([scene["rot"] for scene in current_target_metadata], dim=0) # num_scenes x num_patches x 3
                inputs["first_stage_scale"] = torch.stack([scene["scale"] for scene in current_target_metadata], dim=0) # num_scenes x num_patches x 3

            # NN optimization

            #if specs["Tensorboard"] and current_batch == 0:
            #    tensorboard.add_graph(decoder, input_to_model=[list(inputs.items())], verbose=True)

            anomaly_detection = False # for debugging nan
            if anomaly_detection:
                torch.set_anomaly_enabled(True)

            inputs["sdf_gt"] = sdf_gt
            if patch_encoder:
                pred_sdf, regressed_latent_vectors, extra_loss = decoder(inputs)
                for i, ind in enumerate(indices.numpy()):
                    lat_vecs[ind] = regressed_latent_vectors[i].detach()
            elif use_depth:
                pred_sdf, extra_loss, extra_outputs, regressed_latent_vectors = decoder(inputs)
                for i, ind in enumerate(indices.numpy()):
                    lat_vecs[ind] = regressed_latent_vectors[i].detach().view(1, -1)
            else:
                if return_extra_outputs:
                    pred_sdf, extra_loss, extra_outputs = decoder(inputs)
                else:
                    pred_sdf, extra_loss = decoder(inputs)

            if current_stage == "1-2" and specs["NetworkSpecs"]["mixture_latent_mode"] == "all_implicit"  and specs["NetworkSpecs"]["num_patches"] > 1:
                current_patch_positions = extra_outputs["patch_positions"].detach().clone() # num_scenes x num_patches x 3
                current_patch_rotation = extra_outputs["patch_rotations"].detach().clone() # num_scenes x num_patches x 3
                current_patch_scaling = extra_outputs["patch_scalings"].detach().clone() # num_scenes x num_patches x 1
                for i, ind in enumerate(indices.numpy()):
                    metadata = {"pos": current_patch_positions[i,:,:], "rot": current_patch_rotation[i,:,:], "scale": current_patch_scaling[i,:,:]}
                    first_stage_metadata[ind] = metadata

            if enforce_minmax:
                pred_sdf = deep_sdf.utils.threshold_min_max(
                    pred_sdf, min_vec, max_vec
                )

            if specs["NetworkSpecs"]["use_curriculum_weighting"]:
                curriculum_deepsdf_lambda = 0.5
                curriculum_clamping_distance = 0.1
                clamped_patch_sdfs = torch.clamp(pred_sdf.view(-1).detach(), -curriculum_clamping_distance, +curriculum_clamping_distance)
                clamped_sdf_gt = torch.clamp(sdf_gt.view(-1).detach(), -curriculum_clamping_distance, +curriculum_clamping_distance)
                first_weight = torch.sign(clamped_sdf_gt) # samples 
                second_weight = torch.sign(clamped_sdf_gt - clamped_patch_sdfs) # samples 
                curriculum_weights = 1. + curriculum_deepsdf_lambda * first_weight * second_weight # samples 
                loss = main_loss_weight * torch.mean(curriculum_weights * torch.abs(pred_sdf.view(-1) - sdf_gt.view(-1)))
            else:
                loss = main_loss_weight * loss_l1(pred_sdf.view(-1), sdf_gt.view(-1))
            loss /= batch_split
            _add("reconstruction", loss)
            batch_sublosses["reconstruction"] += loss.item()

            for extra_loss_name, extra_loss_value in extra_loss.items():
                extra_loss_value = torch.mean(extra_loss_value) # for multi-GPU setting
                extra_loss_value /= batch_split
                if "latent_regularization" in extra_loss_name:
                    if specs["DeepSDFMode"]:
                        loss += code_reg_lambda * min(1, epoch / 100) * extra_loss_value
                    else:
                        loss += code_reg_lambda * min(1, epoch / 400) * extra_loss_value
                else:
                    loss += extra_loss_value
                _add(extra_loss_name, extra_loss_value)
                batch_sublosses[extra_loss_name] += extra_loss_value.item()

            if is_training:
                loss.backward()

            batch_loss += loss.item()
            
            # can stop gradient for certain stages. useful when doing all_explicit and staged training.
            if False:
            #if use_tiny_patchnet and specs["NetworkSpecs"]["mixture_latent_mode"] != "all_implicit":
                for ind in indices.numpy():
                    latent_ind = lat_vecs[ind]
                    for i in range(specs["NetworkSpecs"]["num_patches"]):
                        metadata_size = 7
                        offset = (patch_latent_size + metadata_size) * i
                        if current_stage == "1":
                            #print("always stopping some gradients during staged training!!!")
                            latent_ind.grad.data[0, offset : offset + patch_latent_size ] = 0. # keep patch latent fixed during first stage
                        if current_stage == "2": # shouldn't do anything for non-stages training since current_stage == "None" in that case?
                            #print("always stopping some gradients during staged training!!!")
                            latent_ind.grad.data[0, offset + patch_latent_size : offset + patch_latent_size + metadata_size] = 0. # keep metadata fixed during second stage

            _subbatch += 1
            if _subbatch >= batch_split:
                _subbatch = 0

                loss_log.append(batch_loss)

                if specs["Tensorboard"] and current_batch % specs["TensorboardParametersBatches"] == 0:
                    norm_type = 2.
                    total_gradient_norm = 0
                    for p in decoder.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(norm_type)
                            total_gradient_norm += param_norm.item() ** norm_type
                    total_gradient_norm = total_gradient_norm ** (1. / norm_type)

                    prefix = "training" if is_training else "test"
                    tensorboard.add_scalar(prefix + '_batch_loss', batch_loss, epoch)
                    tensorboard.add_scalar(prefix + '_global_gradient_norm', total_gradient_norm, epoch)
                    tensorboard.add_scalar('lrDecoder', lr_schedules[0].get_learning_rate(epoch), epoch)
                    tensorboard.add_scalar('lrLatent', lr_schedules[1].get_learning_rate(epoch), epoch)
                    for name, subloss in batch_sublosses.items():
                        tensorboard.add_scalar(prefix + "_subloss_" + name, subloss, epoch)


                    for name, weight in decoder.named_parameters():
                        tensorboard.add_histogram("network/" + name, weight, epoch) 
                        if weight.requires_grad and weight.grad is not None:
                            tensorboard.add_histogram("network/grad/" + name, weight.grad, epoch)

                if grad_clip is not None and is_training:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

                if is_training:
                    optimizer.step()
                    # This is necessary to correctly do autodecoding. It's a hacky workaround and is only ensured to work if Adam is used as optimizer.
                    for ind in batch_indices_to_optimize:
                        lat_vecs[ind].grad = None

                # Project latent vectors onto sphere
                if code_bound is not None:
                    deep_sdf.utils.project_vecs_onto_sphere(lat_vecs, code_bound)

                current_batch += 1

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        for key, (value, counter) in list(epoch_loss.items()):
            epoch_loss[key] = value / counter
        losses_log.append(epoch_loss)
        info = ("training" if is_training else "testing") + " epoch {} losses:".format(epoch)
        for key, value in list(epoch_loss.items()):
            info += " " + key + " " + str(value) + ","
        logging.info(info)

        if is_training:
            append_parameter_magnitudes(param_mag_log, decoder)
    
            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:
                save_latest(epoch)
                save_logs(
                    results_folder,
                    loss_log,
                    lr_log,
                    timing_log,
                    lat_mag_log,
                    param_mag_log,
                    epoch,
                )
        if not is_training and epoch == epochs[-1] and  test_split_file is not None: # last test epoch
            save_latent_vectors(results_folder, "test_" + str(epoch) + ".pth", test_lat_vecs, epoch)

        if current_stage == "1-2" and specs["NetworkSpecs"]["mixture_latent_mode"] == "all_implicit" and specs["NetworkSpecs"]["num_patches"] > 1:
             first_stage_metadata_folder = results_folder + "FirstStageMetadata/"
             create_folder(first_stage_metadata_folder)

             all_metadata = torch.zeros(0)
             for d in first_stage_metadata:
                object_metadata = torch.cat([d["pos"], d["rot"], d["scale"]], 1)
                all_metadata = torch.cat([all_metadata, object_metadata.cpu().unsqueeze(0)], 0)

             torch.save(
                {"epoch": epoch, "metadata": all_metadata},
                os.path.join(first_stage_metadata_folder, "metadata.pth"),
             )

    if specs["generate_initialization_for_object_latent"]:
        object_latent_init_folder = results_folder + "object_latent_init/"
        create_folder(object_latent_init_folder)

        object_latent_init_file = object_latent_init_folder + "init.npy"
        np.save(object_latent_init_file, decoder.module.patch_metadata.detach().cpu().numpy())

    # cleanup
    tensorboard.close()



def backup(results_folder, special_files):
    print("backing up... ", flush=True, end="")
    special_files_to_copy = ["specs.json"]
    subfolders_to_copy = ["", "deep_sdf/", "deep_sdf/metrics/", "networks/", "networks/torchdiffeq/", "networks/torchdiffeq/_impl/", "localization/", "examples/splits/"]

    this_file = os.path.realpath(__file__)
    this_folder = os.path.dirname(this_file) + "/"
    backup_folder = results_folder + "backup/"
    create_folder(backup_folder)
    # special files
    [shutil.copyfile(this_folder + file, backup_folder + file) for file in special_files_to_copy]
    shutil.copyfile(this_folder + "specs.json", results_folder + "specs.json")
    # folders
    for subfolder in subfolders_to_copy:
        create_folder(backup_folder + subfolder)
        files = os.listdir(this_folder + subfolder)
        files = [file for file in files if os.path.isfile(this_folder + subfolder + file) and file[-3:] == ".py"]
        [shutil.copyfile(this_folder + subfolder + file, backup_folder + subfolder + file) for file in files]

    # special files
    target_folder = backup_folder + "special_files/"
    create_folder(target_folder)
    for file in special_files:
        if file is not None:
            name = os.path.split(file)[1]
            shutil.copyfile(file, target_folder + name)

    print("done.", flush=True)


def create_folder(folder):
    import pathlib
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":


    import argparse

    specs_file = None

    code_folder = os.path.dirname(os.path.realpath(__file__))

    system_specific_settings = get_settings_dictionary()

    if specs_file is None:
        specs_file = code_folder + "/" + system_specific_settings["default_specs_file"]
    with open(specs_file) as specs:
        specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
        specs = json.loads(specs)

    results_folder = system_specific_settings["root_folder"] + "results/" + specs["ResultsFolder"]
    print(results_folder, flush=True)
    train_split_file = code_folder + "/" + specs["TrainSplit"]
    if "TestSplit" in specs:
        test_split_file = code_folder + "/" + specs["TestSplit"] 
    else:
        test_split_file = None

    create_folder(results_folder)
    shutil.rmtree(results_folder)
    backup(results_folder, [train_split_file, test_split_file])

    create_folder(results_folder + "logging/")
    deep_sdf.configure_logging(specs["Logging"], results_folder + "logging/log.log")

    main_function(specs, system_specific_settings, results_folder, code_folder, train_split_file, test_split_file=test_split_file)