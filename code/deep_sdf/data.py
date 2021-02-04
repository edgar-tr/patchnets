#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    npzfiles += [instance_filename]

                else:
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj"))  + list(
        glob.iglob(shape_dir + ".obj"))
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    tensor_nan_fixed = torch.isnan(tensor).any(-1)
    if (tensor_nan != tensor_nan_fixed).any():
        import ipdb; ipdb.set_trace()
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    # split the sample into half
    half = int(subsample / 2)

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        use_normals=False,
        use_depth=False
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.use_normals = use_normals
        self.use_depth = use_depth

        if load_ram:
            self.loaded_data = []
            self.loaded_depth = []
            for i, f in enumerate(self.npyfiles):
                if i % 1000 == 0:
                    print("loading: " + str(i) + " / " + str(len(self.npyfiles)), flush=True)
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                if use_normals:
                    pos_tensor = remove_nans(torch.from_numpy(  np.concatenate([npz["pos"], npz["pos_normals"]], axis=1)  ))
                    neg_tensor = remove_nans(torch.from_numpy(  np.concatenate([npz["neg"], npz["neg_normals"]], axis=1)  ))
                else:
                    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

                max_samples_per_object = 200000
                if pos_tensor.shape[0] + neg_tensor.shape[0] > max_samples_per_object:
                    ratio = max_samples_per_object / float(pos_tensor.shape[0] + neg_tensor.shape[0])
                    pos_number = int(ratio * pos_tensor.shape[0])
                    neg_number = int(ratio * neg_tensor.shape[0])

                    pos_indices = np.random.choice(pos_tensor.shape[0], size=pos_number, replace=False)
                    neg_indices = np.random.choice(neg_tensor.shape[0], size=neg_number, replace=False)

                    pos_tensor = pos_tensor[pos_indices,:].clone()
                    neg_tensor = neg_tensor[neg_indices,:].clone()

                if self.use_depth:
                    depth_map = torch.from_numpy(npz["depth"])
                    self.loaded_depth.append(depth_map)
                    
                noise_std = None
                if noise_std is not None:
                    pos_tensor[:,3] += torch.from_numpy(np.random.normal(scale=noise_std, size=pos_tensor.shape[0])).float()
                    neg_tensor[:,3] += torch.from_numpy(np.random.normal(scale=noise_std, size=neg_tensor.shape[0])).float()


                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        dummy = torch.zeros(0)
        if self.load_ram:
            if self.use_depth:
                depth = self.loaded_depth[idx]
            else:
                depth = dummy
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx, depth
            )
        else:
            if self.use_depth:
                depth = torch.from_numpy(np.load(filename)["depth"])
            else:
                depth = dummy
            return unpack_sdf_samples(filename, self.subsample), idx, depth#
