#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess

import deep_sdf
import deep_sdf.workspace as ws


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    #logging.info(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    #subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc = subprocess.Popen(command)
    subproc.wait()

    returncode = subproc.returncode
    if returncode != 0:
        logging.info(mesh_filepath + " " + str(returncode))


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


# make -C build -j 8 && python preprocess_data.py -d /HPS/implicits/work/datasets/shapenet_v2/preprocessed/ -s /HPS/deepShape/nobackup/datasets/ShapeNetCorev2 -n ShapeNetV2 --split /HPS/implicits/work/local/DeepSDF-master/examples/splits/sv2_sofas_train.json --threads 16
# make -C build -j 8 && python preprocess_data.py -d /HPS/implicits_datasets/work/shapenetv1/deepsdf_preprocessed/ -s /HPS/implicits_datasets/work/shapenetv1/watertight/ -n ShapeNetV1 --threads 16

if __name__ == "__main__":

    process_all_13_shapenet_categories = False
    process_all_13_shapenet_categories_split = "train"

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
        + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=not process_all_13_shapenet_categories,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        #default=False,
        default=process_all_13_shapenet_categories and process_all_13_shapenet_categories_split != "train",
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )
    arg_parser.add_argument(
        "--depth",
        dest="frontal_depth",
        default=False, 
        action="store_true",
        help="If set, the script will include a frontal depth map in the final npz. ",
    )
    arg_parser.add_argument(
        "--randomdepth",
        dest="random_depth",
        default=False,
        action="store_true",
        help="If set, the script will include two random depth maps in the final npz. ",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    if args.surface_sampling:
        executable = "bin/SampleVisibleMeshSurface"
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = "bin/PreprocessMesh"
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]
        if args.frontal_depth:
            additional_general_args += ["--frontal_depth"]
        if args.random_depth:
            additional_general_args += ["--random_depth"]

    if not process_all_13_shapenet_categories:
        with open(args.split_filename, "r") as f:
            split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

    logging.info(
        "Preprocessing data from "
        + args.source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    if process_all_13_shapenet_categories:
        split = "./shapenet_1000.json"
        with open(split, "r") as split:
            split = json.load(split)
        class_directories = list(split.keys())
        print(class_directories)
    else:
        class_directories = split[args.source_name]

    meshes_targets_and_specific_args = []

    for class_dir in class_directories:
        class_path = os.path.join(args.source_dir, class_dir)
        if process_all_13_shapenet_categories:
            instance_dirs = os.listdir(class_path)
            instance_dirs = [dir for dir in instance_dirs if dir[-4:] == ".obj"]
            instance_dirs = sorted(instance_dirs)
            if process_all_13_shapenet_categories_split == "train":
                instance_dirs = instance_dirs[:int(0.75 * len(instance_dirs))]
            elif process_all_13_shapenet_categories_split == "val":
                instance_dirs = instance_dirs[int(0.75 * len(instance_dirs)):int(0.8 * len(instance_dirs))]
            else:
                instance_dirs = instance_dirs[int(0.8 * len(instance_dirs)):]

            print(len(instance_dirs))
        else:
            instance_dirs = class_directories[class_dir]

        logging.debug(
            "Processing " + str(len(instance_dirs)) + " instances of class " + class_dir
        )

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        for instance_dir in instance_dirs:

            shape_dir = os.path.join(class_path, instance_dir)

            processed_filepath = os.path.join(target_dir, instance_dir + extension)
            if args.skip and os.path.isfile(processed_filepath):
                logging.debug("skipping " + processed_filepath)
                continue

            try:
                if shape_dir[-4:] == ".obj" and os.path.isfile(shape_dir):
                    shape_dir, mesh_filename = os.path.split(shape_dir)
                else:
                    mesh_filename = deep_sdf.data.find_mesh_in_directory(shape_dir)
                    if type(mesh_filename) != str:
                        print(mesh_filename)
                        print(shape_dir, flush=True)
                        raise deep_sdf.data.NoMeshFileError

                specific_args = []

                if args.surface_sampling:
                    normalization_param_target_dir = os.path.join(
                        normalization_param_dir, class_dir
                    )

                    if not os.path.isdir(normalization_param_target_dir):
                        os.mkdir(normalization_param_target_dir)

                    normalization_param_filename = os.path.join(
                        normalization_param_target_dir, instance_dir + ".npz"
                    )
                    specific_args = ["-n", normalization_param_filename]

                meshes_targets_and_specific_args.append(
                    (
                        os.path.join(shape_dir, mesh_filename),
                        processed_filepath,
                        specific_args,
                    )
                )
            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + instance_dir)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + instance_dir)


    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_filepath,
            target_filepath,
            specific_args,
        ) in meshes_targets_and_specific_args:
            executor.submit(
                process_mesh,
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )

        executor.shutdown()
