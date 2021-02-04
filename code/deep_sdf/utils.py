#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch
import numpy as np


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )
    arg_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        default=None,
        help="Redundant. Replace debug and quiet flags.",
    )


def configure_logging(logging_level, log_file=None):
    logger = logging.getLogger()
    if logging_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif logging_level == "quiet":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if log_file is not None:
        file_logger_handler = logging.FileHandler(log_file)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))


def project_vecs_onto_sphere(vectors, radius, surface_only=False):
    for i in range(len(vectors)):
        v = vectors[i]
        length = torch.norm(v).detach()

        if surface_only or length.cpu().data.numpy() > radius:
            vectors[i].data = vectors[i].mul(radius / length)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    if type(sdf) == tuple:
        sdf = sdf[0]

    return sdf
