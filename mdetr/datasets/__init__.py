# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .datamodule import CocoDataModule
from .flickr import build as build_flickr
from .flickr_eval import FlickrEvaluator
from .mixed import CustomCocoDetection
from .mixed import build as build_mixed


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, (torchvision.datasets.CocoDetection, CustomCocoDetection)):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
