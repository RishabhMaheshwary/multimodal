# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from .coco import build as build_coco
from .flickr import build as build_flickr
from .mixed import build as build_mixed


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
