# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data class for the Flickr30k entities dataset. The task considered is phrase grounding.
"""
from pathlib import Path

from .coco import ModulatedDetection


class FlickrDetection(ModulatedDetection):
    pass


def build(image_set, args, tokenizer=None, transforms=None):

    img_dir = Path(args.flickr_img_path) / "train"

    if args.GT_type == "merged":
        identifier = "mergedGT"
    elif args.GT_type == "separate":
        identifier = "separateGT"
    else:
        assert False, f"{args.GT_type} is not a valid type of annotation for flickr"

    if args.test:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test.json"
    else:
        ann_file = (
            Path(args.flickr_ann_path) / f"final_flickr_{identifier}_{image_set}.json"
        )

    dataset = FlickrDetection(
        img_dir,
        ann_file,
        transforms=transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
        is_train=image_set == "train",
    )
    return dataset
