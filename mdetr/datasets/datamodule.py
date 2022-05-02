# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DistributedSampler
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from utils.misc import collate_fn

from .coco import build as build_coco
from .coco import make_coco_transforms
from .flickr import build as build_flickr
from .mixed import build as build_mixed


class CocoDataModule(LightningDataModule):
    def __init__(self, dataset_config, distributed=False):
        super().__init__()
        self.transforms = make_coco_transforms
        self.dataset_config = dataset_config
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.distributed = distributed
        self.batch_size = dataset_config.batch_size
        self.num_workers = 0
        self.train, self.val = None, None

    def prepare_data(self):
        print("prepare")
        # download

    def setup(self):
        train = []
        image_set = "train"
        for data in self.dataset_config.combine_datasets:
            if data == "coco":
                dataset = build_coco("train", self.dataset_config)
            elif data == "flickr":
                dataset = build_flickr(
                    "train", self.dataset_config, self.tokenizer, self.transforms
                )
            elif data == "mixed":
                dataset = build_mixed(
                    "train", self.dataset_config, self.tokenizer, self.transforms
                )
            train.append(dataset)
        self.train = ConcatDataset(train)

    def train_dataloader(self):
        if self.distributed:
            self.train = DistributedSampler(self.train)
        else:
            self.train = torch.utils.data.RandomSampler(self.train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            self.train, self.batch_size, drop_last=True
        )
        data_loader_train = DataLoader(
            self.train,
            batch_sampler=batch_sampler_train,
            collate_fn=partial(collate_fn, False),
            num_workers=self.num_workers,
        )
        return data_loader_train

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)
