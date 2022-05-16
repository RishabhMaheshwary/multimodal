# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils

# import util.dist as dist
# import util.misc as utils
from datasets import CocoDataModule

# from engine import evaluate, train_one_epoch
from models import build_model
from utils.args_parse import get_args_parser
from utils.misc import targets_to


def init_distributed_mode(args):

    args.distributed = True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False

    # torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}'.format(
    #     args.rank, args.dist_url), flush=True)

    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def train(model, dataloader, device):
    model.train()
    # for batch_dict in dataloader:
    for i, batch_dict in enumerate(dataloader):
        # curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        positive_map = (
            batch_dict["positive_map"].to(device)
            if "positive_map" in batch_dict
            else None
        )
        targets = batch_dict["targets"]
        answers = (
            {k: v.to(device) for k, v in batch_dict["answers"].items()}
            if "answers" in batch_dict
            else None
        )
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)
        memory_cache = model(samples, captions, encode_and_save=True)
        outputs = model(
            samples, captions, encode_and_save=False, memory_cache=memory_cache
        )


def main(args):

    if args.dataset_config is not None:

        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed  # + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dm = CocoDataModule(args, distributed=args.distributed)
    dm.prepare_data()
    dm.setup()
    data_loader = dm.train_dataloader()
    model = build_model(args)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    train(model, data_loader, device)
    # dataset_train = ConcatDataset(
    #     [build_dataset(name, image_set="train", args=args) for name in args.combine_datasets]
    # )
    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # data_loader_train = DataLoader(
    #     dataset_train,
    #     batch_sampler=batch_sampler_train,
    #     collate_fn=partial(utils.collate_fn, False),
    #     num_workers=args.num_workers,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
