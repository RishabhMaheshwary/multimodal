# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import random
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.utils
import utils.dist as dist
from datasets import CocoDataModule, get_coco_api_from_dataset, FlickrEvaluator

# from engine import evaluate, train_one_epoch
from models import build_model, build_postprocessors
from utils.args_parse import get_args_parser
from utils.metrics import MetricLogger, SmoothedValue
from utils.misc import targets_to
from utils.optim import adjust_learning_rate, update_ema


def train(
    model,
    dataloader,
    device,
    cur_epoch,
    optimizer,
    weight_dict,
    args,
    criterion=None,
    contrastive_criterion=None,
    epochs=40,
    max_norm=0.0,
    model_ema=None,
):

    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(cur_epoch)
    print_freq = 10

    num_training_steps = int(len(dataloader) * epochs)
    # for batch_dict in dataloader:
    for i, batch_dict in enumerate(
        metric_logger.log_every(dataloader, print_freq, header)
    ):
        curr_step = cur_epoch * len(dataloader) + i
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
        outputs, memory_cache = model(
            samples, targets, positive_map, captions, encode_and_save=True
        )
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_criterion_loss = contrastive_criterion(
                memory_cache["text_pooled_op"], memory_cache["img_pooled_op"]
            )
            loss_dict["contrastive_loss"] = contrastive_criterion_loss
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        print(loss_dict_reduced_scaled)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            cur_epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    contrastive_criterion,
    postprocessors,
    weight_dict,
    data_loader,
    evaluator_list,
    device,
    args,
):

    model.eval()
    if criterion is not None:
        criterion.eval()
    if contrastive_criterion is not None:
        contrastive_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
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

        outputs, memory_cache = model(
            samples, targets, positive_map, captions, encode_and_save=True
        )

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(
                memory_cache["text_pooled_op"], memory_cache["img_pooled_op"]
            )
            loss_dict["contrastive_loss"] = contrastive_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        if not args.no_detection:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors["segm"](
                    results, outputs, orig_target_sizes, target_sizes
                )

            flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
            if "flickr_bbox" in postprocessors.keys():
                image_ids = [t["original_img_id"] for t in targets]
                sentence_ids = [t["sentence_id"] for t in targets]
                items_per_batch_element = [t["nb_eval"] for t in targets]
                positive_map_eval = batch_dict["positive_map_eval"].to(device)
                flickr_results = postprocessors["flickr_bbox"](
                    outputs,
                    orig_target_sizes,
                    positive_map_eval,
                    items_per_batch_element,
                )
                assert len(flickr_results) == len(image_ids) == len(sentence_ids)
                for im_id, sent_id, output in zip(
                    image_ids, sentence_ids, flickr_results
                ):
                    flickr_res.append(
                        {"image_id": im_id, "sentence_id": sent_id, "boxes": output}
                    )

            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, results)
            }

            for evaluator in evaluator_list:
                if isinstance(evaluator, FlickrEvaluator):
                    evaluator.update(flickr_res)
                else:
                    evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    flickr_res = None
    for evaluator in evaluator_list:

        if isinstance(evaluator, FlickrEvaluator):
            flickr_res = evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    return stats


def main(args):

    if args.dataset_config is not None:

        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)
    dist.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed  # + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dm = CocoDataModule(args, distributed=args.distributed)
    dm.setup()
    data_loader, train_sampler = dm.train_dataloader()
    model, criterion, contrastive_criterion, weight_dict = build_model(args)
    model.to(device)
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")
    args.start_epoch = 0

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print(
                    "WARNING: ema model not found in checkpoint, resetting to current model"
                )
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    if args.eval:

        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print(
                    "WARNING: ema model not found in checkpoint, resetting to current model"
                )
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

        Val_all = namedtuple(
            typename="val_data",
            field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"],
        )
        val_tuples = []
        val_loader, val_sampler, dset = dm.val_dataloader()
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(
            Val_all(
                dataset_name="flickr",
                dataloader=val_loader,
                base_ds=base_ds,
                evaluator_list=None,
            )
        )

        test_stats = {}
        test_model = model_ema if model_ema is not None else model

        evaluator_list = [
            (
                FlickrEvaluator(
                    args.flickr_dataset_path,
                    "val",
                    merge_boxes=args.GT_type == "merged",
                )
            )
        ]
        for i, item in enumerate(val_tuples):
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
            )
            test_stats.update(
                {item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()}
            )

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)
        return
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_stats = train(
            model,
            data_loader,
            device,
            epoch,
            optimizer,
            weight_dict,
            args,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            epochs=args.epochs,
            max_norm=args.clip_max_norm,
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
        # wandb.log(train_stats)
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
