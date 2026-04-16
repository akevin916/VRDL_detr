# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    writer=None, global_step: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None and utils.is_main_process():
            writer.add_scalar('train/loss',        loss_value,                                       global_step)
            for k, v in loss_dict_reduced_scaled.items():
                # 只記錄主要 loss，跳過 auxiliary decoder layers (_0 ~ _4)
                if not any(k.endswith(f'_{i}') for i in range(5)):
                    writer.add_scalar(f'train/{k}', v.item(), global_step)
            writer.add_scalar('train/class_error', loss_dict_reduced['class_error'].item(),          global_step)
            writer.add_scalar('train/grad_norm',   grad_norm.item(),                                 global_step)
            writer.add_scalar('train/lr',          optimizer.param_groups[0]["lr"],                 global_step)
        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             writer=None, epoch: int = 0, num_queries: int = 100):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if writer is not None and utils.is_main_process() and 'bbox' in coco_evaluator.coco_eval:
            s = coco_evaluator.coco_eval['bbox'].stats
            writer.add_scalar('val/mAP',          s[0], epoch)
            writer.add_scalar('val/mAP50',         s[1], epoch)
            writer.add_scalar('val/mAP75',         s[2], epoch)
            writer.add_scalar('val/mAP_small',     s[3], epoch)
            writer.add_scalar('val/mAP_medium',    s[4], epoch)
            writer.add_scalar('val/mAP_large',     s[5], epoch)
            writer.add_scalar('val/AR_maxDets1',                      s[6], epoch)
            writer.add_scalar('val/AR_maxDets10',                     s[7], epoch)
            writer.add_scalar(f'val/AR_maxDets{num_queries}',         s[8], epoch)
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def visualize_val_samples(model, postprocessors, data_loader, device, output_dir,
                           num_images=16, score_thresh=0.7):
    """對 val set 前 num_images 張做推論，畫出預測框後存成 samples.png。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from pathlib import Path

    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    collected = []  # list of (img_np, boxes, labels, scores)
    for samples, targets in data_loader:
        samples_dev = samples.to(device)
        orig_sizes  = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        outputs     = model(samples_dev)
        results     = postprocessors['bbox'](outputs, orig_sizes)

        imgs = samples.tensors.cpu()
        for i in range(imgs.shape[0]):
            if len(collected) >= num_images:
                break
            h_orig, w_orig = targets[i]['orig_size'].tolist()
            img = (imgs[i] * std + mean).clamp(0, 1)[:, :int(h_orig), :int(w_orig)]
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            keep   = results[i]['scores'].cpu() >= score_thresh
            boxes  = results[i]['boxes'][keep].cpu()
            labels = results[i]['labels'][keep].cpu()
            scores = results[i]['scores'][keep].cpu()
            collected.append((img_np, boxes, labels, scores))

        if len(collected) >= num_images:
            break

    if not collected:
        return

    n_cols = min(4, len(collected))
    n_rows = (len(collected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)

    for idx, (img_np, boxes, labels, scores) in enumerate(collected):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.imshow(img_np)
        ax.axis('off')
        for box, lbl, sc in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 2, 0), f'{lbl.item() - 1}:{sc:.2f}',
                    color='white', fontsize=7, backgroundcolor='red', va='bottom')

    # 隱藏多餘格子
    for idx in range(len(collected), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    save_path = Path(output_dir) / 'samples.png'
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved validation samples -> {save_path}')

