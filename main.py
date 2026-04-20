# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, visualize_val_samples
from models import build_model
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--use_dab', action='store_true',
                        help="使用 DAB-DETR 架構（query_anchor+query_content+動態 sine 位置查詢+迭代 anchor refinement）")
    # DINO 相關參數（--use_dino 自動包含 --use_dab 的所有功能）
    parser.add_argument('--use_dino', action='store_true',
                        help="使用 DINO 架構：CDN 去噪訓練 + Mixed Query Selection + Look Forward Twice")
    parser.add_argument('--cdn_groups', default=1, type=int,
                        help="CDN 去噪組數（每組含正/負樣本各一份，更多組 = 更強監督信號）")
    parser.add_argument('--cdn_label_noise', default=0.5, type=float,
                        help="CDN 正樣本標籤隨機翻轉比例（0 = 不翻轉）")
    parser.add_argument('--cdn_box_noise', default=1.0, type=float,
                        help="CDN box 噪聲尺度（正樣本 < scale/2，負樣本 >= scale/2）")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--exp_name', default='exp', type=str,
                        help='experiment name, used for auto output dir: output/MMDD_<exp_name>')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='',
                        help='TensorBoard log dir, defaults to output_dir/runs if empty')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_weights', default='', type=str,
                        help='Path to pretrained checkpoint (base-DETR or DAB-DETR). '
                             'Loaded with strict=False: 相容 key 自動載入，新增 key 隨機初始化，'
                             '多餘 key 忽略。 --resume 仍可疊加使用（繼續訓練優先）。')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # ── Soft pretrain loading（base-DETR / DAB-DETR → DINO，strict=False）────
    if args.pretrain_weights and not args.resume:
        ckpt = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt)   # 相容 {'model': ...} 或裸 state_dict
        missing, unexpected = model_without_ddp.load_state_dict(state, strict=False)
        print(f'[pretrain] Loaded from {args.pretrain_weights}')
        print(f'[pretrain] missing  ({len(missing)}): {missing}')
        print(f'[pretrain] unexpected ({len(unexpected)}): {unexpected}')

    output_dir = Path(args.output_dir)

    # Save training parameters
    if utils.is_main_process() and args.output_dir:
        with (output_dir / 'params.txt').open('w') as f:
            for k, v in sorted(vars(args).items()):
                f.write(f'{k}: {v}\n')

    # TensorBoard writer (main process only)
    writer = None
    if utils.is_main_process() and args.output_dir:
        tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / 'runs'
        writer = SummaryWriter(log_dir=str(tb_dir))

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    global_step = 0
    best_mAP = 0.0
    history = {'train_loss': [], 'mAP': [], 'mAP50': []}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, global_step = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, writer=writer, global_step=global_step,
            steps_per_epoch=len(data_loader_train))
        lr_scheduler.step()

        if writer is not None:
            writer.add_scalar('train/lr_transformer', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/lr_backbone',    optimizer.param_groups[1]['lr'], epoch)
        # Save last checkpoint every epoch
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, output_dir / 'last.pth')

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            writer=writer, epoch=epoch, num_queries=args.num_queries
        )

        # Track metrics history
        history['train_loss'].append(train_stats['loss'])
        cur_mAP   = test_stats.get('coco_eval_bbox', [0])[0]
        cur_mAP50 = test_stats.get('coco_eval_bbox', [0, 0])[1]
        history['mAP'].append(cur_mAP)
        history['mAP50'].append(cur_mAP50)

        # Save best checkpoint
        if args.output_dir and cur_mAP >= best_mAP:
            best_mAP = cur_mAP
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, output_dir / 'best.pth')
            if utils.is_main_process():
                print(f'  -> New best mAP {best_mAP:.4f}, saved best.pth')
                visualize_val_samples(model, postprocessors, data_loader_val, device,
                                      args.output_dir, num_images=8, score_thresh=0.7)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if writer is not None:
        writer.close()

    # Plot and save metrics curve (2 subplots)
    if utils.is_main_process() and args.output_dir and history['train_loss']:
        epochs_x = list(range(args.start_epoch, args.start_epoch + len(history['train_loss'])))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(epochs_x, history['train_loss'], color='steelblue', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        if history['mAP']:
            ax2.plot(epochs_x, history['mAP'],   color='salmon',        label='mAP')
            ax2.plot(epochs_x, history['mAP50'], color='mediumseagreen', label='mAP50')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('AP')
            ax2.set_title('Validation mAP')
            ax2.legend()
            ax2.grid(True)
        plt.tight_layout()
        plt.savefig(str(output_dir / 'metrics_curve.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f'Saved metrics curve -> {output_dir}/metrics_curve.png')

    # Visualize val samples using best model
    if utils.is_main_process() and args.output_dir:
        best_path = output_dir / 'best.pth'
        if best_path.exists():
            ckpt = torch.load(str(best_path), map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(ckpt['model'])
        visualize_val_samples(model, postprocessors, data_loader_val, device, args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if not args.output_dir:
        date_str = datetime.datetime.now().strftime('%m%d')
        args.output_dir = f'output/{date_str}_{args.exp_name}'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
