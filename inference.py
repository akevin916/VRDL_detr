"""
推理腳本：對 test 資料夾所有圖片執行 DETR 推理，輸出 COCO 格式 JSON。

使用範例（純推理）:
    python inference.py \
        --checkpoint output/0420_exp/best.pth \
        --test_dir   data/nycu-hw2-data/test \
        --output     results/pred.json \
        --score_thr  0.05 \
        --nms_iou    0.5 \
        --vis_n      5 \
        --device     cuda

使用範例（對 val set 掃描最佳 nms_iou）:
    python inference.py \
        --checkpoint output/0420_exp/best.pth \
        --test_dir   data/nycu-hw2-data/valid \
        --gt_json    data/nycu-hw2-data/valid.json \
        --output     results/pred_val.json \
        --score_thr  0.05 \
        --vis_n      0 \
        --device     cuda

輸出:
    results/pred.json     — COCO 格式預測
    results/vis_XXXXX.png — 視覺化圖（預設 5 張）
    results/nms_curve.png — nms_iou vs mAP 曲線（需 --gt_json）

輸出格式:
    [{"image_id": 10000, "bbox": [x, y, w, h], "score": 0.97, "category_id": 5}, ...]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from tqdm import tqdm

import datasets.transforms as T
from models import build_model
from torchvision.ops import nms as torchvision_nms

# category_id 1–10 對應數字 0–9
CATEGORY_NAMES = {i: str(i - 1) for i in range(1, 11)}


def apply_nms(scores: torch.Tensor, labels: torch.Tensor,
              boxes: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Per-class NMS：對每個類別分別執行 NMS，回傳保留的 index tensor。

    Args:
        scores : (N,)
        labels : (N,)  類別編號
        boxes  : (N, 4)  [x0, y0, x1, y1]
        iou_thr: IoU 門溻，超過此値的重叠框會被移除
    Returns:
        keep : (M,) 保留的索引
    """
    if iou_thr <= 0 or len(boxes) == 0:
        return torch.arange(len(boxes))

    keep_all = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        kept = torchvision_nms(boxes[idx], scores[idx], iou_thr)
        keep_all.append(idx[kept])
    return torch.cat(keep_all)


# --------------------------------------------------------------------------- #
# 與訓練相同的 val 前處理 (resize 短邊到 480, max_size 不超過 1333)
# --------------------------------------------------------------------------- #
def make_val_transform():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return T.Compose([
        T.RandomResize([480], max_size=1333),
        normalize,
    ])


# --------------------------------------------------------------------------- #
# 從 checkpoint 還原 args（保留訓練時的模型架構設定）
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # checkpoint 裡存有訓練時的 args
    train_args = ckpt['args']

    # 舊 checkpoint（flag 加入前訓練的）沒有 use_dab 欄位
    # → 從 state_dict key 自動判斷架構，避免手動指定
    state_keys = set(ckpt['model'].keys())
    if 'query_anchor.weight' in state_keys:
        train_args.use_dab = True
    else:
        train_args.use_dab = False

    model, _, postprocessors = build_model(train_args)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f"載入 checkpoint: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}, "
          f"num_queries={train_args.num_queries}, "
          f"dilation={getattr(train_args, 'dilation', False)}, "
          f"use_dab={train_args.use_dab}")

    return model, postprocessors


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_inference(args):
    device = torch.device(args.device)
    model, postprocessors = load_model(args.checkpoint, device)
    post_process = postprocessors['bbox']

    transform = make_val_transform()

    # 收集所有 test 圖片（依名稱排序，保持一致性）
    test_dir = Path(args.test_dir)
    img_paths = sorted(test_dir.glob('*.png')) + sorted(test_dir.glob('*.jpg'))
    if not img_paths:
        raise FileNotFoundError(f"在 {test_dir} 找不到任何圖片")
    print(f"共 {len(img_paths)} 張 test 圖片")

    results = []
    # 儲存每張圖的【原始未過濾】預測，供 threshold sweep 用
    raw_preds = {}   # {image_id: {'scores': list, 'labels': list, 'boxes_xywh': list}}
    # 儲存每張圖的推理結果供事後視覺化（只存 pil_img + filtered preds）
    vis_pool = []   # list of (pil_img, image_id, scores, labels, boxes_xyxy)

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc='推理中'):
            # image_id：使用檔名數字部分（e.g. "10000.png" → 10000）
            image_id = int(img_path.stem)

            # 載入圖片
            pil_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = pil_img.size  # PIL: (w, h)

            # 前處理（傳入空 target，transforms 內部只對 rgb 操作）
            tensor, _ = transform(pil_img, {})
            tensor = tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

            # 原始尺寸 (h, w) 供 PostProcess 反算絕對座標
            orig_size = torch.tensor([[orig_h, orig_w]], dtype=torch.float32, device=device)

            # 推理
            outputs = model(tensor)

            # PostProcess：boxes → 絕對 [x0,y0,x1,y1]
            preds = post_process(outputs, orig_size)[0]  # 取 batch[0]

            scores_all = preds['scores'].cpu()
            labels_all = preds['labels'].cpu()
            boxes_all  = preds['boxes'].cpu()    # [x0,y0,x1,y1]

            # 存完全原始預測（不套任何過濾）供 sweep 用
            boxes_xywh_all = boxes_all.clone()
            boxes_xywh_all[:, 2] -= boxes_xywh_all[:, 0]
            boxes_xywh_all[:, 3] -= boxes_xywh_all[:, 1]
            raw_preds[image_id] = {
                'scores': scores_all.tolist(),
                'labels': labels_all.tolist(),
                'boxes_xywh': boxes_xywh_all.tolist(),
                'boxes_xyxy': boxes_all.tolist(),  # NMS sweep 需要 xyxy 格式
            }

            # 套用 score_thr 過濾
            keep    = scores_all >= args.score_thr
            scores  = scores_all[keep]
            labels  = labels_all[keep]
            boxes   = boxes_all[keep]

            # NMS：移除指向同一物體的重複框
            if args.nms_iou > 0 and len(boxes) > 0:
                keep_nms = apply_nms(scores, labels, boxes, args.nms_iou)
                scores   = scores[keep_nms]
                labels   = labels[keep_nms]
                boxes    = boxes[keep_nms]

            # 暫存供視覺化
            vis_pool.append((pil_img, image_id, scores, labels, boxes))

            # 轉 COCO 格式 [x, y, w, h]
            for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
                x0, y0, x1, y1 = box
                results.append({
                    'image_id':   image_id,
                    'bbox':       [x0, y0, x1 - x0, y1 - y0],
                    'score':      score,
                    'category_id': label,
                })

    # 寫出 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n完成！共 {len(results)} 筆預測，已存至 {output_path}")

    # GT 評估：固定 score_thr，掃描 nms_iou
    if args.gt_json:
        eval_nms_sweep(raw_preds, args.gt_json, output_path.parent,
                       score_thr=args.score_thr,
                       step=args.sweep_nms_step,
                       highlight=args.nms_iou)

    # 視覺化：從所有圖片中隨機挑選 vis_n 張
    if args.vis_n > 0:
        vis_dir = output_path.parent
        n = min(args.vis_n, len(vis_pool))
        chosen = random.sample(vis_pool, n)
        for pil_img, image_id, scores, labels, boxes in chosen:
            save_path = vis_dir / f'vis_{image_id}.png'
            visualize(pil_img, scores.tolist(), labels.tolist(), boxes.tolist(), save_path)
        print(f"視覺化圖片已存至 {vis_dir}/vis_*.png")


# --------------------------------------------------------------------------- #
# NMS IoU sweep 評估（固定 score_thr，掃描 nms_iou）
# --------------------------------------------------------------------------- #
def eval_nms_sweep(raw_preds: dict, gt_json: str, out_dir: Path,
                  score_thr: float = 0.05, step: float = 0.05, highlight: float = 0.5):
    """固定 score_thr，掃描不同 nms_iou，找最佳 NMS 設定。

    Args:
        raw_preds:  {image_id: {'scores', 'labels', 'boxes_xywh', 'boxes_xyxy'}}
        gt_json:    GT COCO JSON 路徑
        out_dir:    結果輸出目錄
        score_thr:  固定的信心門檻
        step:       nms_iou 掃描間距
        highlight:  對應 --nms_iou，在表格中標示 ★
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, os

    print(f"\n=== NMS IoU Sweep（score_thr={score_thr:.2f}, GT: {gt_json}）===")
    coco_gt = COCO(gt_json)

    # 加入 0（不做 NMS）作為對照
    nms_iou_list = [0.0] + [round(t, 2) for t in np.arange(step, 1.0 + step / 2, step)]
    rows = []

    for niou in nms_iou_list:
        dt_list = []
        for image_id, pred in raw_preds.items():
            scores = torch.tensor(pred['scores'])
            labels = torch.tensor(pred['labels'])
            boxes_xyxy = torch.tensor(pred['boxes_xyxy'])
            boxes_xywh = torch.tensor(pred['boxes_xywh'])

            # 先套 score_thr
            keep = scores >= score_thr
            scores     = scores[keep]
            labels     = labels[keep]
            boxes_xyxy = boxes_xyxy[keep]
            boxes_xywh = boxes_xywh[keep]

            # 再套 NMS（niou=0 代表不做）
            if niou > 0 and len(boxes_xyxy) > 0:
                keep_nms   = apply_nms(scores, labels, boxes_xyxy, niou)
                scores     = scores[keep_nms]
                labels     = labels[keep_nms]
                boxes_xywh = boxes_xywh[keep_nms]

            for s, l, b in zip(scores.tolist(), labels.tolist(), boxes_xywh.tolist()):
                dt_list.append({
                    'image_id':    image_id,
                    'category_id': l,
                    'bbox':        b,
                    'score':       s,
                })

        if not dt_list:
            rows.append((niou, 0, 0.0, 0.0, 0.0))
            continue

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = coco_gt.loadRes(dt_list)
                ev = COCOeval(coco_gt, coco_dt, 'bbox')
                ev.evaluate()
                ev.accumulate()
                ev.summarize()

        stats = ev.stats
        rows.append((niou, len(dt_list), float(stats[0]), float(stats[1]), float(stats[2])))

    # 印表格
    header = f"{'nms_iou':>8}  {'#preds':>7}  {'mAP':>7}  {'mAP50':>7}  {'mAP75':>7}"
    print(header)
    print('-' * len(header))
    best_idx = int(np.argmax([r[2] for r in rows]))
    for i, (niou, n, mAP, mAP50, mAP75) in enumerate(rows):
        label = '(no NMS)' if niou == 0.0 else ''
        mark  = ' ★' if abs(niou - highlight) < 1e-6 else ('  *best*' if i == best_idx else '')
        print(f"{niou:>8.2f}  {n:>7d}  {mAP:>7.4f}  {mAP50:>7.4f}  {mAP75:>7.4f}  {label}{mark}")
    best = rows[best_idx]
    print(f"\n最佳 nms_iou: {best[0]:.2f}  →  mAP={best[2]:.4f}, mAP50={best[3]:.4f}")

    # 存曲線圖
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    nious  = [r[0] for r in rows]
    mAPs   = [r[2] for r in rows]
    mAP50s = [r[3] for r in rows]
    ax.plot(nious, mAPs,   marker='o', label='mAP')
    ax.plot(nious, mAP50s, marker='s', label='mAP50')
    ax.axvline(x=highlight, color='gray', linestyle='--', alpha=0.6, label=f'nms_iou={highlight}')
    ax.axvline(x=best[0],   color='red',  linestyle=':',  alpha=0.8, label=f'best={best[0]}')
    ax.set_xlabel('nms_iou')
    ax.set_ylabel('mAP')
    ax.set_title(f'NMS IoU vs mAP  (score_thr={score_thr:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = out_dir / 'nms_curve.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"曲線圖已存至 {save_path}")


# --------------------------------------------------------------------------- #
# 視覺化單張圖片
# --------------------------------------------------------------------------- #
def visualize(pil_img, scores, labels, boxes_xyxy, save_path):
    """在圖片上畫出預測框並存檔。"""
    fig, ax = plt.subplots(1, figsize=(max(6, pil_img.width / 60), max(4, pil_img.height / 60)))
    ax.imshow(pil_img)
    ax.axis('off')

    cmap = plt.get_cmap('tab10')

    for score, label, box in zip(scores, labels, boxes_xyxy):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        color = cmap((label - 1) % 10)
        rect = patches.Rectangle((x0, y0), w, h,
                                  linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        digit = CATEGORY_NAMES.get(label, str(label))
        ax.text(x0, y0 - 2, f'{digit} {score:.2f}',
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.7, pad=1, linewidth=0))

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# arg parser
# --------------------------------------------------------------------------- #
def get_args_parser():
    parser = argparse.ArgumentParser('DETR 推理腳本', add_help=True)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='模型 checkpoint 路徑（e.g. output/0420_exp/best.pth）')
    parser.add_argument('--test_dir', default='data/nycu-hw2-data/test', type=str,
                        help='test 圖片資料夾')
    parser.add_argument('--output', default='results/pred.json', type=str,
                        help='輸出 JSON 檔案路徑')
    parser.add_argument('--vis_n', default=5, type=int,
                        help='視覺化圖片數量（0 = 不輸出）')
    parser.add_argument('--gt_json', default='', type=str,
                        help='GT 標注 JSON（e.g. data/nycu-hw2-data/valid.json）；'
                             '提供後自動掃描 nms_iou 並存曲線圖')
    parser.add_argument('--sweep_nms_step', default=0.05, type=float,
                        help='nms_iou 掃描間距（預設 0.05）')
    parser.add_argument('--nms_iou', default=0.5, type=float,
                        help='NMS IoU 門檻（0 = 不做 NMS，預設 0.5）')
    parser.add_argument('--score_thr', default=0.05, type=float,
                        help='信心分數門檻（預設 0.05）')
    parser.add_argument('--device', default='cuda', type=str,
                        help='推理裝置（cuda / cpu）')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    run_inference(args)
