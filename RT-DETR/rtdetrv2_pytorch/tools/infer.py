"""
Inference script for NYCU HW2 digit detection.
Loads a trained checkpoint, runs on all test images,
and writes COCO-format predictions to answer.json.

Usage:
    cd rtdetrv2_pytorch
    python tools/infer.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_nycu.yml \
        -r output/rtdetrv2_r18vd_120e_nycu/checkpoint0119.pth \
        --test-dir ../data/nycu-hw2-data/test \
        --output answer.json \
        --device cuda \
        --threshold 0.5
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.core import YAMLConfig


def build_model(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return Model().to(args.device)


def main(args):
    model = build_model(args)
    model.eval()

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    test_dir = Path(args.test_dir)
    image_paths = sorted(test_dir.glob('*.png'), key=lambda p: int(p.stem))

    # label (0-based) -> category_id (1-based) for NYCU digit dataset
    # label 0 = digit "0" = category_id 1, ..., label 9 = digit "9" = category_id 10
    label2category = {i: i + 1 for i in range(10)}

    results = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc='Inferring'):
            image_id = int(img_path.stem)

            im = Image.open(img_path).convert('RGB')
            w, h = im.size
            orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(args.device)

            im_tensor = transforms(im).unsqueeze(0).to(args.device)

            labels, boxes, scores = model(im_tensor, orig_size)

            # labels/boxes/scores: each is shape [1, num_queries]
            labels = labels[0].cpu()
            boxes  = boxes[0].cpu()   # xyxy, absolute coords
            scores = scores[0].cpu()

            keep = scores > args.threshold
            labels = labels[keep]
            boxes  = boxes[keep]
            scores = scores[keep]

            for label, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box.tolist()
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                results.append({
                    'image_id':    image_id,
                    'category_id': label2category[int(label.item())],
                    'bbox':        [round(v, 2) for v in bbox_xywh],
                    'score':       round(float(score.item()), 4),
                })

    with open(args.output, 'w') as f:
        json.dump(results, f)

    print(f'Saved {len(results)} predictions to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',    type=str, required=True)
    parser.add_argument('-r', '--resume',    type=str, required=True,
                        help='path to trained checkpoint (.pth)')
    parser.add_argument('--test-dir',        type=str, required=True,
                        help='directory containing test images')
    parser.add_argument('--output',          type=str, default='answer.json')
    parser.add_argument('--device',          type=str, default='cuda')
    parser.add_argument('--threshold',       type=float, default=0.5,
                        help='score threshold for filtering predictions')
    args = parser.parse_args()
    main(args)
