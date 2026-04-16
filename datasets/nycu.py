# Custom dataset for NYCU HW2 digit detection (COCO format, 10 classes: digits 0-9)
from pathlib import Path

import datasets.transforms as T
from .coco import CocoDetection


def make_nycu_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([480, 512, 544, 576, 608], max_size=800),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([480], max_size=800),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "train.json"),
        "val":   (root / "valid", root / "valid.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=make_nycu_transforms(image_set),
        return_masks=False,
    )
    return dataset
