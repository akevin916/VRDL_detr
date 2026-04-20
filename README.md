# DETR — 數字偵測專用版 (NYCU HW2)

本專案基於 Facebook Research 的 [DETR (End-to-End Object Detection with Transformers)](https://github.com/facebookresearch/detr) 修改，針對 **NYCU HW2 數字偵測資料集**進行客製化，支援 10 類數字（0～9）的物件偵測訓練與評估。

> 原始論文：[End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers)

---

## 修改說明

| 檔案 | 修改內容 |
|---|---|
| `datasets/nycu.py` | 自訂的數字資料集包含輕量化 augmentation |
| `datasets/__init__.py` | 新增 `'nycu'` dataset 路由 |
| `models/detr.py` | `num_classes = 11`（category ID 1～10 對應數字 0～9，index 11 為背景） |
| `engine.py` | 新增 TensorBoard step/epoch logging、`visualize_val_samples` |
| `main.py` | 新增 `--exp_name`、自動 output dir、`best.pth`、`metrics_curve.png`、`samples.png`、`params.txt` |

### 類別對應關係

```
模型輸出 index    意義
─────────────────────────
     0           未使用（COCO 格式 category ID 從 1 開始）
   1 ~ 10        數字 "0" ~ "9"（category_id = 1 ~ 10）
    11           背景 / no-object
```

### 初始權重說明

| 模組 | 初始權重 |
|---|---|
| Backbone（ResNet-50） | ImageNet 預訓練 |
| Transformer（Encoder + Decoder） | 隨機初始化 |
| 分類頭 / Bbox 頭 | 隨機初始化 |

---

## 環境安裝

### 1. 確認 CUDA 版本對應

```bash
nvidia-smi   # 查看驅動版本與 CUDA 版本
```

| 驅動版本 | 安裝指令 |
|---|---|
| CUDA 12.4（驅動 >= 550） | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
| CUDA 12.1（驅動 >= 530） | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |

### 2. 安裝其他依賴

```bash
pip install scipy pycocotools matplotlib tensorboard
```

### 3. 確認 GPU 可用

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 應輸出: True  以及  GPU 型號
```

---

## 資料集結構

```
data/nycu-hw2-data/
    train.json        # 訓練集標注（COCO 格式）
    valid.json        # 驗證集標注（COCO 格式）
    train/            # 訓練圖片（.png）
    valid/            # 驗證圖片（.png）
    test/             # 測試圖片（.png，無標注）
```

---

## 訓練

output_dir 為 `output/MMDD_<exp_name>/`，。

### 基本訓練（Backbone ImageNet pretrained + Transformer 從頭訓練）

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --exp_name baseline50ep \
  --epochs 50 \
  --lr_drop 40 \
  --num_queries 20 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda
```

### DC5

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --exp_name dc5 \
  --dilation \
  --epochs 50 \
  --lr_drop 40 \
  --num_queries 20 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda
```

### 主要訓練參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--exp_name` | `'exp'` | 實驗名稱，影響 output dir 命名 |
| `--num_queries` | `100` | 每張圖最多偵測幾個物件，數字圖建議設 20 |
| `--batch_size` | `2` | 依 GPU 記憶體調整，RTX 4090 可設 16～32 |
| `--epochs` | `300` | 訓練總 epoch 數 |
| `--lr_drop` | `200` | 第幾個 epoch 後 LR 降低 10 倍 |
| `--lr` | `1e-4` | Transformer 學習率 |
| `--lr_backbone` | `1e-5` | Backbone 學習率（較小以保留 ImageNet 特徵） |

---

## 評估（Evaluation）

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --resume output/0416_run1/best.pth \
  --eval \
  --batch_size 8 \
  --num_queries 20 \
  --device cuda
```

輸出包含 COCO 標準指標：mAP、mAP50、mAP75、AR 等。

---

## TensorBoard

```bash
tensorboard --logdir output
```

記錄內容：

| 群組 | 指標 |
|---|---|
| `train/`（per step） | `loss`、`loss_ce`、`loss_bbox`、`loss_giou`、`class_error`、`grad_norm`、`lr` |
| `train/`（per epoch） | `epoch_loss`、`epoch_loss_*`、`lr_transformer`、`lr_backbone` |
| `val/`（per epoch） | `mAP`、`mAP50`、`mAP75`、`mAP_small/medium/large`、`AR_maxDets1/10/{num_queries}` |

---

## 輸出檔案

```
output/0416_run1/
├── last.pth             # 每個 epoch 覆蓋，最新一個 epoch 的完整訓練狀態
├── best.pth             # val mAP 最高時更新的完整訓練狀態
├── params.txt           # 本次訓練的所有參數
├── log.txt              # 每個 epoch 的 loss / mAP JSON 紀錄
├── metrics_curve.png    # 訓練曲線圖（上：loss，下：mAP / mAP50）
├── samples.png          # 訓練結束後用 best.pth 對 val 做推論的視覺結果
└── runs/                # TensorBoard log
    └── events.out.tfevents.*
```

> 可直接用 `--resume` 繼續訓練或做推論。

---

## 授權

本專案繼承原始 DETR 的 Apache 2.0 授權，詳見 [LICENSE](LICENSE)。
