# DETR — 數字偵測專用版 (NYCU HW2)

本專案基於 Facebook Research 的 [DETR (End-to-End Object Detection with Transformers)](https://github.com/facebookresearch/detr) 修改，針對 **NYCU HW2 數字偵測資料集**進行客製化，支援 10 類數字（0～9）的物件偵測訓練與評估。

> 原始論文：[End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers)

---

## 新增數字資料集

| 檔案 | 修改內容 |
|---|---|
| `datasets/nycu.py` | 自訂數字 dataset，做輕量化 augmentation |
| `datasets/__init__.py` | 新增 `'nycu'` dataset 路由 |
| `models/detr.py` | `num_classes = 11`（category ID 1～10 對應數字 0～9，index 11 為背景） |

---

## 環境安裝

### 1. 確認 CUDA 版本對應

```bash
nvidia-smi   # 查看驅動版本與 CUDA 版本
```

| 驅動版本 | 對應安裝指令 |
|---|---|
| CUDA 12.4（驅動 >= 550） | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
| CUDA 12.1（驅動 >= 530） | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` |

### 2. 安裝其他依賴

```bash
pip install scipy pycocotools
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

標注範例：

```json
"categories": [
    {"id": 1, "name": "0"},
    {"id": 2, "name": "1"},
    ...
    {"id": 10, "name": "9"}
]
```

---

## 訓練

### 方案一：從頭訓練

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --output_dir output/nycu_run1 \
  --epochs 50 \
  --lr_drop 40 \
  --num_queries 20 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda
```

### 方案二：從 COCO 預訓練 Fine-tune

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --output_dir output/nycu_finetune \
  --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
  --epochs 50 \
  --lr_drop 40 \
  --num_queries 20 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda
```

> **注意：** 載入 COCO 預訓練權重時，分類頭（class head）因 `num_classes` 不同會自動跳過，backbone 與 transformer 的權重正常繼承。

### 主要訓練參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--num_queries` | 20 | 每張圖最多偵測幾個物件，數字圖設 20 已充裕 |
| `--batch_size` | 32 | RTX 4090 可設 16～32 |
| `--epochs` | 50 | 訓練總 epoch 數 |
| `--lr_drop` | 40 | 第幾個 epoch 後將 learning rate 降低 10 倍 |
| `--lr` | 1e-4 | Transformer 學習率 |
| `--lr_backbone` | 1e-5 | Backbone 學習率（較小以保留預訓練特徵） |

---

## 評估（Evaluation）

使用驗證集評估訓練好的模型 mAP：

```bash
python main.py \
  --dataset_file nycu \
  --coco_path data/nycu-hw2-data \
  --resume output/nycu_finetune/checkpoint.pth \
  --eval \
  --batch_size 8 \
  --num_queries 20 \
  --device cuda
```

輸出結果包含 COCO 標準指標：AP、AP50、AP75 等。

---

## 輸出檔案

訓練過程中，`--output_dir` 資料夾會產生：

```
output/nycu_finetune/
    checkpoint.pth        # 最後一個 epoch 的權重
    log.txt               # 每個 epoch 的 loss / mAP 紀錄
```

---

## 授權

本專案繼承原始 DETR 的 Apache 2.0 授權，詳見 [LICENSE](LICENSE)。
