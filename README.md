# DETR 系列 — 數字偵測專用版 (NYCU HW2)

本專案基於 Facebook Research 的 [DETR](https://github.com/facebookresearch/detr) 修改，
針對 **NYCU HW2 數字偵測資料集**進行客製化，並逐步升級至 **DAB-DETR** 與 **DINO-DETR**。
支援 10 類數字（0～9）的物件偵測訓練、評估與推理。

---


## 方法演進

### Baseline DETR（原始客製化）

原版 DETR 針對本任務的修改：

| 檔案 | 修改內容 |
|---|---|
| `datasets/nycu.py` | 自訂資料集 + ColorJitter / RandomResize augmentation |
| `datasets/__init__.py` | 新增 `'nycu'` dataset 路由 |
| `models/detr.py` | `num_classes = 11`（1–10 對應 digit 0–9，11 = background） |
| `engine.py` | TensorBoard per-step/per-epoch logging、`visualize_val_samples` |
| `main.py` | `--exp_name`、自動 output dir、`best.pth`、`metrics_curve.png`、`params.txt` |
| `inference.py` | 獨立推理腳本，支援 NMS/score threshold 掃描 |

**類別對應**

```
model output index    意義
──────────────────────────────
      0               未使用（COCO category ID 從 1 開始）
    1 ~ 10            數字 "0" ~ "9"（category_id = 1 ~ 10）
     11               背景 / no-object
```

---

### DAB-DETR（Anchor-based Query）

原版 DETR query 是 256 維混合向量，沒有位置先驗，需要大量 epoch 才能收斂。
DAB-DETR 改用 **4 維 anchor box** 作為 query，每層 Decoder 動態生成 sine PE。

**主要改動**

| 改動 | 說明 |
|---|---|
| Query 拆分 | `query_anchor (N,4)` 負責位置 + `query_content (N,256)` 負責類別特徵 |
| 4D sine PE | `(cy, cx, h, w)` 各 64 維，w/h 調製讓小 anchor 聚焦更小區域 |
| Iterative Refinement | 每層 Decoder 用共享 `bbox_embed` MLP 精煉 anchor，梯度不截斷 |

```python
# models/transformer.py
def gen_sineembed_for_position(pos_tensor, d_model=256):
    # pos: (N, B, 4) → (N, B, 256)
    # Layout: [cy_64d | cx_64d | h_64d | w_64d]
```

---

### DINO-DETR（Contrastive DeNoising）

Hungarian Matching 前期 supervision 稀疏，DINO 在 training 時從 GT box 加雜訊產生額外 query：

| Query 類型 | 雜訊量 | 監督目標 |
|---|---|---|
| Positive | 小（< noise_scale/2） | GT label + GT box |
| Negative | 大（≥ noise_scale/2） | no-object |

CDN query 跳過 Hungarian Matching，直接提供強監督訊號，訓練初期 loss 更快下降。

```
┌──────────────────────────────────┬──────────────┐
│  CDN region (G × 2P slots)       │  det queries │
│  [grp0-pos | grp0-neg | grp1-..] │  (K queries) │
└──────────────────────────────────┴──────────────┘
各 CDN group 互相隔離；CDN ↔ det queries 互相看不到
```


---

## 環境安裝

```bash
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install scipy pycocotools matplotlib tensorboard

# 確認 GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 資料集結構

```
data/nycu-hw2-data/
    train.json   valid.json
    train/       valid/       test/
```

---

## 訓練

output dir 自動命名為 `outputs/MMDD_<exp_name>/`。

### Baseline

```bash
python main.py \
  --dataset_file nycu --coco_path data/nycu-hw2-data \
  --exp_name baseline \
  --epochs 50 --lr_drop 40 \
  --num_queries 20 --batch_size 32 --num_workers 4 --device cuda
```

### DAB-DETR

```bash
python main.py \
  --dataset_file nycu --coco_path data/nycu-hw2-data \
  --exp_name dab_detr --use_dab \
  --epochs 50 --lr_drop 40 \
  --num_queries 20 --batch_size 16 --num_workers 4 --device cuda
```

### DINO-DETR（以 DAB-DETR checkpoint 為初始權重）

```bash
python main.py \
  --dataset_file nycu --coco_path data/nycu-hw2-data \
  --exp_name dino --use_dino \
  --cdn_groups 2 --cdn_label_noise 0.5 --cdn_box_noise 1.0 \
  --pretrain_weights outputs/0418_dab_detr/best.pth \
  --epochs 50 --lr_drop 40 \
  --num_queries 20 --batch_size 16 --num_workers 4 --device cuda
```

### 主要參數說明

| 參數 | 說明 |
|---|---|
| `--use_dab` | 啟用 DAB-DETR（anchor query + 動態 sine PE + iterative refinement） |
| `--use_dino` | 啟用 DINO-DETR（包含 DAB + Contrastive DeNoising） |
| `--cdn_groups` | CDN group 數量，越多監督越強（建議 1–4） |
| `--pretrain_weights` | 載入預訓練權重（不影響 optimizer state） |
| `--num_queries` | 每張圖最多偵測幾個物件，數字圖建議 20 |
| `--batch_size` | RTX 4090：Baseline 可設 32，DAB/DINO 建議 16 |
| `--lr` / `--lr_backbone` | 預設 1e-4 / 1e-5 |
| `--lr_drop` | 第 N epoch 後 LR 降 10 倍 |

---

## 推理（Inference）

`inference.py` 會從 checkpoint 自動偵測模型架構（Baseline / DAB / DINO），無需手動指定。

### Val 驗證（含 mAP + NMS/score threshold 掃描）

```bash
python inference.py \
  --checkpoint outputs/0418_dab_detr/best.pth \
  --test_dir   data/nycu-hw2-data/valid \
  --gt_json    data/nycu-hw2-data/valid.json \
  --output     results/dab_val.json \
  --score_thr  0.05 --nms_iou 0.5 \
  --sweep_nms_step 0.05 --sweep_score_step 0.05 \
  --vis_n 0
```

掃描結果輸出為 `results/nms_curve.png` 與 `results/score_curve.png`，用於選最佳 threshold。

### Test 推理（無 GT）

```bash
python inference.py \
  --checkpoint outputs/0418_dab_detr/best.pth \
  --test_dir   data/nycu-hw2-data/test \
  --output     results/dab_test.json \
  --score_thr  0.05 --nms_iou 0.5 \
  --vis_n 3
```

輸出 `results/dab_test.json`（COCO 格式）可直接上傳。

---

## TensorBoard

```bash
tensorboard --logdir outputs
```

| 群組 | 指標 |
|---|---|
| `train/`（per 小數 epoch） | `loss`、`loss_ce`、`loss_bbox`、`loss_giou`、`class_error`、`grad_norm`、`lr` |
| `train/`（per epoch） | `lr_transformer`、`lr_backbone` |
| `val/`（per epoch） | `mAP`、`mAP50`、`mAP75`、`mAP_small/medium/large`、`AR_maxDets*` |

---

## 輸出結構

```
outputs/0418_dab_detr/
├── best.pth             # val mAP 最高的 checkpoint
├── last.pth             # 最新 epoch 的 checkpoint
├── params.txt           # 本次訓練的完整參數
├── log.txt              # 每個 epoch 的 loss / mAP JSON
├── metrics_curve.png    # 訓練曲線（loss + mAP）
├── samples.png          # val 推論視覺化
└── runs/                # TensorBoard log
```

---

## 授權

本專案繼承原始 DETR 的 Apache 2.0 授權，詳見 [LICENSE](LICENSE)。
