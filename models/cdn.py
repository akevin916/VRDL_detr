"""
Contrastive DeNoising (CDN) for DINO-DETR.

During training, CDN adds two kinds of queries derived from GT boxes:

  Positive queries : GT box + small noise  → supervise to predict GT label + box
  Negative queries : GT box + large noise  → supervise to predict no-object

These queries have KNOWN GT assignment (no Hungarian matching needed), which:
  1. Provides extra high-quality supervision signal for every GT per image.
  2. Forces the model to distinguish nearby proposals with different labels.
  3. Dramatically reduces duplicate predictions at inference.

CDN queries are only active during training; they are stripped from model output
before standard criterion (Hungarian) is applied, so existing training code is
minimally affected.

Reference: DINO: DETR with Improved DeNoising Anchor Boxes (Zhang et al., 2022)
"""
import torch
import torch.nn.functional as F

from .transformer import inverse_sigmoid
from util import box_ops


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prepare_cdn_queries(targets, num_queries, num_classes, hidden_dim,
                        cdn_groups=1, label_noise_ratio=0.5, box_noise_scale=1.0):
    """Build CDN content queries, anchor boxes and the self-attention mask.

    Layout per image  (G = cdn_groups,  P = max GT count in the batch):

        ┌──────────────────────────────────────────┬───────────────┐
        │     CDN region  (G × 2P slots)           │  det queries  │
        │  [grp0-pos | grp0-neg | grp1-pos | ...]  │  (num_queries)│
        └──────────────────────────────────────────┴───────────────┘

    Attention rules (self-attention inside decoder):
      · Each CDN group sees ONLY itself   → groups are isolated from each other.
      · CDN queries CANNOT see det queries.
      · Det queries CANNOT see CDN queries.

    Args:
        targets          : list[dict] with 'labels' (n,) and 'boxes' (n,4) cxcywh [0,1]
        num_queries      : number of regular detection queries (K)
        num_classes      : number of foreground classes (background = num_classes)
        hidden_dim       : transformer hidden dimension (d)
        cdn_groups       : number of denoising groups G (more groups = stronger signal)
        label_noise_ratio: fraction of GT labels to randomly flip in positive queries
        box_noise_scale  : controls noise magnitude (positive < scale/2, negative ≥ scale/2)

    Returns:
        cdn_tgt       : (bs, cdn_size, d)   zero-init content (model learns from positional embedding)
        cdn_refpoints : (bs, cdn_size, 4)   pre-sigmoid anchors = inverse_sigmoid(noised GT boxes)
        attn_mask     : (cdn_size + num_queries, cdn_size + num_queries)  bool, True = blocked
        cdn_meta      : dict with bookkeeping info
        All four are None if the batch contains no GT boxes.
    """
    device  = targets[0]['labels'].device
    bs      = len(targets)
    gt_nums = [len(t['labels']) for t in targets]
    max_gt  = max(gt_nums) if any(n > 0 for n in gt_nums) else 0

    if max_gt == 0:
        return None, None, None, None

    # Each CDN group has max_gt positive slots + max_gt negative slots
    group_size = 2 * max_gt
    cdn_size   = cdn_groups * group_size
    total_q    = cdn_size + num_queries

    cdn_tgt       = torch.zeros(bs, cdn_size, hidden_dim, device=device)
    cdn_refpoints = torch.zeros(bs, cdn_size, 4,          device=device)

    for b, (t, n) in enumerate(zip(targets, gt_nums)):
        if n == 0:
            continue

        boxes  = t['boxes']    # (n, 4) cxcywh, values in [0, 1]
        labels = t['labels']   # (n,)

        for g in range(cdn_groups):
            pos_start = g * group_size          # start of positive block for group g
            neg_start = pos_start + max_gt      # start of negative block for group g

            # ── positive queries: small noise, possibly label-flipped ──────
            if label_noise_ratio > 0 and n > 0:
                flip  = torch.rand(n, device=device) < label_noise_ratio
                rand_labels = torch.randint(0, num_classes, (n,), device=device)
                noised_labels = torch.where(flip, rand_labels, labels)  # noqa: kept in cdn_tgt slot
                # We don't embed labels here; they are embedded via class_embed during forward.
                # cdn_tgt stays zero; the LABEL supervision in compute_cdn_loss uses noised_labels.
                # Store noised labels in cdn_tgt channel 0 as a float marker (recovered in loss).
                cdn_tgt[b, pos_start:pos_start + n, 0] = noised_labels.float()
            else:
                cdn_tgt[b, pos_start:pos_start + n, 0] = labels.float()

            pos_boxes = _add_box_noise(boxes, box_noise_scale, large=False)
            neg_boxes = _add_box_noise(boxes, box_noise_scale, large=True)

            cdn_refpoints[b, pos_start:pos_start + n] = inverse_sigmoid(pos_boxes)
            cdn_refpoints[b, neg_start:neg_start + n] = inverse_sigmoid(neg_boxes)

    # ── attention mask ─────────────────────────────────────────────────────
    attn_mask = torch.zeros(total_q, total_q, dtype=torch.bool, device=device)

    for g in range(cdn_groups):
        s = g * group_size
        e = s + group_size
        if s > 0:
            attn_mask[s:e, :s]         = True  # can't see earlier CDN groups
        if e < cdn_size:
            attn_mask[s:e, e:cdn_size] = True  # can't see later CDN groups
        attn_mask[s:e, cdn_size:]      = True  # can't see det queries

    attn_mask[cdn_size:, :cdn_size] = True     # det can't see any CDN query

    cdn_meta = dict(
        padding     = max_gt,
        groups      = cdn_groups,
        cdn_size    = cdn_size,
        num_queries = num_queries,
        gt_nums     = gt_nums,
    )

    return cdn_tgt, cdn_refpoints, attn_mask, cdn_meta


def compute_cdn_loss(cdn_out, targets, num_classes, eos_coef, cdn_meta,
                     bbox_loss_coef=5.0, giou_loss_coef=2.0):
    """Compute denoising losses via DIRECT (non-Hungarian) GT assignment.

    Positive slots → GT label + GT box.
    Negative slots → no-object label (no box loss).

    Args:
        cdn_out        : dict with 'pred_logits' (bs, cdn_size, C+1),
                                   'pred_boxes'  (bs, cdn_size, 4),
                           optionally 'aux_outputs' list of same.
        targets        : same list[dict] used in main criterion
        num_classes    : C (background index = C)
        eos_coef       : weight for no-object class in cross-entropy
        cdn_meta       : returned by prepare_cdn_queries
        bbox_loss_coef : weight for L1 box loss
        giou_loss_coef : weight for GIoU loss

    Returns:
        dict of scalar tensors, keys prefixed with 'dn_'.
    """
    max_gt     = cdn_meta['padding']
    groups     = cdn_meta['groups']
    group_size = 2 * max_gt
    gt_nums    = cdn_meta['gt_nums']
    total_pos  = sum(n * groups for n in gt_nums)
    norm_n     = max(total_pos, 1)

    def _one(pred_logits, pred_boxes):
        bs, cdn_size, _ = pred_logits.shape
        device = pred_logits.device

        # ── classification ────────────────────────────────────────────────
        tgt_cls = torch.full((bs, cdn_size), num_classes,
                             dtype=torch.int64, device=device)

        for b, (t, n) in enumerate(zip(targets, gt_nums)):
            if n == 0:
                continue
            gt_labels = t['labels']
            for g in range(groups):
                pos_s = g * group_size
                # Set GT label for positive slots
                tgt_cls[b, pos_s:pos_s + n] = gt_labels
                # Negative slots keep num_classes (already set)

        weight = torch.ones(num_classes + 1, device=device)
        weight[-1] = eos_coef
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), tgt_cls, weight)

        # ── box regression (positive slots only) ─────────────────────────
        src_list, tgt_list = [], []
        for b, (t, n) in enumerate(zip(targets, gt_nums)):
            if n == 0:
                continue
            gt_boxes = t['boxes']
            for g in range(groups):
                pos_s = g * group_size
                src_list.append(pred_boxes[b, pos_s:pos_s + n])
                tgt_list.append(gt_boxes)

        if src_list:
            sb = torch.cat(src_list)
            tb = torch.cat(tgt_list)
            loss_bbox = F.l1_loss(sb, tb, reduction='sum') / norm_n
            loss_giou = (1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(sb),
                    box_ops.box_cxcywh_to_xyxy(tb),
                )
            )).sum() / norm_n
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        return loss_ce, loss_bbox * bbox_loss_coef, loss_giou * giou_loss_coef

    losses = {}
    ce, bb, gi = _one(cdn_out['pred_logits'], cdn_out['pred_boxes'])
    losses['dn_loss_ce']   = ce
    losses['dn_loss_bbox'] = bb
    losses['dn_loss_giou'] = gi

    for i, aux in enumerate(cdn_out.get('aux_outputs', [])):
        ce, bb, gi = _one(aux['pred_logits'], aux['pred_boxes'])
        losses[f'dn_loss_ce_{i}']   = ce
        losses[f'dn_loss_bbox_{i}'] = bb
        losses[f'dn_loss_giou_{i}'] = gi

    return losses


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_box_noise(boxes, scale, large: bool):
    """Add random perturbation to cxcywh boxes.

    large=False  → positive queries: |diff| < scale/2   (stays near GT)
    large=True   → negative queries: diff ∈ [scale/2, scale)  (farther from GT)
    """
    noise = torch.rand_like(boxes)
    if large:
        diff = noise * (scale / 2) + (scale / 2)
    else:
        diff = noise * (scale / 2)

    sign = (torch.randint_like(boxes, 0, 2).float() * 2 - 1)  # ±1

    cx, cy, w, h = boxes.unbind(-1)
    d = diff * sign
    new_cx = (cx + d[:, 0] * w).clamp(0, 1)
    new_cy = (cy + d[:, 1] * h).clamp(0, 1)
    new_w  = (w  * (1 + d[:, 2])).clamp(1e-3, 1)
    new_h  = (h  * (1 + d[:, 3])).clamp(1e-3, 1)

    return torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)
