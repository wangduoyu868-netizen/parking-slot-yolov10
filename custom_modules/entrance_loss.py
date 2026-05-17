"""Entrance-line losses for YOLO-style PS2.0 parking-slot detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class EntranceLineLoss(nn.Module):
    def __init__(self, w_line=4.0, w_body=1.0, w_type=0.5):
        super().__init__()
        self.w_line = w_line
        self.w_body = w_body
        self.w_type = w_type

    def forward(self, pred_extra, target_extra, fg_mask):
        if fg_mask.sum() == 0:
            return pred_extra.sum() * 0.0

        pred = pred_extra[fg_mask]
        target = target_extra[fg_mask]

        pred_line = pred[:, :4].sigmoid()
        target_line = target[:, :4]
        direct = F.smooth_l1_loss(pred_line, target_line, reduction="none").sum(1)
        swapped_target = torch.cat([target_line[:, 2:4], target_line[:, 0:2]], dim=1)
        swapped = F.smooth_l1_loss(pred_line, swapped_target, reduction="none").sum(1)
        loss_line = torch.minimum(direct, swapped).mean()

        pred_body = pred[:, 4:6].tanh()
        pred_body = pred_body / pred_body.norm(dim=1, keepdim=True).clamp(min=1e-8)
        target_body = target[:, 4:6]
        loss_body = (1.0 - (pred_body * target_body).sum(dim=1)).mean()

        target_type = target[:, 6].long().clamp(0, 2)
        loss_type = F.cross_entropy(pred[:, 6:9], target_type)

        return self.w_line * loss_line + self.w_body * loss_body + self.w_type * loss_type


class v8EntranceDetectionLoss(v8DetectionLoss):
    """Extend YOLO detection loss with true entrance-line supervision."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.entrance_loss = EntranceLineLoss()

    def loss(self, preds, batch):
        batch_size = preds["boxes"].shape[0]
        (fg_mask, target_gt_idx, _, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)

        loss = torch.zeros(4, device=self.device)
        loss[:3] = det_loss

        if "entrance" in preds and "entrance" in batch and fg_mask.sum() > 0:
            pred_extra = preds["entrance"].permute(0, 2, 1).contiguous()
            gt_entrance = batch["entrance"].to(self.device)
            if gt_entrance.dim() == 2 and gt_entrance.shape[-1] >= 8:
                batch_idx = gt_entrance[:, 0].long()
                batch_counts = torch.bincount(batch_idx, minlength=batch_size).tolist()
                extra_list = []
                offset = 0
                for i in range(batch_size):
                    n = batch_counts[i] if i < len(batch_counts) else 0
                    extra_list.append(gt_entrance[offset:offset + n, 1:8])
                    offset += n

                max_n = max((e.shape[0] for e in extra_list), default=0)
                if max_n > 0:
                    padded = torch.zeros(batch_size, max_n, 7, device=self.device)
                    for i, e in enumerate(extra_list):
                        if e.shape[0] > 0:
                            padded[i, :e.shape[0]] = e
                    clamped_idx = target_gt_idx.clamp(0, max_n - 1)
                    target_extra = padded.gather(1, clamped_idx.unsqueeze(-1).expand(-1, -1, 7))
                    loss[3] = self.entrance_loss(pred_extra, target_extra, fg_mask)

        return loss * batch_size, loss.detach()
