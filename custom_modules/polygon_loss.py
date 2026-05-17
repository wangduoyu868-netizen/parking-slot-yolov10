"""PolygonPlus losses for YOLO-style parking-slot quadrilateral detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class PolygonPlusLoss(nn.Module):
    """Ordered-corner, body-direction, and slot-type loss."""

    def __init__(self, w_poly=4.0, w_body=1.0, w_type=0.5):
        super().__init__()
        self.w_poly = w_poly
        self.w_body = w_body
        self.w_type = w_type

    def forward(self, pred_extra, target_extra, fg_mask):
        if fg_mask.sum() == 0:
            return pred_extra.sum() * 0.0

        pred = pred_extra[fg_mask]
        target = target_extra[fg_mask]

        pred_poly = pred[:, :8].sigmoid()
        target_poly = target[:, :8]
        loss_poly = F.smooth_l1_loss(pred_poly, target_poly, reduction="mean")

        pred_body = pred[:, 8:10].tanh()
        pred_body = pred_body / pred_body.norm(dim=1, keepdim=True).clamp(min=1e-8)
        target_body = target[:, 8:10]
        loss_body = 1.0 - (pred_body * target_body).sum(dim=1)
        loss_body = loss_body.mean()

        target_type = target[:, 10].long().clamp(0, 2)
        loss_type = F.cross_entropy(pred[:, 10:13], target_type)

        return self.w_poly * loss_poly + self.w_body * loss_body + self.w_type * loss_type


class v8PolygonDetectionLoss(v8DetectionLoss):
    """Extend YOLO detection loss with PolygonPlus supervision."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.polygon_loss = PolygonPlusLoss(
            w_poly=getattr(self.hyp, "polygon", 4.0),
            w_body=getattr(self.hyp, "body", 1.0),
            w_type=getattr(self.hyp, "slot_type", 0.5),
        )

    def loss(self, preds, batch):
        batch_size = preds["boxes"].shape[0]
        (fg_mask, target_gt_idx, _, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)

        loss = torch.zeros(4, device=self.device)
        loss[:3] = det_loss

        if "polygon" in preds and "polygon" in batch and fg_mask.sum() > 0:
            pred_extra = preds["polygon"].permute(0, 2, 1).contiguous()
            gt_polygon = batch["polygon"].to(self.device)
            if gt_polygon.dim() == 2 and gt_polygon.shape[-1] >= 12:
                batch_idx = gt_polygon[:, 0].long()
                batch_counts = torch.bincount(batch_idx, minlength=batch_size).tolist()
                extra_list = []
                offset = 0
                for i in range(batch_size):
                    n = batch_counts[i] if i < len(batch_counts) else 0
                    extra_list.append(gt_polygon[offset:offset + n, 1:12])
                    offset += n

                max_n = max((e.shape[0] for e in extra_list), default=0)
                if max_n > 0:
                    padded = torch.zeros(batch_size, max_n, 11, device=self.device)
                    for i, e in enumerate(extra_list):
                        if e.shape[0] > 0:
                            padded[i, :e.shape[0]] = e
                    clamped_idx = target_gt_idx.clamp(0, max_n - 1)
                    target_extra = padded.gather(1, clamped_idx.unsqueeze(-1).expand(-1, -1, 11))
                    loss[3] = self.polygon_loss(pred_extra, target_extra, fg_mask)

        return loss * batch_size, loss.detach()
