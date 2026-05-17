"""Loss for GNN Slot Detector: MSE(point) + Angular(dir) + BCE(edge)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNSlotLoss(nn.Module):
    """Combined point detection + edge pairing loss.

    point_loss: MSE with gradient masking (channels 0-3: conf, shape, offset)
    dir_loss: angular loss 1-cos(pred-target) (channels 4-5: cos, sin)
    edge_loss: BCE on pairwise edge scores
    """

    def __init__(self, w_point=1.0, w_edge=2.0, w_dir=2.0):
        super().__init__()
        self.w_point = w_point
        self.w_edge = w_edge
        self.w_dir = w_dir

    def point_loss(self, pred, target, mask):
        """MSE loss for channels 0-3 (conf, shape, offset_x, offset_y).

        pred: (B, 6, 16, 16)
        target: (B, 6, 16, 16)
        mask: (B, 6, 16, 16)
        """
        diff = (pred[:, :4] - target[:, :4]) ** 2
        return (diff * mask[:, :4]).sum() / mask[:, :4].sum().clamp(min=1)

    def dir_loss(self, pred, target, mask):
        """Angular loss for channels 4-5 (cos, sin).

        Uses 1 - cos(theta_pred - theta_target) = 1 - (cos_p*cos_t + sin_p*sin_t).
        Respects angular periodicity, bounded [0, 2].
        Predicted directions are L2-normalized to unit vectors.
        """
        cos_pred, sin_pred = pred[:, 4:5], pred[:, 5:6]
        cos_tgt, sin_tgt = target[:, 4:5], target[:, 5:6]
        # Normalize predicted direction to unit vector
        norm = (cos_pred ** 2 + sin_pred ** 2).sqrt().clamp(min=1e-8)
        cos_pred = cos_pred / norm
        sin_pred = sin_pred / norm
        cos_diff = cos_pred * cos_tgt + sin_pred * sin_tgt
        ang_loss = 1.0 - cos_diff
        dir_mask = mask[:, 4:5]
        return (ang_loss * dir_mask).sum() / dir_mask.sum().clamp(min=1)

    def edge_loss(self, edge_pred, edge_target, npoints):
        """BCE loss for edge prediction."""
        B, _, total = edge_pred.shape
        N = int(total ** 0.5)
        target = edge_target.reshape(B, 1, N * N)

        mask = torch.zeros_like(target)
        for b in range(B):
            n = npoints[b].item()
            if n > 0:
                mask[b, :, :n * n] = 1.0

        loss = F.binary_cross_entropy(edge_pred, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def forward(self, preds, batch):
        """Compute total loss."""
        lp = self.point_loss(
            preds["point_pred"],
            batch["point_target"],
            batch["point_mask"],
        )
        ld = self.dir_loss(
            preds["point_pred"],
            batch["point_target"],
            batch["point_mask"],
        )

        le = self.edge_loss(
            preds["edge_pred"],
            batch["edge_target"],
            batch["npoints"],
        )

        total = self.w_point * lp + self.w_dir * ld + self.w_edge * le
        return total, {
            "loss_point": lp.item(),
            "loss_dir": ld.item(),
            "loss_edge": le.item(),
            "loss_total": total.item(),
        }
