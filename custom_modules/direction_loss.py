"""方向向量损失函数 + 扩展 v8DetectionLoss 支持方向预测。

方向损失：1 - cos_similarity，对向量长度不敏感，梯度平滑。
"""
import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss


class DirectionCosLoss(nn.Module):
    """方向向量损失：1 - cos_similarity。

    对每个正样本，计算预测方向与 GT 方向的余弦相似度损失。
    方向是单位向量，所以 cos_similarity = dot(pred, gt)。
    损失 = 1 - cos_similarity，范围 [0, 2]。
    """

    def forward(self, pred_dir, target_dir, fg_mask):
        """
        Args:
            pred_dir: (batch, anchors, 2) 预测方向向量 (已 tanh 归一化)
            target_dir: (batch, max_gt, 2) GT 方向向量
            fg_mask: (batch, anchors) 前景掩码
            target_gt_idx: (batch, anchors) 每个前景 anchor 对应的 GT 索引

        Returns:
            标量损失
        """
        if fg_mask.sum() == 0:
            return pred_dir.sum() * 0.0  # 返回 0 但保持梯度图

        # 提取前景 anchor 的预测方向
        pred = pred_dir[fg_mask]  # (n_fg, 2)

        # L2 归一化预测方向（tanh 输出可能不是单位向量）
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)

        # target_dir 已经是单位向量
        target = target_dir[fg_mask]  # (n_fg, 2)

        cos_sim = (pred_norm * target).sum(dim=1)
        loss = 1.0 - cos_sim
        return loss.mean()


class v8DirectionDetectionLoss(v8DetectionLoss):
    """扩展 v8DetectionLoss，添加方向向量损失。

    继承 box + cls + dfl 损失，额外计算 direction 损失。
    方向目标从 batch["direction"] 获取。
    """

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.direction_loss = DirectionCosLoss()
        # 从模型头获取方向维度
        m = model.model[-1]
        self.ne = getattr(m, "ne", 2)

    def loss(self, preds, batch):
        """计算 box + cls + dfl + direction 四项损失。"""
        batch_size = preds["boxes"].shape[0]

        # 基础检测损失 (box, cls, dfl)
        # get_assigned_targets_and_loss 返回:
        #   (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, det_loss_detach
        (fg_mask, target_gt_idx, _, _, _), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )

        # 方向损失
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, direction
        loss[:3] = det_loss

        if "direction" in preds and "direction" in batch and fg_mask.sum() > 0:
            pred_dir = preds["direction"].permute(0, 2, 1).contiguous()  # (batch, anchors, 2)

            # 从 batch 中获取 GT 方向并按 batch 索引重建
            gt_directions = batch["direction"].to(self.device)  # (n_gt_total, 3): [batch_idx, dx, dy]
            if gt_directions.dim() == 2 and gt_directions.shape[-1] >= 3:
                # 用 direction 张量自己的 batch_idx 列计数每张图的 GT 数
                dir_batch_idx = gt_directions[:, 0].long()
                batch_counts = torch.bincount(dir_batch_idx, minlength=batch_size).tolist()

                # 按图像分组构建 (batch, max_n_gt, 2) 的方向张量
                dir_list = []
                offset = 0
                for i in range(batch_size):
                    n = batch_counts[i] if i < len(batch_counts) else 0
                    img_dirs = gt_directions[offset:offset + n, 1:3]  # (n, 2)
                    offset += n
                    dir_list.append(img_dirs)

                # 填充到统一尺寸 (batch, max_n_gt, 2)
                max_n = max(d.shape[0] for d in dir_list) if dir_list else 0
                if max_n == 0:
                    loss_detach = loss.detach()
                    return loss * batch_size, loss_detach

                padded = torch.zeros(batch_size, max_n, 2, device=self.device)
                for i, d in enumerate(dir_list):
                    if d.shape[0] > 0:
                        padded[i, :d.shape[0]] = d

                # 用 target_gt_idx 索引 GT 方向
                # target_gt_idx: (batch, anchors) 每个 anchor 对应的 GT 索引
                clamped_idx = target_gt_idx.clamp(0, max_n - 1)
                target_dir = padded.gather(
                    1, clamped_idx.unsqueeze(-1).expand(-1, -1, 2)
                )  # (batch, anchors, 2)

                dir_loss = self.direction_loss(pred_dir, target_dir, fg_mask)
                loss[3] = dir_loss * getattr(self.hyp, "direction", 2.0)

        loss_detach = loss.detach()
        return loss * batch_size, loss_detach
