"""DirectionDetect: 扩展 v10Detect，每个检测额外输出方向向量 (dx, dy)。

参考 ultralytics OBB 头的扩展模式，添加 cv4 分支预测 2 维方向。
方向向量从 GT 标注的 dir_x, dir_y 计算得到（单位向量）。
"""
import copy
import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import v10Detect
from ultralytics.nn.modules.conv import Conv


class DirectionDetect(v10Detect):
    """YOLOv10 检测头 + 方向向量预测。

    每个 anchor 预测：
    - box: 4 * reg_max (DFL)
    - cls: nc
    - direction: ne (默认 2，即 dx, dy)
    """

    def __init__(self, nc: int = 1, ne: int = 2, ch: tuple = ()):
        """初始化。

        Args:
            nc: 类别数
            ne: 额外参数数（方向向量维度，2 = dx, dy）
            ch: 各尺度特征通道数
        """
        super().__init__(nc, ch)
        self.ne = ne

        # 方向预测头：每尺度一个
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
        # end2end one2one 分支
        self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3, dir_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, dir_head=self.one2one_cv4)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        dir_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        """前向传播头部：box + cls + direction。"""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        result = dict(boxes=boxes, scores=scores, feats=x)

        if dir_head is not None:
            direction = torch.cat(
                [dir_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], dim=-1
            )
            result["direction"] = direction

        return result

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码 box + cls + direction。"""
        dbox = self._get_decode_boxes(x)
        preds = torch.cat((dbox, x["scores"].sigmoid()), 1)
        if "direction" in x:
            # 用 tanh 将方向限制在 [-1, 1]
            direction = x["direction"].tanh()
            preds = torch.cat((preds, direction), 1)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理：输出 [x1, y1, x2, y2, max_cls_prob, cls_idx, dx, dy]。"""
        # preds: (batch, anchors, 4 + nc + ne)
        total_extra = 4 + self.nc + self.ne
        boxes, scores, direction = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        direction = direction.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, direction], dim=-1)

    def fuse(self) -> None:
        """移除 one2many 头，优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = None
