"""EntranceDetect: YOLOv10 head with true slot-entrance extra output."""
import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import v10Detect


class EntranceDetect(v10Detect):
    """YOLOv10 detect head plus PS2.0 entrance-line output.

    Extra channels:
      - x1 y1 x2 y2: normalized entrance endpoints
      - body_dx body_dy: unit body direction
      - type logits: 3 values
    """

    def __init__(self, nc: int = 1, ne: int = 9, ch: tuple = ()):
        super().__init__(nc, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
        self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, entrance_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, entrance_head=self.one2one_cv4)

    def forward_head(self, x: list[torch.Tensor], box_head, cls_head, entrance_head=None) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        result = dict(boxes=boxes, scores=scores, feats=x)
        if entrance_head is not None:
            entrance = torch.cat([entrance_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], dim=-1)
            result["entrance"] = entrance
        return result

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        preds = torch.cat((dbox, x["scores"].sigmoid()), 1)
        if "entrance" in x:
            raw = x["entrance"]
            endpoints = raw[:, :4].sigmoid()
            body = raw[:, 4:6].tanh()
            slot_type = raw[:, 6:9].softmax(1)
            preds = torch.cat((preds, endpoints, body, slot_type), 1)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, extra = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        extra = extra.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, extra], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None
