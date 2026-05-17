"""PolygonDetect: YOLOv10 detection head with ordered parking-slot polygon output."""
import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import v10Detect


class PolygonDetect(v10Detect):
    """YOLOv10 detect head plus PolygonPlus extra outputs.

    Extra channels per anchor:
      - 8 ordered polygon coordinates, normalized via sigmoid
      - 2 body direction values, normalized via tanh + cosine loss
      - 3 slot-type logits
    """

    def __init__(self, nc: int = 1, ne: int = 13, ch: tuple = ()):
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
        return dict(box_head=self.cv2, cls_head=self.cv3, poly_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, poly_head=self.one2one_cv4)

    def forward_head(self, x: list[torch.Tensor], box_head, cls_head, poly_head=None) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        result = dict(boxes=boxes, scores=scores, feats=x)
        if poly_head is not None:
            polygon = torch.cat([poly_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], dim=-1)
            result["polygon"] = polygon
        return result

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        preds = torch.cat((dbox, x["scores"].sigmoid()), 1)
        if "polygon" in x:
            raw = x["polygon"]
            poly = raw[:, :8].sigmoid()
            body = raw[:, 8:10].tanh()
            slot_type = raw[:, 10:13].softmax(1)
            preds = torch.cat((preds, poly, body, slot_type), 1)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, extra = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        extra = extra.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, extra], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None
