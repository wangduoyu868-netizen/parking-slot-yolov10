"""
PS2.0 marking points + CNR bounding boxes in one Gradio app (port 7861).
Classifier: ImageFolder order class 0 = occupied, 1 = vacant (folder names occupied/, vacant/).

运行: python scripts/app_gradio_ps20_cnr.py
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


# 与 torchvision.datasets.ImageFolder 一致：子目录按字母序 occupied < vacant → 类别 0=occupied, 1=vacant
CLS_LAYOUT_IMAGEFOLDER = ("occupied", "vacant")
CLS_LAYOUT_SWAPPED = ("vacant", "occupied")


@dataclass
class PipelineParams:
    det_conf: float = 0.52
    line_dist_thresh: int = 15
    max_lines: int = 2
    short_min: int = 140
    short_max: int = 220
    long_min: int = 320
    long_max: int = 390
    patch_w: int = 256
    patch_h: int = 256
    depth_ratio: float = 1.35
    depth_min: int = 120
    depth_max: int = 280


@dataclass
class CnRParams:
    det_conf: float = 0.25
    box_pad_ratio: float = 0.08
    small_mult: float = 0.90
    large_mult: float = 1.12
    fallback_small_ratio_of_min_side: float = 0.10
    fallback_large_ratio_of_min_side: float = 0.18


def _ps20_line_size_en(dist: float, p: PipelineParams) -> str:
    if p.short_min <= dist <= p.short_max:
        return "short"
    if p.long_min <= dist <= p.long_max:
        return "long"
    return "other"


def _build_cls_mobilenet(device: torch.device, state_path: str) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    m = models.mobilenet_v3_small(weights=weights)
    inf = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(inf, 2)
    m.load_state_dict(torch.load(state_path, map_location=device))
    return m.to(device).eval()


class ParkingSlotPipeline:
    def __init__(
        self,
        cls_model_path_ps20: str,
        cls_model_path_cnr: str,
        det_ps20_path: Optional[str] = None,
        det_cnr_path: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_ps20: Optional[YOLO] = YOLO(det_ps20_path) if det_ps20_path else None
        self.det_cnr: Optional[YOLO] = YOLO(det_cnr_path) if det_cnr_path else None

        self.cls_ps20 = _build_cls_mobilenet(self.device, cls_model_path_ps20)
        self.cls_cnr = _build_cls_mobilenet(self.device, cls_model_path_cnr)

        self.cls_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def point_line_distance(px, py, x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        denom = math.sqrt(a * a + b * b)
        if denom < 1e-6:
            return 1e9
        return abs(a * px + b * py + c) / denom

    def is_valid_slot_length(self, dist, p: PipelineParams):
        return (p.short_min <= dist <= p.short_max) or (p.long_min <= dist <= p.long_max)

    def find_best_line_group(self, points, remaining_indices, dist_thresh):
        best_group = []
        if len(remaining_indices) < 2:
            return best_group

        for i in range(len(remaining_indices)):
            for j in range(i + 1, len(remaining_indices)):
                idx1 = remaining_indices[i]
                idx2 = remaining_indices[j]

                x1, y1, _ = points[idx1]
                x2, y2, _ = points[idx2]
                if math.hypot(x2 - x1, y2 - y1) < 5:
                    continue

                group = []
                for idx in remaining_indices:
                    px, py, _ = points[idx]
                    d = self.point_line_distance(px, py, x1, y1, x2, y2)
                    if d < dist_thresh:
                        group.append(idx)

                if len(group) > len(best_group):
                    best_group = group
        return best_group

    @staticmethod
    def sort_points_along_line(points, group_indices):
        coords = np.array([[points[idx][0], points[idx][1]] for idx in group_indices], dtype=np.float32)
        center = coords.mean(axis=0)
        coords_centered = coords - center

        cov = np.cov(coords_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        main_dir = eigvecs[:, np.argmax(eigvals)]
        projections = coords_centered @ main_dir
        sorted_pairs = sorted(zip(group_indices, projections), key=lambda x: x[1])
        return [idx for idx, _ in sorted_pairs]

    @staticmethod
    def draw_quad(img, quad, color, thickness=2):
        quad_int = quad.astype(int)
        for i in range(4):
            p1 = tuple(quad_int[i])
            p2 = tuple(quad_int[(i + 1) % 4])
            cv2.line(img, p1, p2, color, thickness)

    def detect_marking_points(self, img, det_conf):
        if self.det_ps20 is None:
            raise RuntimeError("未加载 PS2.0 检测模型。")
        results = self.det_ps20(img, conf=det_conf, verbose=False)[0]
        points = []
        if results.boxes is None or len(results.boxes) == 0:
            return points

        xywh = results.boxes.xywh.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for box, conf in zip(xywh, confs):
            xc, yc, _, _ = box
            points.append((float(xc), float(yc), float(conf)))
        return points

    def build_pred_slot_lines(self, points, p: PipelineParams):
        remaining = list(range(len(points)))
        line_groups = []
        for _ in range(p.max_lines):
            group = self.find_best_line_group(points, remaining, p.line_dist_thresh)
            if len(group) < 2:
                break
            line_groups.append(group)
            remaining = [idx for idx in remaining if idx not in group]

        pred_lines = []
        for group in line_groups:
            sorted_group = self.sort_points_along_line(points, group)
            for k in range(len(sorted_group) - 1):
                idx1 = sorted_group[k]
                idx2 = sorted_group[k + 1]
                x1, y1, _ = points[idx1]
                x2, y2, _ = points[idx2]
                dist = math.hypot(x2 - x1, y2 - y1)
                if self.is_valid_slot_length(dist, p):
                    pred_lines.append(((x1, y1), (x2, y2)))
        return pred_lines

    def build_slot_quad(self, p1, p2, img_w, img_h, p: PipelineParams):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        vec = p2 - p1
        line_len = np.linalg.norm(vec)
        if line_len < 1e-6:
            return None

        t = vec / line_len
        n1 = np.array([-t[1], t[0]], dtype=np.float32)
        n2 = -n1

        mid = (p1 + p2) / 2.0
        center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
        depth = np.clip(line_len * p.depth_ratio, p.depth_min, p.depth_max)

        cand1 = mid + n1 * depth
        cand2 = mid + n2 * depth
        d1 = np.linalg.norm(cand1 - center)
        d2 = np.linalg.norm(cand2 - center)
        n = n1 if d1 > d2 else n2

        # 入口边严格为 slot line 两端点，不再沿切向外扩
        a = p1.copy()
        b = p2.copy()
        c = b + n * depth
        d_pt = a + n * depth

        def clip_point(pt):
            x, y = pt
            x = max(0, min(img_w - 1, x))
            y = max(0, min(img_h - 1, y))
            return np.array([x, y], dtype=np.float32)

        quad = np.array([clip_point(a), clip_point(b), clip_point(c), clip_point(d_pt)], dtype=np.float32)
        return quad

    def warp_patch(self, img, quad, p: PipelineParams):
        dst = np.array(
            [[0, 0], [p.patch_w - 1, 0], [p.patch_w - 1, p.patch_h - 1], [0, p.patch_h - 1]],
            dtype=np.float32,
        )
        mat = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(img, mat, (p.patch_w, p.patch_h))

    def classify_patch(
        self,
        patch_bgr: np.ndarray,
        cls_layout: tuple,
        legacy_bgr_as_rgb: bool,
        cls_net: nn.Module,
    ):
        """
        legacy_bgr_as_rgb=True：不把 BGR 转成 RGB，直接交给 ToPILImage（与旧版 app_gradio 一致）。
        legacy_bgr_as_rgb=False：BGR→RGB，与 ImageFolder / CNR patch 训练一致。
        """
        arr = patch_bgr if legacy_bgr_as_rgb else cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        x = self.cls_transform(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = cls_net(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            state = cls_layout[pred_idx]
            return state, float(probs[pred_idx])

    def infer_image_ps20(
        self,
        image_rgb: np.ndarray,
        p: PipelineParams,
        cls_layout: tuple,
        legacy_bgr_ps20: bool,
    ):
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        vis = img.copy()
        h, w = img.shape[:2]

        points = self.detect_marking_points(img, p.det_conf)
        pred_lines = self.build_pred_slot_lines(points, p)

        occupied = 0
        vacant = 0
        for line in pred_lines:
            (x1, y1), (x2, y2) = line
            quad = self.build_slot_quad((x1, y1), (x2, y2), w, h, p)
            if quad is None:
                continue

            patch = self.warp_patch(img, quad, p)
            state, conf = self.classify_patch(patch, cls_layout, legacy_bgr_ps20, self.cls_ps20)
            dist = math.hypot(x2 - x1, y2 - y1)
            size_en = _ps20_line_size_en(dist, p)

            if state == "occupied":
                occupied += 1
            else:
                vacant += 1

            color = (0, 0, 255) if state == "occupied" else (0, 255, 0)
            # 车位区域：由 slot line 两端点 (x1,y1)-(x2,y2) 经 build_slot_quad 推出的四边形（与分类 warp 一致）
            self.draw_quad(vis, quad, color, 2)
            # 恢复出的车位线（检测点连线），叠画在入口一侧便于对照
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)
            label = f"{state} {conf:.2f} ({size_en})"
            cv2.putText(
                vis,
                label,
                (mx - 60, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        total = occupied + vacant
        info = (
            f"模式: PS2.0（标记点）\n"
            f"标记点数量: {len(points)}\n"
            f"恢复的车位线: {len(pred_lines)}\n"
            f"总车位数: {total}\n"
            f"空车位: {vacant}，已占用: {occupied}"
        )
        return vis_rgb, info

    @staticmethod
    def _categorize_slot_size(
        max_side: float,
        sides: List[float],
        img_min_side: int,
        cp: CnRParams,
    ) -> str:
        if len(sides) >= 3:
            med = float(np.median(np.array(sides, dtype=np.float32)))
            if med < 1e-3:
                return "medium"
            if max_side < med * cp.small_mult:
                return "small"
            if max_side > med * cp.large_mult:
                return "large"
            return "medium"
        t_s = img_min_side * cp.fallback_small_ratio_of_min_side
        t_l = img_min_side * cp.fallback_large_ratio_of_min_side
        if max_side < t_s:
            return "small"
        if max_side > t_l:
            return "large"
        return "medium"

    def infer_image_cnr(
        self,
        image_rgb: np.ndarray,
        cp: CnRParams,
        cls_layout: tuple,
        legacy_bgr_cnr: bool,
    ):
        if self.det_cnr is None:
            raise RuntimeError("未加载 CNR 检测模型。")
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        vis = img.copy()
        h, w = img.shape[:2]
        img_min_side = min(h, w)

        results = self.det_cnr(img, conf=cp.det_conf, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            info = (
                "模式: CNR（检测框）\n"
                "总车位数: 0\n"
                "空车位: 0，已占用: 0\n"
                "（未检测到车位框）"
            )
            return vis_rgb, info

        xyxy = results.boxes.xyxy.cpu().numpy()

        sides: List[float] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            sides.append(max(float(x2 - x1), float(y2 - y1)))

        occupied = vacant = 0
        size_counts = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            bw = x2 - x1
            bh = y2 - y1
            pad = cp.box_pad_ratio * max(bw, bh, 1.0)
            xi1 = int(max(0, math.floor(x1 - pad)))
            yi1 = int(max(0, math.floor(y1 - pad)))
            xi2 = int(min(w - 1, math.ceil(x2 + pad)))
            yi2 = int(min(h - 1, math.ceil(y2 + pad)))
            if xi2 <= xi1 or yi2 <= yi1:
                continue

            crop = img[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue

            state, pconf = self.classify_patch(crop, cls_layout, legacy_bgr_cnr, self.cls_cnr)
            if state == "occupied":
                occupied += 1
                occ_idx = 1
            else:
                vacant += 1
                occ_idx = 0

            cat = self._categorize_slot_size(sides[i], sides, img_min_side, cp)
            size_counts[cat][occ_idx] += 1

            color = (0, 0, 255) if state == "occupied" else (0, 255, 0)
            # 可视化用检测器原始框（原图尺度）；分类用的 pad 裁剪不单独画框，避免误以为是「缩小车位」
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            tx = int(max(0, min(w - 1, x1)))
            ty = int(max(0, min(h - 1, y1 - 8)))
            if ty < 12:
                ty = int(min(h - 1, y2 + 22))
            label = f"{state} {pconf:.2f} ({cat})"
            cv2.putText(
                vis,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        total = occupied + vacant

        def _line(bucket: str) -> str:
            v, o = size_counts[bucket]
            zh = {"small": "小", "medium": "中", "large": "大"}[bucket]
            return f"{zh}: 空{v} / 占{o}"

        info = (
            f"模式: CNR（检测框）\n"
            f"总车位数: {total}\n"
            f"空车位: {vacant}，已占用: {occupied}\n"
            f"按尺寸 — {_line('small')}；{_line('medium')}；{_line('large')}"
        )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return vis_rgb, info


pipeline = None
pipeline_paths: dict = {}


def run_inference(
    image,
    mode,
    det_model_ps20,
    det_model_cnr,
    cls_model_path_ps20,
    cls_model_path_cnr,
    det_conf_ps20,
    line_dist_thresh,
    max_lines,
    short_min,
    short_max,
    long_min,
    long_max,
    det_conf_cnr,
    box_pad_ratio,
    cnr_small_mult,
    cnr_large_mult,
    cnr_fb_small,
    cnr_fb_large,
    swap_cls_indices,
    legacy_bgr_ps20,
    legacy_bgr_cnr,
):
    global pipeline, pipeline_paths
    if image is None:
        raise gr.Error("请先上传图片。")
    p20 = (cls_model_path_ps20 or "").strip()
    cnrp = (cls_model_path_cnr or "").strip()
    if not p20 or not os.path.isfile(p20):
        raise gr.Error("请填写有效的 PS2.0 分类模型路径 (.pth)。")
    if not cnrp or not os.path.isfile(cnrp):
        raise gr.Error("请填写有效的 CNR 分类模型路径 (.pth)。")

    cls_layout = CLS_LAYOUT_SWAPPED if swap_cls_indices else CLS_LAYOUT_IMAGEFOLDER

    mode = (mode or "").strip()
    if mode.startswith("PS2.0"):
        if not det_model_ps20:
            raise gr.Error("PS2.0 模式需要填写 PS2.0 检测模型路径。")
        det_ps20 = det_model_ps20.strip()
        det_cnr = det_model_cnr.strip() if det_model_cnr else ""
    else:
        if not det_model_cnr:
            raise gr.Error("CNR 模式需要填写 CNR 检测模型路径。")
        det_cnr = det_model_cnr.strip()
        det_ps20 = det_model_ps20.strip() if det_model_ps20 else ""

    key = (mode, p20, cnrp, det_ps20, det_cnr)
    if pipeline is None or pipeline_paths.get("key") != key:
        pipeline = ParkingSlotPipeline(
            cls_model_path_ps20=p20,
            cls_model_path_cnr=cnrp,
            det_ps20_path=det_ps20 if det_ps20 else None,
            det_cnr_path=det_cnr if det_cnr else None,
        )
        pipeline_paths = {"key": key}

    if mode.startswith("PS2.0"):
        if pipeline.det_ps20 is None:
            raise gr.Error("PS2.0 检测模型加载失败。")
        params = PipelineParams(
            det_conf=det_conf_ps20,
            line_dist_thresh=int(line_dist_thresh),
            max_lines=int(max_lines),
            short_min=int(short_min),
            short_max=int(short_max),
            long_min=int(long_min),
            long_max=int(long_max),
        )
        return pipeline.infer_image_ps20(image, params, cls_layout, bool(legacy_bgr_ps20))

    if pipeline.det_cnr is None:
        raise gr.Error("CNR 检测模型加载失败。")
    cp = CnRParams(
        det_conf=float(det_conf_cnr),
        box_pad_ratio=float(box_pad_ratio),
        small_mult=float(cnr_small_mult),
        large_mult=float(cnr_large_mult),
        fallback_small_ratio_of_min_side=float(cnr_fb_small),
        fallback_large_ratio_of_min_side=float(cnr_fb_large),
    )
    return pipeline.infer_image_cnr(image, cp, cls_layout, bool(legacy_bgr_cnr))


with gr.Blocks(title="停车位可视化 PS2.0 + CNR") as demo:
    gr.Markdown(
        "## 停车位检测与占用（PS2.0 / CNR）\n"
        "- **输出图像上的文字**为英文：`vacant` / `occupied`，尺寸 `small|medium|large`（CNR）或 `short|long|other`（PS2.0）。\n"
        "- **PS2.0 绘图**：由恢复的 **slot line 两端点** 经 `build_slot_quad` 推出车位四边形并描边，再叠画粗线标出该车位线。\n"
        "- 分类器：**类别 0 = occupied，1 = vacant**。若整体反了请勾选「交换类别」。\n"
        "- **两套权重**：PS2.0 默认 `best_mobilenetv3_small.pth`（与 `app_gradio.py` 一致）；CNR 默认 CNR-EXT 150 patch 训练权重。**Legacy BGR** 分两档：PS2.0 默认勾选（旧训练）；CNR 默认不勾选（RGB 训练）。\n"
        "- 自裁 CNR 小图微调仍可用 `crop_cnr_patches_for_occ_dataset.py` + `finetune_occ_classifier_cnr.py`。"
    )

    mode = gr.Radio(
        choices=["PS2.0（标记点）", "CNR（检测框）"],
        value="PS2.0（标记点）",
        label="推理模式",
    )

    with gr.Row():
        image_input = gr.Image(type="numpy", label="输入图片")
        image_output = gr.Image(type="numpy", label="输出结果（图上英文标注）")

    with gr.Row():
        det_model_ps20 = gr.Textbox(
            label="PS2.0 检测模型 (.pt)",
            value=r"E:\parking_yolov10_runs\yolov10s_baseline\weights\best.pt",
        )
        det_model_cnr = gr.Textbox(
            label="CNR 车位检测模型 (.pt)",
            value=r"E:\Programs\download\ps2.0\parking-slot-yolov10\runs\detect\train\weights\best.pt",
        )

    cls_model_path_ps20 = gr.Textbox(
        label="PS2.0 占用分类 (.pth，透视 ROI)",
        value=r"E:\parking_slot_cls_runs\best_mobilenetv3_small.pth",
    )
    cls_model_path_cnr = gr.Textbox(
        label="CNR 占用分类 (.pth，检测框裁剪)",
        value=r"E:\parking_slot_cls_runs_cnr_ext_p150\best_mobilenetv3_cnr_ext_p150.pth",
    )

    with gr.Row():
        swap_cls_indices = gr.Checkbox(
            label="交换类别索引（空/占整体反了时勾选）",
            value=False,
        )
        legacy_bgr_ps20 = gr.Checkbox(
            label="PS2.0：Legacy BGR（与 app_gradio.py 一致，默认开）",
            value=True,
        )
        legacy_bgr_cnr = gr.Checkbox(
            label="CNR：Legacy BGR（EXT patch 训练请关，默认关）",
            value=False,
        )

    gr.Markdown("### PS2.0 参数（仅标记点模式）")
    with gr.Row():
        det_conf_ps20 = gr.Slider(0.1, 0.95, value=0.52, step=0.01, label="PS2.0 DET_CONF")
        line_dist_thresh = gr.Slider(5, 40, value=15, step=1, label="LINE_DIST_THRESH")
        max_lines = gr.Slider(1, 4, value=2, step=1, label="MAX_LINES")

    with gr.Row():
        short_min = gr.Slider(50, 300, value=140, step=1, label="SHORT_MIN")
        short_max = gr.Slider(50, 300, value=220, step=1, label="SHORT_MAX")
        long_min = gr.Slider(200, 500, value=320, step=1, label="LONG_MIN")
        long_max = gr.Slider(200, 500, value=390, step=1, label="LONG_MAX")

    gr.Markdown("### CNR 参数（仅检测框模式）")
    with gr.Row():
        det_conf_cnr = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="CNR 检测置信度")
        box_pad_ratio = gr.Slider(0.0, 0.3, value=0.08, step=0.01, label="分类 ROI 外扩（×框最大边）")

    with gr.Row():
        cnr_small_mult = gr.Slider(0.5, 0.99, value=0.90, step=0.01, label="尺寸「小」: 长边 < 中位数×")
        cnr_large_mult = gr.Slider(1.01, 1.5, value=1.12, step=0.01, label="尺寸「大」: 长边 > 中位数×")
        cnr_fb_small = gr.Slider(0.04, 0.2, value=0.10, step=0.01, label="框少时「小」: < 短边×")
        cnr_fb_large = gr.Slider(0.1, 0.35, value=0.18, step=0.01, label="框少时「大」: > 短边×")

    run_btn = gr.Button("开始推理", variant="primary")
    text_output = gr.Textbox(label="统计信息（中文）", lines=6)

    run_btn.click(
        fn=run_inference,
        inputs=[
            image_input,
            mode,
            det_model_ps20,
            det_model_cnr,
            cls_model_path_ps20,
            cls_model_path_cnr,
            det_conf_ps20,
            line_dist_thresh,
            max_lines,
            short_min,
            short_max,
            long_min,
            long_max,
            det_conf_cnr,
            box_pad_ratio,
            cnr_small_mult,
            cnr_large_mult,
            cnr_fb_small,
            cnr_fb_large,
            swap_cls_indices,
            legacy_bgr_ps20,
            legacy_bgr_cnr,
        ],
        outputs=[image_output, text_output],
    )


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True, inbrowser=True)
