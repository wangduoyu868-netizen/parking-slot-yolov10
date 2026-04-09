import math
import os
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


CLASS_NAMES = ["occupied", "vacant"]


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


class ParkingSlotPipeline:
    def __init__(self, det_model_path: str, cls_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_model = YOLO(det_model_path)

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.cls_model = models.mobilenet_v3_small(weights=weights)
        in_features = self.cls_model.classifier[3].in_features
        self.cls_model.classifier[3] = nn.Linear(in_features, 2)
        self.cls_model.load_state_dict(torch.load(cls_model_path, map_location=self.device))
        self.cls_model = self.cls_model.to(self.device)
        self.cls_model.eval()

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
        results = self.det_model(img, conf=det_conf, verbose=False)[0]
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

    def classify_patch(self, patch):
        x = self.cls_transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cls_model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            return CLASS_NAMES[pred_idx], float(probs[pred_idx])

    def infer_image(self, image_rgb: np.ndarray, p: PipelineParams):
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
            state, conf = self.classify_patch(patch)

            if state == "occupied":
                occupied += 1
            else:
                vacant += 1

            color = (0, 0, 255) if state == "occupied" else (0, 255, 0)
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            self.draw_quad(vis, quad, color, 2)

            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)
            cv2.putText(
                vis,
                f"{state}:{conf:.2f}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        info = (
            f"检测到 marking points: {len(points)}\n"
            f"恢复 slot lines: {len(pred_lines)}\n"
            f"occupied: {occupied}, vacant: {vacant}"
        )
        return vis_rgb, info


pipeline = None
pipeline_paths = {"det": None, "cls": None}


def run_inference(
    image,
    det_model_path,
    cls_model_path,
    det_conf,
    line_dist_thresh,
    max_lines,
    short_min,
    short_max,
    long_min,
    long_max,
):
    global pipeline, pipeline_paths
    if image is None:
        raise gr.Error("请先上传图片。")
    if not det_model_path or not cls_model_path:
        raise gr.Error("请填写检测模型和分类模型路径。")

    if (
        pipeline is None
        or pipeline_paths["det"] != det_model_path
        or pipeline_paths["cls"] != cls_model_path
    ):
        pipeline = ParkingSlotPipeline(det_model_path, cls_model_path)
        pipeline_paths = {"det": det_model_path, "cls": cls_model_path}

    params = PipelineParams(
        det_conf=det_conf,
        line_dist_thresh=int(line_dist_thresh),
        max_lines=int(max_lines),
        short_min=int(short_min),
        short_max=int(short_max),
        long_min=int(long_min),
        long_max=int(long_max),
    )
    return pipeline.infer_image(image, params)


with gr.Blocks(title="Parking Slot Visualizer") as demo:
    gr.Markdown("## 停车位检测与占用识别可视化")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="输入图片")
        image_output = gr.Image(type="numpy", label="输出结果")

    with gr.Row():
        det_model_path = gr.Textbox(
            label="检测模型路径 (YOLO .pt)",
            value=r"E:\parking_yolov10_runs\yolov10s_baseline\weights\best.pt",
        )
        cls_model_path = gr.Textbox(
            label="分类模型路径 (.pth)",
            value=r"E:\parking_slot_cls_runs\best_mobilenetv3_small.pth",
        )

    with gr.Row():
        det_conf = gr.Slider(0.1, 0.95, value=0.52, step=0.01, label="DET_CONF")
        line_dist_thresh = gr.Slider(5, 40, value=15, step=1, label="LINE_DIST_THRESH")
        max_lines = gr.Slider(1, 4, value=2, step=1, label="MAX_LINES")

    with gr.Row():
        short_min = gr.Slider(50, 300, value=140, step=1, label="SHORT_MIN")
        short_max = gr.Slider(50, 300, value=220, step=1, label="SHORT_MAX")
        long_min = gr.Slider(200, 500, value=320, step=1, label="LONG_MIN")
        long_max = gr.Slider(200, 500, value=390, step=1, label="LONG_MAX")

    run_btn = gr.Button("开始推理", variant="primary")
    text_output = gr.Textbox(label="统计信息", lines=4)

    run_btn.click(
        fn=run_inference,
        inputs=[
            image_input,
            det_model_path,
            cls_model_path,
            det_conf,
            line_dist_thresh,
            max_lines,
            short_min,
            short_max,
            long_min,
            long_max,
        ],
        outputs=[image_output, text_output],
    )


if __name__ == "__main__":
    # Avoid localhost calls being routed via proxy on some Windows setups.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=True)
