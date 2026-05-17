"""Gradio demo for YOLOv10 + EntranceDetect parking-slot detection.

The app detects parking-slot entrance lines, body directions, slot geometry,
and optional occupied/vacant state with the MobileNetV3 classifier used by the
older demo.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import scripts.train_entrance_yolo  # noqa: F401,E402 - registers custom modules
from scripts.entrance_app_utils import TYPE_NAMES, build_slot_quad, direction_angle_deg, normalize_vec  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEFAULT_DET_WEIGHTS = r"E:\谷歌下载\serveroutput\wline24_ft40e_formal\final_wline24_ft40e_last.pt"
DEFAULT_CLS_WEIGHTS = r"E:\parking_slot_cls_runs\best_mobilenetv3_small.pth"
STATE_NAMES = {"vacant": "空车位", "occupied": "占用车位", "unknown": "未分类"}
STATE_LABELS = {"vacant": "vacant", "occupied": "occupied", "unknown": "unknown"}


@dataclass
class DemoParams:
    conf: float = 0.40
    max_det: int = 40
    imgsz: int = 608
    depth_ratio: float = 1.35
    long_depth_ratio: float = 0.60
    depth_min: int = 80
    depth_max: int = 420
    show_boxes: bool = False
    show_arrows: bool = True
    show_quads: bool = True
    classify_occupancy: bool = True
    patch_w: int = 256
    patch_h: int = 256


class EntranceParkingDemo:
    def __init__(self, det_weights_path: str, cls_weights_path: str | None = None):
        self.det_weights_path = det_weights_path
        self.cls_weights_path = cls_weights_path or ""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(det_weights_path)
        self.model.model.to(self.device).eval()
        self.cls_model = None
        self.cls_transform = None
        if cls_weights_path and Path(cls_weights_path).exists():
            self.load_classifier(cls_weights_path)

    def load_classifier(self, cls_weights_path: str):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        cls_model = models.mobilenet_v3_small(weights=weights)
        in_features = cls_model.classifier[3].in_features
        cls_model.classifier[3] = nn.Linear(in_features, 2)
        cls_model.load_state_dict(torch.load(cls_weights_path, map_location=self.device))
        cls_model.to(self.device).eval()
        self.cls_model = cls_model
        self.cls_weights_path = cls_weights_path
        self.cls_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def warp_patch(image_rgb: np.ndarray, quad: np.ndarray, patch_w: int, patch_h: int):
        dst = np.array(
            [[0, 0], [patch_w - 1, 0], [patch_w - 1, patch_h - 1], [0, patch_h - 1]],
            dtype=np.float32,
        )
        mat = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(image_rgb, mat, (patch_w, patch_h))

    def classify_patch(self, patch_rgb: np.ndarray):
        if self.cls_model is None or self.cls_transform is None:
            return "unknown", 0.0
        x = self.cls_transform(patch_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.cls_model(x), dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return ("occupied" if pred_idx == 0 else "vacant"), float(probs[pred_idx])

    def preprocess(self, image_rgb: np.ndarray, imgsz: int) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_rgb: np.ndarray, params: DemoParams):
        x = self.preprocess(image_rgb, params.imgsz)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.model(x)[0][0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        arr = pred.detach().cpu().numpy()
        arr = arr[arr[:, 4] >= params.conf]
        if len(arr):
            arr = arr[np.argsort(-arr[:, 4])][: params.max_det]

        h, w = image_rgb.shape[:2]
        detections = []
        for row in arr:
            x1, y1, x2, y2 = row[6:10]
            p1 = np.array([x1 * w, y1 * h], dtype=np.float32)
            p2 = np.array([x2 * w, y2 * h], dtype=np.float32)
            body = normalize_vec([row[10] * w, row[11] * h])
            type_probs = row[12:15]
            slot_type = int(np.argmax(type_probs))
            quad = build_slot_quad(
                p1,
                p2,
                body,
                depth_ratio=params.long_depth_ratio if slot_type == 1 else params.depth_ratio,
                depth_min=params.depth_min,
                depth_max=params.depth_max,
            )
            state = "unknown"
            state_conf = 0.0
            if params.classify_occupancy:
                patch = self.warp_patch(image_rgb, quad, params.patch_w, params.patch_h)
                state, state_conf = self.classify_patch(patch)
            detections.append(
                {
                    "box": row[:4].astype(np.float32),
                    "conf": float(row[4]),
                    "p1": p1,
                    "p2": p2,
                    "body": body,
                    "type": slot_type,
                    "type_conf": float(type_probs[slot_type]),
                    "quad": quad,
                    "state": state,
                    "state_conf": state_conf,
                }
            )
        return detections, infer_ms

    @staticmethod
    def draw_detection(canvas: np.ndarray, det: dict, params: DemoParams, index: int):
        state = det.get("state", "unknown")
        if state == "occupied":
            color = (220, 38, 38)
        elif state == "vacant":
            color = (22, 163, 74)
        else:
            color = (20, 184, 166) if det["type"] == 2 else (37, 99, 235)

        entrance_color = (15, 23, 42)
        arrow_color = (234, 179, 8)
        p1 = tuple(det["p1"].astype(int))
        p2 = tuple(det["p2"].astype(int))
        quad = det["quad"].astype(int)

        if params.show_quads:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [quad], color, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.17, canvas, 0.83, 0, canvas)
            cv2.polylines(canvas, [quad], True, color, 2, cv2.LINE_AA)

        cv2.line(canvas, p1, p2, entrance_color, 3, cv2.LINE_AA)

        mid = ((det["p1"] + det["p2"]) / 2.0).astype(np.int32)
        if params.show_arrows:
            end = (mid + det["body"] * 82).astype(np.int32)
            cv2.arrowedLine(canvas, tuple(mid), tuple(end), arrow_color, 3, cv2.LINE_AA, tipLength=0.25)

        if params.show_boxes:
            h, w = canvas.shape[:2]
            scale_x = w / params.imgsz
            scale_y = h / params.imgsz
            bx1, by1, bx2, by2 = det["box"]
            cv2.rectangle(
                canvas,
                (int(bx1 * scale_x), int(by1 * scale_y)),
                (int(bx2 * scale_x), int(by2 * scale_y)),
                (99, 102, 241),
                1,
                cv2.LINE_AA,
            )

        state_text = STATE_LABELS.get(state, "unknown")
        if state == "unknown":
            label = f"{index} {TYPE_NAMES.get(det['type'], 'slot')} {det['conf']:.2f} {direction_angle_deg(det['body']):.0f}deg"
        else:
            label = f"{index} {state_text} {det['state_conf']:.2f} | {TYPE_NAMES.get(det['type'], 'slot')}"
        label_pos = (int(mid[0]) + 6, int(mid[1]) - 6)
        cv2.putText(canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 23, 42), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def infer_image(self, image_rgb: np.ndarray, params: DemoParams):
        detections, infer_ms = self.predict(image_rgb, params)
        canvas = image_rgb.copy()
        type_counts = {0: 0, 1: 0, 2: 0}
        state_counts = {"vacant": 0, "occupied": 0, "unknown": 0}
        confs = []
        for idx, det in enumerate(detections, 1):
            type_counts[det["type"]] = type_counts.get(det["type"], 0) + 1
            state_counts[det.get("state", "unknown")] = state_counts.get(det.get("state", "unknown"), 0) + 1
            confs.append(det["conf"])
            self.draw_detection(canvas, det, params, idx)

        avg_conf = float(np.mean(confs)) if confs else 0.0
        fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
        info = {
            "slots": len(detections),
            "rectangular": type_counts.get(0, 0) + type_counts.get(1, 0),
            "right_angle": type_counts.get(0, 0),
            "long_entrance": type_counts.get(1, 0),
            "acute_obtuse": type_counts.get(2, 0),
            "vacant": state_counts.get("vacant", 0),
            "occupied": state_counts.get("occupied", 0),
            "unknown": state_counts.get("unknown", 0),
            "avg_conf": round(avg_conf, 4),
            "inference_ms": round(infer_ms, 3),
            "fps": round(fps, 2),
            "device": str(self.device),
        }
        return canvas, info


pipeline = None
pipeline_det_path = None
pipeline_cls_path = None


def get_pipeline(det_weights_path: str, cls_weights_path: str):
    global pipeline, pipeline_det_path, pipeline_cls_path
    if not det_weights_path:
        raise gr.Error("请填写检测模型权重路径。")
    if not Path(det_weights_path).exists():
        raise gr.Error(f"检测权重文件不存在：{det_weights_path}")

    cls_path = cls_weights_path or ""
    if pipeline is None or pipeline_det_path != det_weights_path:
        pipeline = EntranceParkingDemo(det_weights_path, cls_path)
        pipeline_det_path = det_weights_path
        pipeline_cls_path = cls_path
    elif pipeline_cls_path != cls_path:
        if cls_path and Path(cls_path).exists():
            pipeline.load_classifier(cls_path)
        else:
            pipeline.cls_model = None
            pipeline.cls_transform = None
        pipeline_cls_path = cls_path
    return pipeline


def make_params(conf, max_det, imgsz, depth_ratio, long_depth_ratio, depth_min, depth_max, show_boxes, show_arrows, show_quads, classify_occupancy):
    return DemoParams(
        conf=float(conf),
        max_det=int(max_det),
        imgsz=int(imgsz),
        depth_ratio=float(depth_ratio),
        long_depth_ratio=float(long_depth_ratio),
        depth_min=int(depth_min),
        depth_max=int(depth_max),
        show_boxes=bool(show_boxes),
        show_arrows=bool(show_arrows),
        show_quads=bool(show_quads),
        classify_occupancy=bool(classify_occupancy),
    )


def run_single(image, det_weights_path, cls_weights_path, conf, max_det, imgsz, depth_ratio, long_depth_ratio, depth_min, depth_max, show_boxes, show_arrows, show_quads, classify_occupancy):
    if image is None:
        raise gr.Error("请先上传图片。")
    params = make_params(conf, max_det, imgsz, depth_ratio, long_depth_ratio, depth_min, depth_max, show_boxes, show_arrows, show_quads, classify_occupancy)
    app = get_pipeline(det_weights_path, cls_weights_path)
    vis, info = app.infer_image(image, params)
    summary = (
        f"检测车位数: {info['slots']}\n"
        f"空车位: {info['vacant']}  |  占用车位: {info['occupied']}  |  未分类: {info['unknown']}\n"
        f"矩形车位: {info['rectangular']}（其中长入口: {info['long_entrance']}） |  锐角/钝角车位: {info['acute_obtuse']}\n"
        f"平均检测置信度: {info['avg_conf']:.4f}\n"
        f"推理耗时: {info['inference_ms']:.3f} ms  |  FPS: {info['fps']:.2f}\n"
        f"设备: {info['device']}"
    )
    return vis, summary


def run_batch(files, det_weights_path, cls_weights_path, conf, max_det, imgsz, depth_ratio, long_depth_ratio, depth_min, depth_max, show_boxes, show_arrows, show_quads, classify_occupancy):
    if not files:
        raise gr.Error("请先选择图片文件。")
    params = make_params(conf, max_det, imgsz, depth_ratio, long_depth_ratio, depth_min, depth_max, show_boxes, show_arrows, show_quads, classify_occupancy)
    app = get_pipeline(det_weights_path, cls_weights_path)
    tmp_dir = tempfile.mkdtemp(prefix="entrance_demo_")
    gallery = []
    total_slots = 0
    total_rectangular = 0
    total_angled = 0
    total_vacant = 0
    total_occupied = 0
    total_unknown = 0
    total_ms = 0.0

    for f in files:
        fpath = f.name if hasattr(f, "name") else str(f)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis, info = app.infer_image(img_rgb, params)
        total_slots += info["slots"]
        total_rectangular += info["rectangular"]
        total_angled += info["acute_obtuse"]
        total_vacant += info["vacant"]
        total_occupied += info["occupied"]
        total_unknown += info["unknown"]
        total_ms += info["inference_ms"]
        stem = Path(fpath).stem
        out_path = Path(tmp_dir) / f"{stem}_entrance_result.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        gallery.append((str(out_path), f"{stem} ({info['slots']} slots)"))

    zip_path = Path(tmp_dir) / "entrance_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path, _caption in gallery:
            zf.write(img_path, Path(img_path).name)

    n = max(len(gallery), 1)
    summary = (
        f"处理图片: {len(gallery)}\n"
        f"总车位数: {total_slots}\n"
        f"空车位: {total_vacant}  |  占用车位: {total_occupied}  |  未分类: {total_unknown}\n"
        f"矩形车位: {total_rectangular}  |  锐角/钝角车位: {total_angled}\n"
        f"平均推理耗时: {total_ms / n:.3f} ms/image"
    )
    return gallery, summary, str(zip_path)


def clear_single():
    return None, None, ""


CSS = """
.gradio-container {
  max-width: 1320px !important;
  background: #f8fafc;
}
#hero {
  padding: 22px 24px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #ffffff;
  color: #0f172a;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}
#hero h1 {
  margin: 0 0 8px 0;
  font-size: 28px;
  letter-spacing: 0;
}
#hero p {
  margin: 0;
  color: #475569;
  font-size: 15px;
}
#legend {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 12px 0 4px 0;
}
.legend-item {
  border: 1px solid #e2e8f0;
  background: #ffffff;
  border-radius: 8px;
  padding: 10px 12px;
  color: #334155;
  font-size: 14px;
}
.swatch {
  display: inline-block;
  width: 18px;
  height: 5px;
  border-radius: 2px;
  margin-right: 8px;
  vertical-align: middle;
}
.line { background: #0f172a; }
.arrow { background: #eab308; }
.vacant { background: #16a34a; }
.occupied { background: #dc2626; }
.metric-note { color: #475569; font-size: 13px; }
"""


with gr.Blocks(title="车位入口线、方向与占用状态识别系统") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>车位入口线、方向与占用状态识别系统</h1>
          <p>基于 YOLOv10 EntranceDetect 识别入口线、车身方向、车位类型，并结合 MobileNetV3 判断空车位/占用车位。</p>
        </div>
        <div id="legend">
          <div class="legend-item"><span class="swatch line"></span>深色线：入口线</div>
          <div class="legend-item"><span class="swatch arrow"></span>黄色箭头：车身方向</div>
          <div class="legend-item"><span class="swatch vacant"></span>绿色：空车位</div>
          <div class="legend-item"><span class="swatch occupied"></span>红色：占用车位</div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("单张检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="输入图片", height=460)
                    with gr.Row():
                        run_btn = gr.Button("开始检测", variant="primary", size="lg")
                        clear_btn = gr.Button("清空", variant="secondary")
                with gr.Column(scale=1):
                    image_output = gr.Image(type="numpy", label="检测结果", height=460)
                    text_output = gr.Textbox(label="统计信息", lines=7)

        with gr.Tab("批量检测"):
            batch_files = gr.File(file_count="multiple", file_types=["image"], label="批量上传图片")
            with gr.Row():
                batch_btn = gr.Button("开始批量检测", variant="primary", size="lg")
                download_output = gr.DownloadButton("下载结果 zip", variant="secondary")
            gallery_output = gr.Gallery(label="批量结果", columns=3, height=430, object_fit="contain")
            batch_text_output = gr.Textbox(label="批量统计信息", lines=6)

    with gr.Accordion("模型与参数", open=True):
        det_weights_path = gr.Textbox(label="EntranceDetect 检测权重", value=DEFAULT_DET_WEIGHTS)
        cls_weights_path = gr.Textbox(label="MobileNetV3 空/占用分类权重", value=DEFAULT_CLS_WEIGHTS)
        with gr.Row():
            conf = gr.Slider(0.05, 0.95, value=0.40, step=0.01, label="检测置信度阈值")
            max_det = gr.Slider(1, 100, value=40, step=1, label="最大检测数量")
            imgsz = gr.Dropdown([608, 640, 768], value=608, label="推理尺寸")
        with gr.Row():
            depth_ratio = gr.Slider(0.6, 2.4, value=1.35, step=0.05, label="普通/斜角车位深度比例")
            long_depth_ratio = gr.Slider(0.3, 1.2, value=0.60, step=0.05, label="长入口车位深度比例")
        with gr.Row():
            depth_min = gr.Slider(20, 300, value=80, step=5, label="最小深度 px")
            depth_max = gr.Slider(120, 800, value=420, step=10, label="最大深度 px")
        with gr.Row():
            classify_occupancy = gr.Checkbox(value=True, label="识别空/占用状态")
            show_quads = gr.Checkbox(value=True, label="显示完整车位线")
            show_arrows = gr.Checkbox(value=True, label="显示车身方向箭头")
            show_boxes = gr.Checkbox(value=False, label="显示 YOLO 框")
        gr.Markdown(
            "<span class='metric-note'>说明：分类模型为空时仍可检测入口线和方向，但空/占用状态会显示为未分类。</span>"
        )

    single_inputs = [
        image_input,
        det_weights_path,
        cls_weights_path,
        conf,
        max_det,
        imgsz,
        depth_ratio,
        long_depth_ratio,
        depth_min,
        depth_max,
        show_boxes,
        show_arrows,
        show_quads,
        classify_occupancy,
    ]
    batch_inputs = [
        batch_files,
        det_weights_path,
        cls_weights_path,
        conf,
        max_det,
        imgsz,
        depth_ratio,
        long_depth_ratio,
        depth_min,
        depth_max,
        show_boxes,
        show_arrows,
        show_quads,
        classify_occupancy,
    ]

    run_btn.click(fn=run_single, inputs=single_inputs, outputs=[image_output, text_output])
    clear_btn.click(fn=clear_single, inputs=[], outputs=[image_input, image_output, text_output])
    batch_btn.click(fn=run_batch, inputs=batch_inputs, outputs=[gallery_output, batch_text_output, download_output])


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
