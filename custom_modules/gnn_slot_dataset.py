"""Dataset for GNN Slot Detector.

Loads YOLO-format labels (.txt) + direction files (.dir),
generates:
  - 6-channel point target (conf, shape, offset_x, offset_y, cos, sin)
  - edge pairing target (which point pairs form entrance lines)
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def build_entrance_lines(points, img_w, img_h,
                         max_pair_dist=250, min_facing_proj=0.3):
    """Unified entrance line detection (perp + facing).

    Args:
        points: list of (xc_px, yc_px, dx, dy)
        Returns: list of (i, j) index pairs
    """
    pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            xi, yi, dxi, dyi = points[i]
            xj, yj, dxj, dyj = points[j]
            dist = math.hypot(xj - xi, yj - yi)
            if dist < 5 or dist > max_pair_dist:
                continue

            vx, vy = xj - xi, yj - yi
            vn = math.hypot(vx, vy)
            if vn < 1e-6:
                continue
            vnx, vny = vx / vn, vy / vn

            cross_i = abs(dxi * vny - dyi * vnx)
            cross_j = abs(dxj * vny - dyj * vnx)
            proj_i = dxi * vnx + dyi * vny
            proj_j = dxj * (-vnx) + dyj * (-vny)
            dot_dir = dxi * dxj + dyi * dyj

            is_perp = cross_i > 0.8 and cross_j > 0.8 and dot_dir > 0.5
            is_facing = proj_i > min_facing_proj and proj_j > min_facing_proj

            if is_perp or is_facing:
                pairs.append((i, j))
    return pairs


class GNNSlotDataset(Dataset):
    """Dataset for point detection + GNN edge pairing.

    Directory structure:
        root/
          images/{split}/*.jpg
          labels/{split}/*.txt   (YOLO: class xc yc w h)
          labels/{split}/*.dir   (dx dy per line)
    """

    FEATURE_MAP_SIZE = 16
    NUM_CHANNELS = 6  # conf, shape, offset_x, offset_y, cos, sin

    def __init__(self, root, split="train", imgsz=512, max_points=30):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.max_points = max_points

        img_dir = self.root / "images" / split
        self.samples = []
        for p in sorted(img_dir.glob("*.jpg")):
            label = self.root / "labels" / split / f"{p.stem}.txt"
            dir_file = self.root / "labels" / split / f"{p.stem}.dir"
            if label.exists() and dir_file.exists():
                self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def _load_points(self, img_path):
        """Load marking points from label + dir files.

        Returns:
            points: list of (xc, yc, dx, dy) in pixel coords
        """
        stem = img_path.stem
        label_path = self.root / "labels" / self.split / f"{stem}.txt"
        dir_path = self.root / "labels" / self.split / f"{stem}.dir"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = img.shape[:2]

        points = []
        with open(label_path) as fl, open(dir_path) as fd:
            for ll, dl in zip(fl, fd):
                parts = ll.strip().split()
                dp = dl.strip().split()
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                dx = float(dp[0])
                dy = float(dp[1])
                points.append((xc, yc, dx, dy))
        return img, points, w, h

    def _generate_targets(self, points, img_w, img_h):
        """Generate 6-channel feature map target + edge target.

        Returns:
            point_target: (6, 16, 16) tensor
            point_mask: (6, 16, 16) tensor
            norm_points: (N, 2) normalized coords [0,1] for grid_sample
            edge_pairs: list of (i, j) ground truth pairs
        """
        S = self.FEATURE_MAP_SIZE
        point_target = torch.zeros(self.NUM_CHANNELS, S, S)
        point_mask = torch.zeros(self.NUM_CHANNELS, S, S)
        point_mask[0].fill_(1.)  # conf channel always has gradient

        norm_points = []
        for xc, yc, dx, dy in points:
            nx = xc / img_w
            ny = yc / img_h
            col = int(nx * S)
            row = int(ny * S)
            col = min(col, S - 1)
            row = min(row, S - 1)

            # Direction: normalize (dx, dy) directly to unit vector
            norm = math.hypot(dx, dy)
            if norm > 1e-8:
                cos_val = dx / norm
                sin_val = dy / norm
            else:
                cos_val, sin_val = 1.0, 0.0

            # Confidence
            point_target[0, row, col] = 1.0
            # Shape: 0 as default (will learn)
            point_target[1, row, col] = 0.0
            # Offset
            point_target[2, row, col] = nx * S - col
            point_target[3, row, col] = ny * S - row
            # Direction as cos/sin (direct from dx, dy)
            point_target[4, row, col] = cos_val
            point_target[5, row, col] = sin_val

            # Gradient mask: backprop all channels where point exists
            point_mask[1:, row, col].fill_(1.)

            norm_points.append((nx, ny))

        # Generate pairing targets
        edge_pairs = build_entrance_lines(points, img_w, img_h)

        return point_target, point_mask, norm_points, edge_pairs

    def __getitem__(self, index):
        img_path = self.samples[index]
        try:
            img, points, img_w, img_h = self._load_points(img_path)
        except Exception:
            # Fallback to a different sample on error
            return self.__getitem__((index + 1) % len(self.samples))

        # Resize to 512×512
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        point_target, point_mask, norm_points, edge_pairs = \
            self._generate_targets(points, img_w, img_h)

        # Pack norm_points into fixed-size tensor
        n = len(norm_points)
        points_tensor = torch.zeros(self.max_points, 2)
        for i, (nx, ny) in enumerate(norm_points[:self.max_points]):
            points_tensor[i] = torch.tensor([nx, ny])

        # Edge target matrix (max_points × max_points)
        edge_target = torch.zeros(self.max_points, self.max_points)
        for i, j in edge_pairs:
            if i < self.max_points and j < self.max_points:
                edge_target[i, j] = 1.0
                edge_target[j, i] = 1.0

        return {
            "image": img,
            "point_target": point_target,
            "point_mask": point_mask,
            "points": points_tensor,
            "npoints": min(n, self.max_points),
            "edge_target": edge_target,
            "im_file": str(img_path),
        }


def gnn_slot_collate_fn(batch):
    """Custom collate for GNNSlotDataset."""
    images = torch.stack([b["image"] for b in batch])
    point_targets = torch.stack([b["point_target"] for b in batch])
    point_masks = torch.stack([b["point_mask"] for b in batch])
    points = torch.stack([b["points"] for b in batch])
    npoints = torch.tensor([b["npoints"] for b in batch])
    edge_targets = torch.stack([b["edge_target"] for b in batch])
    im_files = [b["im_file"] for b in batch]

    return {
        "image": images,
        "point_target": point_targets,
        "point_mask": point_masks,
        "points": points,
        "npoints": npoints,
        "edge_target": edge_targets,
        "im_file": im_files,
    }
