"""GNN Slot Detector: DMPR-style point detection + GCN-style learned pairing.

Architecture:
  Backbone (ResNet18) → Point Head (6ch) + Descriptor Head (256ch)
  GT points / detected points → GNN (4-layer self-attn) → Edge MLP → slot pairs
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class ResNet18Backbone(nn.Module):
    """ResNet18 backbone, output 512ch feature map at 1/32 resolution."""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Don't use layer4 — we want 1/16 = 32×32 for 512 input → 32×32
        # Actually 512/16 = 32, but we want 16×16, so use layer4
        x = self.layer4(x)
        return x  # (B, 512, 16, 16) for 512×512 input


# ---------------------------------------------------------------------------
# Detection Heads
# ---------------------------------------------------------------------------
class PointHead(nn.Module):
    """Predicts 6 channels: conf, shape, offset_x, offset_y, cos(θ), sin(θ)."""

    def __init__(self, in_ch=512, hidden_ch=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 6, 1),
        )

    def forward(self, feat):
        raw = self.head(feat)
        # Apply activations
        conf = raw[:, 0:1].sigmoid()
        shape = raw[:, 1:2].sigmoid()
        offset = raw[:, 2:4].sigmoid()
        dir_raw = raw[:, 4:6]  # no activation — angular loss handles range
        return torch.cat([conf, shape, offset, dir_raw], dim=1)


class DescriptorHead(nn.Module):
    """Predicts per-cell descriptor map."""

    def __init__(self, in_ch=512, desc_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, desc_dim, 3, padding=1),
            nn.BatchNorm2d(desc_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(desc_dim, desc_dim, 3, padding=1),
            nn.BatchNorm2d(desc_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(desc_dim, desc_dim, 1),
        )

    def forward(self, feat):
        desc = self.head(feat)
        # L2 normalize
        desc = F.normalize(desc, p=2, dim=1)
        return desc


# ---------------------------------------------------------------------------
# GNN (SuperGlue-style self-attention)
# ---------------------------------------------------------------------------
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = F.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(d_model, d_model, 1) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        q, k, v = [
            l(x).view(b, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(q, k, v)
        return self.merge(x.contiguous().view(b, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(),
            nn.Conv1d(d_model * 2, d_model, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, d_model, num_layers=4, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            delta = layer(x, x)
            x = x + delta
        return x


# ---------------------------------------------------------------------------
# Edge Predictor
# ---------------------------------------------------------------------------
class EdgePredictor(nn.Module):
    """MLP that predicts edge score from concatenated descriptor pairs."""

    def __init__(self, desc_dim=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(desc_dim * 2, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(hidden, hidden // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden // 2, 1, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, desc):
        """Args: desc (B, C, N) — descriptors for N points.
        Returns: (B, 1, N*N) edge scores.
        """
        b, c, n = desc.shape
        # Build all pairs
        desc_i = desc.unsqueeze(3).expand(b, c, n, n)  # (B,C,N,N)
        desc_j = desc.unsqueeze(2).expand(b, c, n, n)  # (B,C,N,N)
        pairs = torch.cat([desc_i, desc_j], dim=1)  # (B,2C,N,N)
        pairs = pairs.reshape(b, c * 2, n * n)
        return torch.sigmoid(self.mlp(pairs))


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class GNNSlotDetector(nn.Module):
    def __init__(self, desc_dim=256, gnn_layers=4, pretrained=True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.point_head = PointHead(in_ch=512, hidden_ch=256)
        self.desc_head = DescriptorHead(in_ch=512, desc_dim=desc_dim)
        self.gnn = AttentionalGNN(desc_dim, num_layers=gnn_layers)
        self.edge_predictor = EdgePredictor(desc_dim)

        self.desc_dim = desc_dim
        self.max_det = 30

    def sample_descriptors(self, desc_map, points_norm):
        """Sample descriptors at normalized point locations.

        Args:
            desc_map: (B, C, H, W) descriptor feature map
            points_norm: (B, N, 2) normalized coords [0, 1]
        Returns:
            (B, C, N) sampled descriptors
        """
        # Convert [0,1] → [-1,1] for grid_sample
        pts = points_norm * 2 - 1
        B, C, H, W = desc_map.shape
        grid = pts.view(B, 1, -1, 2)  # (B, 1, N, 2)
        sampled = F.grid_sample(desc_map, grid, mode="bilinear",
                                align_corners=True)  # (B, C, 1, N)
        sampled = sampled.reshape(B, C, -1)
        return F.normalize(sampled, p=2, dim=1)

    def detect_points(self, point_pred, conf_thresh=0.3):
        """Extract points from prediction map with direction-aware NMS.

        Args:
            point_pred: (B, 6, 16, 16)
        Returns:
            all_points: list of (N_i, 2) tensors, normalized coords [0,1]
            all_dirs: list of (N_i, 2) tensors, (cos, sin) direction per point
        """
        B = point_pred.shape[0]
        S = point_pred.shape[2]
        all_points = []
        all_dirs = []
        for b in range(B):
            conf = point_pred[b, 0]  # (S, S)
            mask = conf > conf_thresh
            rows, cols = mask.nonzero(as_tuple=True)
            if len(rows) == 0:
                all_points.append(torch.zeros(1, 2, device=point_pred.device))
                all_dirs.append(torch.zeros(1, 2, device=point_pred.device))
                continue
            ox = point_pred[b, 2, rows, cols]
            oy = point_pred[b, 3, rows, cols]
            nx = (cols.float() + ox) / S
            ny = (rows.float() + oy) / S
            pts = torch.stack([nx, ny], dim=1)

            cos_v = point_pred[b, 4, rows, cols]
            sin_v = point_pred[b, 5, rows, cols]

            # Direction-aware scoring: prefer points with strong direction
            dir_mag = (cos_v ** 2 + sin_v ** 2).sqrt()
            conf_vals = conf[rows, cols]
            score = conf_vals * (0.5 + 0.5 * dir_mag.clamp(0, 1))

            if len(pts) > self.max_det:
                topk = score.topk(self.max_det).indices
                pts = pts[topk]
                cos_v = cos_v[topk]
                sin_v = sin_v[topk]

            all_points.append(pts)
            # Normalize direction to unit vector
            dnorm = (cos_v ** 2 + sin_v ** 2).sqrt().clamp(min=1e-8)
            all_dirs.append(torch.stack([cos_v / dnorm, sin_v / dnorm], dim=1))
        return all_points, all_dirs

    def forward(self, images, gt_points=None, npoints=None):
        """Forward pass.

        Args:
            images: (B, 3, 512, 512)
            gt_points: (B, max_points, 2) normalized [0,1], used during training
            npoints: (B,) actual number of points per image
        Returns:
            dict with keys: point_pred, edge_pred, detected_points (inference only)
        """
        feat = self.backbone(images)  # (B, 512, 16, 16)

        # Point detection
        point_pred = self.point_head(feat)  # (B, 6, 16, 16)

        # Descriptor map
        desc_map = self.desc_head(feat)  # (B, 256, 16, 16)

        result = {"point_pred": point_pred}

        if gt_points is not None:
            # Training mode: use GT points to sample descriptors
            desc = self.sample_descriptors(desc_map, gt_points)  # (B, C, max_pts)
        else:
            # Inference: detect points then sample
            detected, detected_dirs = self.detect_points(point_pred)
            # Pad to same size for batching
            max_n = max(d.shape[0] for d in detected)
            padded = torch.zeros(len(detected), max_n, 2,
                                 device=images.device)
            padded_dirs = torch.zeros(len(detected), max_n, 2,
                                      device=images.device)
            for i, d in enumerate(detected):
                padded[i, :d.shape[0]] = d
                padded_dirs[i, :detected_dirs[i].shape[0]] = detected_dirs[i]
            gt_points = padded
            desc = self.sample_descriptors(desc_map, padded)
            result["detected_points"] = detected
            result["detected_dirs"] = detected_dirs
            result["npoints"] = torch.tensor([d.shape[0] for d in detected])

        # GNN refinement
        desc = self.gnn(desc)  # (B, C, N)

        # Edge prediction
        edge_pred = self.edge_predictor(desc)  # (B, 1, N*N)
        result["edge_pred"] = edge_pred
        result["points"] = gt_points

        return result


if __name__ == "__main__":
    # Quick sanity check
    model = GNNSlotDetector()
    x = torch.randn(2, 3, 512, 512)
    pts = torch.rand(2, 10, 2)
    npoints = torch.tensor([8, 10])
    out = model(x, gt_points=pts, npoints=npoints)
    print("point_pred:", out["point_pred"].shape)
    print("edge_pred:", out["edge_pred"].shape)
    # Inference
    model.eval()
    with torch.no_grad():
        out2 = model(x)
    print("detected:", [d.shape for d in out2["detected_points"]])
    print("edge_pred:", out2["edge_pred"].shape)
