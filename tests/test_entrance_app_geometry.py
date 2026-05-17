import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.entrance_app_utils import build_slot_quad, normalize_vec


def test_build_slot_quad_extends_from_entrance_along_body_direction():
    p1 = np.array([100.0, 200.0], dtype=np.float32)
    p2 = np.array([300.0, 200.0], dtype=np.float32)
    body_dir = np.array([0.0, 1.0], dtype=np.float32)

    quad = build_slot_quad(p1, p2, body_dir, depth_ratio=1.0, depth_min=10, depth_max=500)

    assert np.allclose(quad[0], [100.0, 200.0])
    assert np.allclose(quad[1], [300.0, 200.0])
    assert np.allclose(quad[2], [300.0, 400.0])
    assert np.allclose(quad[3], [100.0, 400.0])


def test_normalize_vec_returns_unit_fallback_for_zero_vector():
    vec = normalize_vec(np.array([0.0, 0.0], dtype=np.float32), fallback=(1.0, 0.0))

    assert math.isclose(float(np.linalg.norm(vec)), 1.0, abs_tol=1e-6)
    assert np.allclose(vec, [1.0, 0.0])


if __name__ == "__main__":
    test_build_slot_quad_extends_from_entrance_along_body_direction()
    test_normalize_vec_returns_unit_fallback_for_zero_vector()
