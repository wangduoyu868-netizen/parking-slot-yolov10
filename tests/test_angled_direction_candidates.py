import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.angled_direction_tools import angle_candidates


def dot_abs(a, b):
    return abs(a[0] * b[0] + a[1] * b[1])


def test_angle_candidates_are_unit_vectors_at_expected_angle():
    entrance_dir = (1.0, 0.0)

    cand_a, cand_b = angle_candidates(entrance_dir, 67.0)

    assert math.isclose(math.hypot(*cand_a), 1.0, abs_tol=1e-6)
    assert math.isclose(math.hypot(*cand_b), 1.0, abs_tol=1e-6)
    assert math.isclose(dot_abs(cand_a, entrance_dir), math.cos(math.radians(67.0)), abs_tol=1e-6)
    assert math.isclose(dot_abs(cand_b, entrance_dir), math.cos(math.radians(67.0)), abs_tol=1e-6)
    assert cand_a[1] > 0
    assert cand_b[1] < 0


if __name__ == "__main__":
    test_angle_candidates_are_unit_vectors_at_expected_angle()
