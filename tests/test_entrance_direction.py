import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_ps20_to_entrance_yolo import entrance_from_slot


def test_body_direction_falls_back_to_entrance_normal_when_mark_dirs_are_parallel():
    marks = [
        [123.30765713807581, 474.59840964373865, 170.18115945638604, 368.39677131691923, 0],
        [186.02909465673062, 332.078563500872, 141.75833210143139, 431.94711532588, 1],
    ]

    _, extra = entrance_from_slot(marks, [1, 2, 1, 90])

    x1, y1, x2, y2, dx, dy, _slot_type = extra
    ex = x2 - x1
    ey = y2 - y1
    elen = math.hypot(ex, ey)
    dot_abs = abs((ex * dx + ey * dy) / elen)

    assert dot_abs < 0.25


def test_body_direction_uses_normal_when_mat_marks_have_no_direction():
    marks = [
        [100.0, 100.0],
        [100.0, 300.0],
    ]

    _, extra = entrance_from_slot(marks, [2, 1, 1, 90])

    x1, y1, x2, y2, dx, dy, _slot_type = extra
    ex = x2 - x1
    ey = y2 - y1
    elen = math.hypot(ex, ey)
    dot_abs = abs((ex * dx + ey * dy) / elen)

    assert dot_abs < 0.25


if __name__ == "__main__":
    test_body_direction_falls_back_to_entrance_normal_when_mark_dirs_are_parallel()
    test_body_direction_uses_normal_when_mat_marks_have_no_direction()
