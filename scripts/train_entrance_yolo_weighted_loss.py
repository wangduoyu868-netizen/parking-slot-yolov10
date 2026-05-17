"""Train/fine-tune EntranceDetect with configurable entrance loss weights."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from custom_modules.entrance_loss import EntranceLineLoss
from custom_modules.entrance_loss import v8EntranceDetectionLoss as BaseEntranceLoss

import scripts.train_entrance_yolo as base_train

LOSS_WEIGHTS = (4.0, 1.0, 0.5)


class WeightedEntranceLoss(BaseEntranceLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        w_line, w_body, w_type = LOSS_WEIGHTS
        self.entrance_loss = EntranceLineLoss(
            w_line=w_line,
            w_body=w_body,
            w_type=w_type,
        )


def parse_weight_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--w-line", type=float, default=4.0)
    parser.add_argument("--w-body", type=float, default=1.0)
    parser.add_argument("--w-type", type=float, default=0.5)
    return parser.parse_known_args(argv)


def main() -> None:
    global LOSS_WEIGHTS
    weight_args, remaining = parse_weight_args(sys.argv[1:])
    LOSS_WEIGHTS = (weight_args.w_line, weight_args.w_body, weight_args.w_type)
    base_train.v8EntranceDetectionLoss = WeightedEntranceLoss
    sys.argv = [sys.argv[0], *remaining]
    print(
        f"Using entrance loss weights: "
        f"w_line={weight_args.w_line}, "
        f"w_body={weight_args.w_body}, "
        f"w_type={weight_args.w_type}"
    )
    base_train.main()


if __name__ == "__main__":
    main()
