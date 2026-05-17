"""C2f module augmented with CBAM (Convolutional Block Attention Module).

CBAM is applied after the C2f output (post-cv2), operating on c2 channels.
Channel attention → Spatial attention → output.
"""

import torch.nn as nn
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import CBAM


class C2fCBAM(C2f):
    """C2f with CBAM attention on output.

    Inherits C2f so pretrained C2f weights transfer seamlessly via intersect_dicts.
    CBAM parameters (~87K across 3 backbone blocks) are randomly initialized.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        return self.cbam(super().forward(x))
