from __future__ import division
from layers_2D import RotConv, VectorMaxPool, VectorBatchNorm, Vector2Magnitude, VectorUpsampling
from torch import nn


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.main = nn.Sequential(
            RotConv(1, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),  # The first RotConv must have mode=1
            VectorMaxPool(2),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
            # The next RotConv has mode=2 (since the input is vector field)
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(),
            # This call converts the vector field to a conventional multichannel image/feature image

            nn.Conv2d(32, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),

        )

    def forward(self, x):
        x = self.main(x)
        return x





