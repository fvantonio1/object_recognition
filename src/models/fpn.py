import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):

    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1)
            for c in in_channels
        ])

        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])

        self.p6 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.p7 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, features):
        c2 = features[0]
        c3 = features[1]
        c4 = features[2]
        c5 = features[3]

        p5 = self.lateral[3](c5)

        p4 = self.lateral[2](c4) + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest"
        )

        p3 = self.lateral[1](c3) + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest"
        )

        p2 = self.lateral[0](c2) + F.interpolate(
            p3,
            size=c2.shape[-2:],
            mode="nearest"
        )

        p5 = self.output[3](p5)
        p4 = self.output[2](p4)
        p3 = self.output[1](p3)
        p2 = self.output[0](p2)

        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        return [p3, p4, p5, p6, p7]