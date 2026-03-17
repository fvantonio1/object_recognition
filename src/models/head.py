import torch.nn as nn
from .backbone import build_backbone
import math

class Head(nn.Module):

    def __init__(self, in_channels, num_anchors, num_classes, num_convs=4, prior=0.01):
        super().__init__()

        cls_layers = []
        box_layers = []

        for _ in range(num_convs):

            cls_layers.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
            cls_layers.append(nn.ReLU())

            box_layers.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
            box_layers.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_layers)
        self.box_tower = nn.Sequential(*box_layers)

        self.cls_head = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            padding=1
        )

        self.box_head = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            padding=1
        )

        self.pi = prior

        self._intialize_weights()

    def _intialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

        cls_bias = -math.log((1 - self.pi) / self.pi)
        nn.init.constant_(self.cls_head.bias, cls_bias)

    def forward(self, features):

        cls_outputs = []
        box_outputs = []

        for f in features:
            cls_outputs.append(self.cls_head(f))
            box_outputs.append(self.box_head(f))

        return cls_outputs, box_outputs
