import torch.nn as nn
import torch
from .anchor import AnchorGenerator
from .head import Head
from .backbone import build_backbone
from .fpn import FPN


class Detector(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg['BACKBONE'])

        self.fpn = FPN(
            in_channels=cfg['FPN']['IN_CHANNELS'],
            out_channels=cfg['FPN']['OUT_CHANNELS']
        )
        
        self.anchor_generator = AnchorGenerator(
            sizes=cfg['ANCHORS']['SIZES'],
            aspect_ratios=cfg['ANCHORS']['RATIOS'],
            strides=cfg['ANCHORS']['STRIDES'],
            scales=cfg['ANCHORS']['SCALES']
        )

        self.head = Head(
            in_channels=cfg['HEAD']['CHANNELS'], 
            num_anchors=len(cfg['ANCHORS']['RATIOS']) * len(cfg['ANCHORS']['SCALES']),
            num_classes=cfg['NUM_CLASSES'],
            num_convs=cfg['HEAD']['NUM_CONVS']
        )


    def forward(self, x):

        features = self.backbone(x)

        pyramid = self.fpn(features)

        cls_outputs, box_outputs = self.head(pyramid)

        anchors = self.anchor_generator(pyramid)

        return cls_outputs, box_outputs, anchors.detach()
