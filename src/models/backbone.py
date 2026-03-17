import torch
import torch.nn as nn
import torchvision

def build_backbone(cfg):

    if cfg['NAME'].startswith('resnet'):
        return ResNetBackbone(cfg['NAME'])

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

class ResNetBackbone(Backbone):
    def __init__(self, architecture):
        super().__init__()

        model = getattr(torchvision.models, architecture)(weights="DEFAULT")

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    @property
    def out_channels(self):
        return self._out_channels
    
    def forward(self, x):
        c1 = self.stem(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]