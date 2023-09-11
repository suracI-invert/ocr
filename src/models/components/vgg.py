import torch
from torch import nn
from torchvision import models


class Vgg(nn.Module):
    def __init__(self, ss, ks, hidden, weights=None, dropout=0.5):
        super().__init__()
        self.ss = ss
        self.ks = ks
        self.hidden = hidden
        self.weights = weights
        self.dropout = dropout
        cnn = models.vgg11_bn(self.weights)
        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(
                    kernel_size=self.ks[pool_idx], stride=self.ss[pool_idx], padding=0
                )
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(self.dropout)
        self.last_conv_1x1 = nn.Conv2d(512, self.hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (B, C, H, W)
            - output: (W, B, C)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv
