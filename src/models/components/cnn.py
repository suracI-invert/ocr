import torch
from torch import nn
from src.models.components.vgg import Vgg
from src.models.components.resnet import Resnet50

class CNN(nn.Module):
    def __init__(self, cnn_arch, **kwargs):
        super(CNN, self).__init__()

        if cnn_arch == 'vgg':
            self.model = Vgg(**kwargs)
        elif cnn_arch == 'resnet50':
            self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
