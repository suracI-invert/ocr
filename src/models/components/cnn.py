import torch
from torch import nn
from torchvision import models

class Vgg(nn.Module):
    def __init__(self, ss, ks, hidden, pretrained: False, dropout= 0.5):
        super().__init__()
        weights = models.VGG19_BN_Weights if pretrained else None
        cnn = models.vgg19_bn(weights)
        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size= ks[pool_idx], stride= ss[pool_idx], padding= 0)
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

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
    
class EfficentNet(nn.Module):
    def __init__(self, ver= 'b0', hidden= 256, pretrained= False, dropout= 0.1):
        super().__init__()

        if ver == 'b0':
            weights = models.EfficientNet_B0_Weights if pretrained else None
            model = models.efficientnet_b0(weights)
            out_dim = 1280
        elif ver == 'b1':
            weights = models.EfficientNet_B1_Weights if pretrained else None
            model = models.efficientnet_b1(weights)
            out_dim = 1280
        elif ver == 'b2':
            weights = models.EfficientNet_B2_Weights if pretrained else None
            model = models.efficientnet_b2(weights)
            out_dim = 1408
        elif ver == 'b3':
            weights = models.EfficientNet_B3_Weights if pretrained else None
            model = models.efficientnet_b3(weights)
            out_dim = 1536
        elif ver == 'b4':
            weights = models.EfficientNet_B4_Weights if pretrained else None
            model = models.efficientnet_b4(weights)
            out_dim = 1792
        elif ver == 'b5':
            weights = models.EfficientNet_B5_Weights if pretrained else None
            model = models.efficientnet_b5(weights)
            out_dim = 2048
        elif ver == 'b6':
            weights = models.EfficientNet_B6_Weights if pretrained else None
            model = models.efficientnet_b6(weights)
            out_dim = 2304
        elif ver == 'b7':
            weights = models.EfficientNet_B7_Weights if pretrained else None
            model = models.efficientnet_b7(weights)
            out_dim = 2560
        else:
            raise('backbone not found')
        
        self.features = model.features
        self.avgpool = model.avgpool
        self.last_layer = nn.Sequential(
            nn.Conv2d(out_dim, 512, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(512, hidden, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.last_layer(x)

        x = x.transpose(-1, -1)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        conv = self.relu(x)
        
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)

        return conv

def Resnet50(ss, hidden):
    return ResNet(3, hidden, BasicBlock, [1, 2, 5, 3])

class CNN(nn.Module):
    def __init__(self, arch, **kwargs):
        super(CNN, self).__init__()

        if arch == 'vgg':
            self.model = Vgg(**kwargs)
        elif arch == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif arch == 'efficientnet':
            self.model = EfficentNet(**kwargs)
        else:
            raise('Not supported cnn backbone')

    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

