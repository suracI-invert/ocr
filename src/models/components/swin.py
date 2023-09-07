from transformers import AutoImageProcessor, SwinModel
from PIL import Image
from torch.nn import Conv2d, Module

class SwinTransformer(Module):
    def __init__(self, hidden_channels, pretrained= 'microsoft/swin-tiny-patch4-window7-224'):
        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        self.swin = SwinModel.from_pretrained(pretrained)

        for p in self.swin.parameters():
            p.requires_grad = False

        self.last_conv = Conv2d(768, hidden_channels, 1)
    
    def forward(self, x):
        x = self.processor(x, return_tensors= 'pt')
        x = self.swin(**x, output_hidden_states= True).reshaped_hidden_states[-1]
        conv = self.last_conv(x)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)