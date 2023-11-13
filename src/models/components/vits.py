from transformers import SwinModel, ViTModel
from torch.nn import Conv2d, Module, Dropout, Linear

class VisionTransformer(Module):
    def __init__(self, arch, **kwargs):
        super().__init__()

        if arch == 'swin':
            self.backbone = SwinTransformer(**kwargs)
        elif arch == 'vit':
            self.backbone = VITEncoder(**kwargs)
        else:
            raise('No vit-based backbone supported')
    
    def forward(self, x):
        return self.backbone(x)

class SwinTransformer(Module):
    def __init__(self, hidden, dropout= 0.5, pretrained= 'microsoft/swin-tiny-patch4-window7-224'):
        super().__init__()
        self.swin = SwinModel.from_pretrained(pretrained)

        for p in self.swin.parameters():
            p.requires_grad = False
        self.dropout = Dropout(dropout)
        self.last_conv = Conv2d(768, hidden, 1)
    
    def forward(self, x):
        x = self.swin(x, output_hidden_states= True).reshaped_hidden_states[-1]
        # print(x)
        # print(x.shape)
        # x.shape (64, 768, 7, 7)
        conv = self.dropout(x)
        conv = self.last_conv(conv)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        # print(conv.shape)
        # conv.shape (49, 64, 256)
        return conv
    
class VITEncoder(Module):
    def __init__(self, hidden, dropout= 0.01, pretrained= 'google/vit-base-patch32-384'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained)
        self.dropout = Dropout(dropout)
        self.lin = Linear(768, hidden)

    def forward(self, x):
        # print(x.shape)
        x = self.vit(x, return_dict= True)['last_hidden_state']
        # print(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.lin(x)
        x = x.permute(1, 0, 2)
        return x
