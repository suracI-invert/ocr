from transformers import SwinModel
from PIL import Image
from torch.nn import Conv2d, Module, Dropout

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
        conv = self.dropout(x)
        conv = self.last_conv(conv)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv