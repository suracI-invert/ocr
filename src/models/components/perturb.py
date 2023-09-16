from torch import nn, stack
from torch.nn import functional as F

class PerturbationLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.convp = nn.LazyConv2d(3, 3, 1, 1)
    
    def forward(self, x, p_imgs):
        c1 = self.conv1(x)
        c1p = []
        for i in range(c1.shape[0]):
            for pi in p_imgs:
                c1p.append(c1[i])
                c1p.append(pi)
        c1p = stack(c1p, dim= 1)
        c2p = self.convp(c1p)

        c1 = self.bn(c1)
        c1 = self.act(c1)

        return c1 * c2p

        