import torch

from src.utils.transforms import Resize, ToTensor

img_test = torch.rand((3, 32, 32))
print(img_test)
