from src.utils.transforms import Resize, ToTensor
import torch

img_test = torch.rand((3, 32, 32))
print(img_test)