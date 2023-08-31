from torchvision import transforms
from torch.nn.functional import sigmoid
import random
import torch


lst = random.sample([
    transforms.GaussianBlur(kernel_size= 1, sigma= (0.1, 1)),
    transforms.ColorJitter(contrast= (0.4, 0.6), hue= (-0.5, 0.5), saturation= (0, 10)),
    transforms.RandomInvert(0.25),
    transforms.RandomSolarize(128, 0.5)
], k= 2)

transforms_lst = [transforms.ToTensor()] + lst + [transforms.Resize((10, 20))]
print(transforms.Compose(transforms_lst))