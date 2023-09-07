import math
import numpy as np
from PIL import Image
from torch import tensor, float32, Tensor, uint8
from torchvision.transforms import Compose, RandomApply, RandomChoice, RandomOrder, AutoAugment, AugMix
# import imgaug as ia
# import imgaug.augmenters as iaa
from random import sample

class Resize(object):
    def __init__(self, height, min_width):
        self.height = height
        self.min_width = min_width
        # self.max_width = max_width

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = img.convert('RGB')
        w, h = img.size
        # new_w = self.get_new_size(w, h)

        # variable width when batching, need to bucket/sampler to group data with same width
        # img = img.resize((new_w, self.height), Image.ANTIALIAS)
        
        img = img.resize((self.min_width, self.height), Image.ANTIALIAS)
        img = np.asarray(img)
        return img

    # def get_new_size(self, w, h):
    #     new_w = int(self.height * float(w) / float(h))
    #     round_to = 10
    #     new_w = math.ceil(new_w/round_to)*round_to
    #     new_w = max(new_w, self.min_width)
    #     new_w = min(new_w, self.max_width)

    #     return new_w
    
class ToTensor(object):
    def __call__(self, img: np.ndarray) -> Tensor:
        img = img.transpose(2, 0, 1)
        img = img / 225
        return tensor(img, dtype= float32)

class ToUInt8(object):
    def __call__(self, img: Tensor) -> Tensor:
        if img.dtype != uint8:
            img = img.type(uint8)
        return img
    
class ToFloat32(object):
    def __call__(self, img: Tensor) -> Tensor:
        if img.dtype != float32:
            img = img.type(float32)
        return img
    
class Augmenters(object):
    def __init__(self) -> None:
        self.aug = Compose([
            Resize(70, 140),
            ToTensor(),
            RandomChoice([
                # ToUInt8(),
                # AutoAugment(),
                # ToFloat32(),
                AugMix(),
            ], p= [0.4]),
        ])
    
    def __call__(self, img: Image.Image) -> Tensor:
        return self.aug(img)

# the fucking imgaug lib is dogshit -> np.bool decaped
# class VietOCRAug(object):
#     def __init__(self) -> None:
#         sometimes = lambda aug: iaa.Sometimes(0.3, aug)

#         self.aug = iaa.Sequential(iaa.SomeOf((1, 5), 
#             [
#             # blur

#             sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
#                                 iaa.MotionBlur(k=3)])),
            
#             # color
#             sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
#             sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
#             sometimes(iaa.Invert(0.25, per_channel=0.5)),
#             sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
#             sometimes(iaa.Dropout2d(p=0.5)),
#             sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
#             sometimes(iaa.Add((-40, 40), per_channel=0.5)),

#             sometimes(iaa.JpegCompression(compression=(5, 80))),
            
#             # distort
#             sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
#             sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
#             sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), 
#     #                            rotate=(-5, 5), shear=(-5, 5), 
#                                 order=[0, 1], cval=(0, 255), 
#                                 mode=ia.ALL)),
#             sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
#             sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                 iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

#         ],
#             random_order=True),
#         random_order=True)

#     def __call__(self, img: np.ndarray) -> np.ndarray:
#         return self.aug.augment_image(img)