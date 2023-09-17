import math
from typing import Any, Tuple
import numpy as np
from PIL import Image, ImageOps
from torch import tensor, float32, Tensor, uint8
from torchvision.transforms import Compose, RandomApply, RandomChoice, RandomOrder, AutoAugment, AugMix, Normalize
from transformers import AutoImageProcessor
# import imgaug as ia
# import imgaug.augmenters as iaa
from random import sample
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Resize(object):
    def __init__(self, size):
        """
            Resize with size (width, height)
            Return nparray image
        """
        self.width = size[0]
        self.height = size[1]

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = img.convert('RGB')
        
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        img = np.asarray(img)
        return img
    
class ResizeWithPadding(object):
    def __init__(self, size):
        """
            Resize with padding to make image size uniform (width, height)
            return nparray
        """
        self.size = size

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = img.convert('RGB')
        img = resize_with_padding(img, self.size)
        return np.asarray(img)

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


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
    
class DefaultAugmenter(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.aug = Compose([
            # ResizeWithPadding(size),
            Resize(size[0], size[1]),
            ToTensor(),
        ])
    
    def __call__(self, img: Image.Image) -> Tensor:
        return self.aug(img)
    
class SwinAugmenter(object):
    def __init__(self, pretrained) -> None:
        image_processor = AutoImageProcessor.from_pretrained(pretrained)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        self.aug = Compose([Resize(size[0], size[1]), ToTensor(), Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])
    
    def __call__(self, img: Image.Image) -> Tensor:
        return self.aug(img)
    
class AlbumentationsWithPadding(object):
    def __init__(self, size: Tuple[int, int]):
        self.resize = ResizeWithPadding(size)
        self.aug = A.Compose([
            # Blur
            A.OneOf([
                A.GaussianBlur(p=0.3, sigma_limit=(0, 1.0)),
                A.MotionBlur(p=0.3, blur_limit=3),
            ]),
            
            # Color
            A.OneOf([
                A.HueSaturationValue(p=0.3, hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10)),
                A.GaussNoise(p=0.3),
                A.ColorJitter(p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                A.RGBShift(p=0.3),
                A.ChannelShuffle(p=0.3),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            ]),
            
            # Solarize and invert
            # A.OneOf([
            #     A.Solarize(p=0.05, threshold=(32, 128)),
            #     A.InvertImg(p=0.1),
            # ]),

            # Dropout and multiply
            A.OneOf([
                A.CoarseDropout(p=0.1, max_holes=4, max_height=0.08, max_width=0.08),
                A.MultiplicativeNoise(p=0.3, multiplier=(0.6, 1.4)),
            ]),

            # Compression
            A.ImageCompression(p=0.3, quality_lower=70, quality_upper=90),

            # Distortions
            A.OneOf([
                A.Perspective(p=0.3, scale=(0.01, 0.01)),
                A.ShiftScaleRotate(p=0.3, scale_limit=(-0.1, 0.1), shift_limit=(-0.05, 0.05), rotate_limit=(-5, 5)),
                A.Affine(p=0.3, scale=(0.7, 1.3), translate_percent=(-0.1, 0.1)),
                A.PiecewiseAffine(p=0.3, scale=(0.01, 0.01)),
            ]),

            # Crop
            # A.OneOf([
            #     # A.Crop(p=0.3, percent=(0.01, 0.05)),
            #     A.CenterCrop(p=0.3, height=0.8*size[0], width=0.8*size[1]),
            # ]),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def __call__(self, img):
        img = self.resize(img)
        augmented = self.aug(image=img)
        img = augmented['image']
        return img

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

class AlbumentationsTransform(object):
    def __init__(self, size: Tuple[int, int]):
        self.aug = A.Compose([
            # Blur
            A.OneOf([
                A.GaussianBlur(p=0.3, sigma_limit=(0, 1.0)),
                A.MotionBlur(p=0.3, blur_limit=3),
            ]),
            
            # Color
            A.OneOf([
                A.HueSaturationValue(p=0.3, hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10)),
                A.GaussNoise(p=0.3),
                A.ColorJitter(p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                A.RGBShift(p=0.3),
                A.ChannelShuffle(p=0.3),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            ]),
            
            # Solarize and invert
            # A.OneOf([
            #     A.Solarize(p=0.05, threshold=(32, 128)),
            #     A.InvertImg(p=0.1),
            # ]),

            # Dropout and multiply
            A.OneOf([
                A.CoarseDropout(p=0.1, max_holes=4, max_height=0.08, max_width=0.08),
                A.MultiplicativeNoise(p=0.3, multiplier=(0.6, 1.4)),
            ]),

            # Compression
            A.ImageCompression(p=0.3, quality_lower=70, quality_upper=90),

            # Distortions
            A.OneOf([
                A.Perspective(p=0.3, scale=(0.01, 0.01)),
                A.ShiftScaleRotate(p=0.3, scale_limit=(-0.1, 0.1), shift_limit=(-0.05, 0.05), rotate_limit=(-5, 5)),
                A.Affine(p=0.3, scale=(0.7, 1.3), translate_percent=(-0.1, 0.1)),
                A.PiecewiseAffine(p=0.3, scale=(0.01, 0.01)),
            ]),

            # Crop
            # A.OneOf([
            #     # A.Crop(p=0.3, percent=(0.01, 0.05)),
            #     A.CenterCrop(p=0.3, height=0.8*size[0], width=0.8*size[1]),
            # ]),
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def __call__(self, img):
        img = np.array(img)
        augmented = self.aug(image=img)
        img = augmented['image']
        return img
