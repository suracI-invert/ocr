import math
from random import sample
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor, float32, tensor, uint8
from torchvision.transforms import (
    AugMix,
    AutoAugment,
    Compose,
    Normalize,
    RandomApply,
    RandomChoice,
    RandomOrder,
)
from transformers import AutoImageProcessor


class Resize:
    def __init__(self, height, min_width):
        self.height = height
        self.min_width = min_width
        # self.max_width = max_width

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
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


class ToTensor:
    def __call__(self, img: np.ndarray) -> Tensor:
        img = img.transpose(2, 0, 1)
        img = img / 225
        return tensor(img, dtype=float32)


class ToUInt8:
    def __call__(self, img: Tensor) -> Tensor:
        if img.dtype != uint8:
            img = img.type(uint8)
        return img


class ToFloat32:
    def __call__(self, img: Tensor) -> Tensor:
        if img.dtype != float32:
            img = img.type(float32)
        return img


class Augmenters:
    def __init__(self) -> None:
        self.aug = Compose(
            [
                Resize(70, 140),
                ToTensor(),
                RandomChoice(
                    [
                        # ToUInt8(),
                        # AutoAugment(),
                        # ToFloat32(),
                        AugMix(),
                    ],
                    p=[0.4],
                ),
            ]
        )

    def __call__(self, img: Image.Image) -> Tensor:
        return self.aug(img)


class SwinAugmenter:
    def __init__(self, pretrained) -> None:
        image_processor = AutoImageProcessor.from_pretrained(pretrained)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        self.aug = Compose(
            [
                Resize(size[0], size[1]),
                ToTensor(),
                Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ]
        )

    def __call__(self, img: Image.Image) -> Tensor:
        return self.aug(img)


# class AlbumentationsTransform:
#     def __init__(self):
#         self.aug = A.Compose([
#             # Blur
#             A.OneOf([
#                 A.GaussianBlur(blur_limit=(0, 1.0)),
#                 A.MotionBlur(blur_limit=(3, 3)),
#             ], p=0.3),

#             # Color
#             A.OneOf([
#                 A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=1),
#                 A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1),
#                 A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
#             ], p=0.3),

#             A.OneOf([
#                 A.InvertImg(always_apply=True, p=1),
#                 A.Solarize(threshold=(32, 128), p=1),
#             ], p=0.3),

#             A.OneOf([
#                 A.ChannelDropout(channel_drop_range=(1, 1), p=0.5),
#                 A.ChannelShuffle(p=0.5),
#                 A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
#             ], p=0.3),

#             # JPEG Compression
#             A.JpegCompression(quality_lower=5, quality_upper=80, p=0.3),

#             # Distortions
#             A.OneOf([
#                 A.Crop(percent=(0.01, 0.05), p=1),
#                 A.Perspective(scale=(0.01, 0.01), p=1),
#                 A.Affine(scale=(0.7, 1.3), translate_percent={"x": (-0.1, 0.1)}, mode=A.BORDER_REFLECT, p=1),
#                 A.PiecewiseAffine(scale=(0.01, 0.01), p=1),
#                 A.OneOf([
#                     A.CoarseDropout(max_holes=1, max_height=int(0.02*70), max_width=int(0.25*140), p=1),
#                     A.CoarseDropout(max_holes=1, max_height=int(0.25*70), max_width=int(0.02*140), p=1),
#                 ], p=0.5),
#             ], p=0.5),

#             ToTensorV2(),
#         ])

#     def __call__(self, img):
#         img = np.array(img)
#         augmented = self.aug(image=img)
#         img = augmented['image']
#         return img
