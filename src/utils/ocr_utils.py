import torch
import matplotlib.pyplot as plt
import numpy as np

def imshow_batch(images, grid_shape=(8, 8)):

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    images = denormalize(images)

    fig = plt.figure(figsize=(8, 8))

    for i, img in enumerate(images):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i + 1, xticks=[], yticks=[])
        ax.imshow(img)
    plt.show()