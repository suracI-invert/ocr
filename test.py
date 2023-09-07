from transformers import AutoImageProcessor, SwinModel
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import Compose, Normalize

from src.utils.transforms import Resize, ToTensor

from src.models.tokenizer import Tokenizer
from src.data.components.collator import Collator
from src.data.datamodule import OCRDataModule


if __name__ == '__main__':
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    tokenizer = Tokenizer()
    collator = Collator()

    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    dataModule = OCRDataModule(
        data_dir= './data/', map_file= 'train_annotation.txt',
        test_dir= './data/new_public_test',
        tokenizer= tokenizer,
        train_val_split= [100_000, 3_000],
        batch_size= 64,
        num_workers= 6,
        pin_memory= True,
        transforms= Compose([Resize(size[0], size[1]), ToTensor(), Normalize(mean=image_processor.image_mean, std=image_processor.image_std)]),
        collate_fn= collator,
        sampler= None
    )

    dataModule.setup()

    print(len(dataModule.data_train))
    print(len(dataModule.data_valid))
    print(len(dataModule.data_test))

    sample = next(iter(dataModule.train_dataloader()))


    image = sample['img']


    for param in model.parameters():
        param.requires_grad = False

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        embedding = model(image, output_hidden_states=True).reshaped_hidden_states[-1]

    print(embedding.shape)


    conv_layer = last_conv_1x1 = torch.nn.Conv2d(768, 255, 1)
    conv = conv_layer(embedding)
    conv = conv.transpose(-1, -2)
    conv = conv.flatten(2)
    conv = conv.permute(-1, 0, 1)

    print(conv.shape)