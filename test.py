from transformers import AutoImageProcessor, SwinModel
from PIL import Image
import torch

image = Image.open('./data/new_train/train_img_0.jpg')

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    embedding = model(**inputs).last_hidden_state

print(embedding.shape)

