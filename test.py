from src.data.components.sampler import VariableSizeSampler
from src.data.components.datasets import OCRDataset
from src.models.tokenizer import Tokenizer
from src.models.model import Net
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import random
import torch
from tqdm import tqdm
from torchvision import models
from src.utils.transforms import VariableResize, ToTensor, VarialeSizeAugmenter

print(models.efficientnet_b0())

# from PIL import Image

# aug = VarialeSizeAugmenter(70, 100, 200)

# img = Image.open('./data/new_train/train_img_0.jpg')

# print(aug(img).shape)

# class Config(dict):
#     def __init__(self, config_dict):
#         super(Config, self).__init__(**config_dict)
#         self.__dict__ = self

#     @staticmethod
#     def load_config_from_file(path):
#         base_config = {}
#         with open(path, mode= 'r', encoding= 'utf8') as f:
#             config = yaml.safe_load(f)
#         base_config.update(config)
#         return Config(base_config)
    
# cfg = Config.load_config_from_file('./config.yaml')
# cfg = Config.load_config_from_file('./backbone.yaml')
# # print(cfg)
# print(cfg['transformer'])
# print(cfg['backbone'])
# print(cfg['data'])
# print(cfg['optimizer'])
# print(cfg['logger'])
# print(cfg['checkpoint'])
# print(cfg['early_stopping'])
# print(cfg['profiler'])
# print(cfg['trainer'])

# mapping = []
# with open('./data/train_gt.txt', encoding= 'utf8') as f:
#     for l in f.readlines():
#         path, label = l.strip().split()
#         mapping.append((path, label))

# data = OCRDataset('./data/new_train', mapping, False, tokenizer= Tokenizer(), transforms= Compose([VariableResize(), ToTensor()]))
# b_s = 16

# sampler = VariableSizeSampler(data, b_s)

# print(len(list(sampler)))

# dm = DataLoader(data, batch_sampler= sampler)
# print(len(dm))

# batch_lists = []
# for bucket, idx in data.bucket.items():
#     random.shuffle(idx)
#     batches = [idx[i:i + b_s] for i in range(0, len(idx), b_s)]
#     random.shuffle(batches)
#     batch_lists.append(batches)
# lst = [item for sublist in batch_lists for item in sublist]
# random.shuffle(lst)
# print(len(lst))

# for _ in tqdm(iter(lst)):
#     pass

# conv = torch.nn.Conv2d
# print(conv)
# convv2 = conv(12, 12, 1)
# print(convv2)

# print(len(lst))

# for i in lst:
#     print(data[i]['img'].shape)
# print(len(idx))


# vit_args = {
#         'arch': 'vit',
#         'hidden': 256,
#         'dropout': 0.1,
#         'pretrained': 'google/vit-base-patch32-384'
#     }

# trans_args = {
#         "d_model": 256,
#         "nhead": 8,
#         "num_encoder_layers": 6,
#         "num_decoder_layers": 6,
#         "dim_feedforward": 2048,
#         "max_seq_length": 512,
#         "pos_dropout": 0.2,
#         "trans_dropout": 0.1
#     }
# net = Net(len(Tokenizer().chars), 'transformers', vit_args, trans_args)

# print(net.backbone)

