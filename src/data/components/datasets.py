

from typing import Tuple
from torch.utils.data import Dataset
from torch import tensor, float32
from PIL import Image
from os.path import join, basename
from os import listdir
from torchvision.transforms.functional import pil_to_tensor

from src.models.tokenizer import Tokenizer
from src.utils.transforms import resize_variable_size

from tqdm import tqdm
from collections import defaultdict

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, mapping: [Tuple[str, str]] = None, 
                test_data: bool = False, tokenizer: Tokenizer = None, h: int = 70, 
                min_w: int = 100, max_w: int = 300,
                transforms= None
            ):

        self.root_dir = root_dir
        self.paths = []
        self.labels = []

        self.test_data = test_data

        self.h = h
        self.min_w = min_w
        self.max_w = max_w

        if test_data:
            self.paths = listdir(self.root_dir)

        if mapping:
            for l in mapping:
                path, label = l
                self.paths.append(path)
                self.labels.append(label)
        
            self.bucket = defaultdict(list)
            for i, p in tqdm(enumerate(self.paths), desc= 'Loading metadata'):
                w, h = Image.open(join(self.root_dir, p)).size
                w, _ = resize_variable_size(w, h, self.h, self.min_w, self.max_w)
                self.bucket[w].append(i)

        self.tokenizer = tokenizer
        self.transforms = transforms


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.test_data:
            img = Image.open(join(self.root_dir, self.paths[idx]))
            if self.transforms:
                img = self.transforms(img)
            else:
                img = img.resize((70, 140))
                img = pil_to_tensor(img.convert('RGB'))
            return {
                'img': img,
                'filename': basename(self.paths[idx])
            }


        img = Image.open(join(self.root_dir, self.paths[idx]))

        if self.transforms:
            img = self.transforms(img)
        else:
            img = img.resize((70, 140))
            img = pil_to_tensor(img.convert('RGB'))

        tokenized_labels = self.tokenizer.encode(self.labels[idx], padding= True, return_tensor= True)
        return {
            'img': img,
            'label_ids': tokenized_labels['input_ids'],
            'attn_mask': tokenized_labels['attention_masks'], 
            'filename': basename(self.paths[idx])
        }