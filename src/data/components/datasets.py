from torch.utils.data import Dataset
from torch import tensor, float32
from PIL import Image
from os.path import join, basename
from os import listdir
from torchvision.transforms.functional import pil_to_tensor

from src.models.tokenizer import Tokenizer


class OCRDataset(Dataset):
    def __init__(self, root_dir: str, map_file: str = None, test_data: bool = False, tokenizer: Tokenizer = None,
                max_target_length: int = 128, 
                transforms= None
            ):

        self.root_dir = root_dir
        self.paths = []
        self.labels = []

        self.test_data = test_data

        if test_data:
            self.paths = listdir(self.root_dir)

        if map_file:
            with open(join(root_dir, map_file), encoding= 'utf8') as f:
                for l in f.readlines():
                    path, label = l.strip().split()
                    self.paths.append(path)
                    self.labels.append(label)
    
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

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

        tokenized_labels = self.tokenizer.encode(self.labels[idx], padding= True, max_length= self.max_target_length, return_tensor= True)
        
        return {
            'img': img,
            'label_ids': tokenized_labels['input_ids'],
            'attn_mask': tokenized_labels['attention_masks'], 
            'filename': basename(self.paths[idx])
        }