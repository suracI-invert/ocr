from torch.utils.data import Sampler
import random

from collections import defaultdict

class VariableSizeSampler(Sampler):
    def __init__(self, data_src, batch_size, shuffle= False):
        self.data = data_src.bucket
        self._len = len(data_src)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return self._len

    def __iter__(self):
        batch_lists = []
        for bucket, blucket_ids in self.data.items():
            if self.shuffle:
                random.shuffle(blucket_ids)

            batches = [blucket_ids[i:i + self.batch_size] for i in range(0, len(blucket_ids), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)
        
        lst = [item for sublist in batch_lists for item in sublist]
        if self.shuffle:
            random.shuffle(lst)
        lst = [item for sublist in lst for item in sublist]
        return iter(lst)
        