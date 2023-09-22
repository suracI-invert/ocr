from torch.utils.data import BatchSampler
import random

class VariableSizeSampler(object):
    def __init__(self, data_src, batch_size, shuffle= False):
        batch_lists = []
        for bucket, blucket_ids in data_src.bucket.items():
            if shuffle:
                random.shuffle(blucket_ids)
            batches = [blucket_ids[i:i + batch_size] for i in range(0, len(blucket_ids), batch_size)]
            if shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)
        
        self.lst = [item for sublist in batch_lists for item in sublist]
        if shuffle:
            random.shuffle(self.lst)
    
    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        return iter(self.lst)
        