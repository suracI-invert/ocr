from torch import tensor, float32, int64, bool
import numpy as np

class Collator(object):
    def __init__(self, masked_language_model: bool = True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        img = []
        tgt_input = []
        target_weights = []
        filenames = []

        for sample in batch:
            img.append(sample['img'])
            tgt_input.append(sample['label_ids'])
            target_weights.append(sample['attn_mask'])
            filenames.append(sample['filename'])
            
        img = np.array(img, dtype=np.float32)

        tgt_input = np.array(tgt_input, dtype=np.int64).T

        # output got shifted left 1 pos -> got rid of <s> token ? no idea why do that : currently testing
        tgt_output = np.roll(tgt_input, -1, 0).T
        # tgt_output = np.roll(tgt_input, 0, 0).T
        tgt_output[:, -1] = 0
        
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)

        return {
            'img': tensor(img, dtype= float32),
            'tgt_input': tensor(tgt_input, dtype= int64),
            'tgt_output': tensor(tgt_output, dtype= int64),
            'tgt_padding_mask': tensor(tgt_padding_mask, dtype= bool),
            'filename': filenames
        }   