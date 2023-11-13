from src.data.components.collator import Collator
from src.models.tokenizer import Tokenizer
from src.data.datamodule import OCRDataModule
from src.data.components.sampler import VariableSizeSampler
from src.utils.transforms import Resize, ToTensor, AlbumentationsTransform, VITAugmenter, VariableResize, VarialeSizeAugmenter
from torchvision.transforms import Compose
from torchvision.utils import save_image

from tqdm import tqdm

if __name__ == '__main__':
    tokenizer = Tokenizer()
    collator = Collator()
    # Augmenter = AlbumentationsTransform((70, 140))

    dataModule = OCRDataModule(
        data_dir= './data/new_train', map_file= './data/train_gt.txt',
        test_dir= './data/new_public_test',
        tokenizer= tokenizer,
        train_val_split= [100_000, 3_000],
        batch_size= 16,
        num_workers= 2,
        pin_memory= False,
        transforms= VarialeSizeAugmenter(32, 32, 512),
        collate_fn= collator,
        sampler= VariableSizeSampler,
        h= 32,
        min_w= 32,
        max_w= 512
    )

    dataModule.setup()



    for _ in tqdm(dataModule.train_dataloader(), desc= 'test running train dm'):
        pass
    for _ in tqdm(dataModule.val_dataloader(), desc= 'test running val dm'):
        pass
    for _ in tqdm(dataModule.test_dataloader(), desc= 'test running test dm'):
        pass

    # print(sample['filename'])
    # print(sample['tgt_input'])
    # print(sample['tgt_output'])
    # print(sample['tgt_input'].shape)
    # print(sample['tgt_output'].shape)

    # print(tokenizer.batch_decode(sample['tgt_output']))

    # print(sample)
    # print(sample['img'].shape)
    # print(next(iter(dataModule.test_dataloader())))