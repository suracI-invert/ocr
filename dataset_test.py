from src.data.components.collator import Collator
from src.models.tokenizer import Tokenizer
from src.data.datamodule import OCRDataModule
from src.utils.transforms import Resize, ToTensor, Augmenters
from torchvision.transforms import Compose

tokenizer = Tokenizer()
collator = Collator()

dataModule = OCRDataModule(
    data_dir= './data/', map_file= 'train_annotation.txt',
    test_dir= './data/new_public_test',
    tokenizer= tokenizer,
    train_val_split= [100_000, 3_000],
    batch_size= 2,
    num_workers= 0,
    pin_memory= False,
    transforms= Compose([Resize(70, 140), ToTensor()]),
    collate_fn= collator,
    sampler= None
)

dataModule.setup()

print(len(dataModule.data_train))
print(len(dataModule.data_valid))
print(len(dataModule.data_test))

sample = next(iter(dataModule.train_dataloader()))

# print(sample['filename'])
# print(sample['tgt_input'])
# print(sample['tgt_output'])
# print(sample['tgt_input'].shape)
# print(sample['tgt_output'].shape)

# print(tokenizer.batch_decode(sample['tgt_output']))

print(sample)
print(next(iter(dataModule.test_dataloader())))