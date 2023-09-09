from src.models.model import Net
from src.models.lit_module import OCRLitModule

from torch import set_float32_matmul_precision, rand
from torch.optim import AdamW, lr_scheduler
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.profilers import AdvancedProfiler

from src.data.components.collator import Collator
from src.models.tokenizer import Tokenizer
from src.data.datamodule import OCRDataModule
from src.utils.transforms import Resize, ToTensor, SwinAugmenter
from torchvision.transforms import Compose, RandomChoice, AugMix, AutoAugment

if __name__ == '__main__':
    set_float32_matmul_precision('medium')

    cnn_args = {
        'weights': 'IMAGENET1K_V1',
        'ss': [
            [2, 2],
            [2, 2],
            [2, 1],
            [2, 1],
            [1, 1]
        ],
        'ks': [
            [2, 2],
            [2, 2],
            [2, 1],
            [2, 1],
            [1, 1]
        ],
        'hidden': 256
    }

    swin_args = {
        'hidden': 256,
        'dropout': 0.2,
        'pretrained': 'microsoft/swin-tiny-patch4-window7-224'
    }

    trans_args = {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "max_seq_length": 512,
        "pos_dropout": 0.2,
        "trans_dropout": 0.1
    }

    scheduler_params = {
        'mode': 'min',
        'factor': 0.1,
        'patience': 3,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 0,
        'eps': 1e-8,
        'verbose': True,
    }

    tokenizer = Tokenizer()
    collator = Collator()

    Augmenter = SwinAugmenter(swin_args['pretrained'])

    dataModule = OCRDataModule(
        data_dir= './data/', map_file= 'train_annotation.txt',
        test_dir= './data/new_public_test',
        tokenizer= tokenizer,
        train_val_split= [100_000, 3_000],
        batch_size= 64,
        num_workers= 6,
        pin_memory= True,
        transforms= Augmenter,
        collate_fn= collator,
        sampler= None
    )

    dataModule.setup()

    net = Net(len(tokenizer.chars), 'swin', swin_args, trans_args)

    train_loader = dataModule.train_dataloader()
    val_loader = dataModule.val_dataloader()
    # test_loader = dataModule.test_dataloader()

    OCRModel = OCRLitModule(net, 
                            tokenizer, 
                            AdamW, 
                            lr_scheduler.ReduceLROnPlateau, 
                            learning_rate= 8.317637711026709e-05,
                            scheduler_params= scheduler_params,
                            monitor_metric= 'val_loss',
                            interval= 'epoch',
                            frequency= 3,
                            # example_input_array= next(iter(train_loader))
                        )
    
    print(next(iter(train_loader)))