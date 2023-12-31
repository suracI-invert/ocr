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
from src.utils.transforms import Resize, ToTensor, SwinAugmenter, DefaultAugmenter, AlbumentationsTransform, AlbumentationsWithPadding, VITAugmenter
from torchvision.transforms import Compose, RandomChoice, AugMix, AutoAugment

if __name__ == '__main__':
    set_float32_matmul_precision('medium')

    # cnn_args = {
    #     'arch': 'vgg',
    #     'weights': 'IMAGENET1K_V1',
    #     'ss': [
    #         [2, 2],
    #         [2, 2],
    #         [2, 1],
    #         [2, 1],
    #         [1, 1]
    #     ],
    #     'ks': [
    #         [2, 2],
    #         [2, 2],
    #         [2, 1],
    #         [2, 1],
    #         [1, 1]
    #     ],
    #     'hidden': 256
    # }

    cnn_args = {
        'arch': 'resnet50',
        'ss': [
            [2, 2],
            [2, 1],
            [2, 1],
            [2, 1],
            [1, 1]
        ],
        'hidden': 256
    }

    vit_args = {
        'arch': 'swin',
        'hidden': 256,
        'dropout': 0.2,
        'pretrained': 'microsoft/swin-tiny-patch4-window7-224'
    }

    # vit_args = {
    #     'arch': 'vit',
    #     'hidden': 256,
    #     'dropout': 0.1,
    #     'pretrained': 'google/vit-base-patch32-384'
    # }

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
        'patience': 2,
        'threshold': 0.01,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 0,
        'eps': 1e-8,
        'verbose': True,
    }

    tokenizer = Tokenizer(trans_args['max_seq_length'])
    collator = Collator()

    Augmenter = SwinAugmenter(vit_args['pretrained'])
    # Augmenter = AlbumentationsTransform((70, 140))
    # Augmenter = AlbumentationsWithPadding((224, 112))
    # Augmenter = VITAugmenter(vit_args['pretrained'])

    dataModule = OCRDataModule(
        data_dir= './data/', map_file= 'train_line.txt',
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

    net = Net(len(tokenizer.chars), 'transformers', vit_args, trans_args)

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

    logger = loggers.TensorBoardLogger(save_dir= './log/', log_graph= True)
    # logger = loggers.WandbLogger(save_dir="./log", project="Neuromancers-OCR")
    ckpt_callback = ModelCheckpoint(dirpath= './weights/', 
                                    filename= 'vietocr_resnet_{epoch:02d}_{val_cer:0.2f}',
                                    monitor= 'val_cer', 
                                    save_on_train_epoch_end= True,
                                    save_top_k= 1
                                )
    lr_monitor = LearningRateMonitor('step', True)
    es_callback = EarlyStopping('val_loss', min_delta= 0.0001, patience= 2)
    # dstat_callback = DeviceStatsMonitor(cpu_stats= True)

    profiler = AdvancedProfiler(dirpath="./log/profiler", filename="perf_logs")

    trainer = Trainer(accelerator= 'gpu',
                      precision= '16-mixed', 
                    #   max_time= '00:00:02:00',
                      max_epochs= 25, 
                      benchmark= True,
                      logger= logger,
                      profiler= profiler,
                      log_every_n_steps= 5,
                      check_val_every_n_epoch= 1,
                      num_sanity_val_steps= 2,
                      callbacks=[ckpt_callback, lr_monitor, es_callback]
                    )
    # setup tuner
    # tuner = Tuner(trainer)

    # got batch_size 768 => pruned
    # tuner.scale_batch_size(OCRModel, datamodule= dataModule, mode= 'binsearch')
    # found lr = 8.317637711026709e-05
    # tuner.lr_find(OCRModel, datamodule= dataModule)
    print('Training started')
    
    trainer.fit(OCRModel, datamodule= dataModule)

    print('DONE!')