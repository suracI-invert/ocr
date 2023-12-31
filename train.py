from src.models.model import Net
from src.models.lit_module import OCRLitModule
from src.models.tokenizer import Tokenizer
from src.data.datamodule import OCRDataModule
from src.data.components.collator import Collator
from src.data.components.sampler import VariableSizeSampler
from src.utils.transforms import get_augmenter
from src.utils.config import Config
from src.utils.callbacks import get_callbacks
from src.utils.logger import get_logger
from src.utils.profiler import get_profiler
from src.utils.optim import get_lr_scheduler, get_optimizer

from torch import set_float32_matmul_precision

from lightning import Trainer

if __name__ == '__main__':
    cfg = Config.load_config_from_file('./config/config.yaml')
    if cfg['misc']['set_precision_matmul']:
        set_float32_matmul_precision(cfg['misc']['precision'])
    
    tokenizer = Tokenizer(cfg['transformer']['max_seq_length'])
    collator = Collator()
    Augmenter = get_augmenter(cfg)
    sampler = VariableSizeSampler if cfg['backbone']['type'] == 'cnn' else None
    # Sampler (with convert True alledgely) introduce error while training/ might need further changes or revert back to fixed resize

    dataModule = OCRDataModule(
        data_dir= cfg['data']['data_dir'],
        map_file= cfg['data']['map_file'],
        tokenizer= tokenizer,
        test_dir= cfg['data']['test_dir'],
        train_val_split= cfg['data']['train_val_split'],
        batch_size= cfg['data']['batch_size'],
        num_workers= cfg['data']['num_workers'],
        pin_memory= cfg['data']['pin_memory'],
        transforms= Augmenter,
        collate_fn= collator,
        sampler= sampler,
        h= cfg['transform']['h'],
        min_w= cfg['transform']['min_w'],
        max_w= cfg['transform']['max_w']
    )

    dataModule.setup()

    net = Net(len(tokenizer.chars), cfg['backbone']['type'], cfg['backbone']['arg'], cfg['transformer'])

    lr_scheduler = get_lr_scheduler(cfg)
    optimizer = get_optimizer(cfg)

    OCRModel = OCRLitModule(
        net, tokenizer,
        optimizer= optimizer['optimizer'],
        optimizer_params= optimizer['args'],
        scheduler= lr_scheduler['lr_scheduler'],
        scheduler_params= lr_scheduler['args'],
        extra_scheduler_params= lr_scheduler['extra_args']
    )

    callbacks = get_callbacks(cfg)
    logger = get_logger(cfg)
    profiler = get_profiler(cfg)

    if cfg['logger']['type'] == 'wandb' and cfg['logger']['watch']:
        logger.watch(net, log= 'all')

    trainer = Trainer(
        accelerator= cfg['trainer']['accelerator'],
        precision= cfg['trainer']['precision'],
        max_epochs= cfg['trainer']['max_epochs'],
        benchmark= cfg['trainer']['benchmark'],
        log_every_n_steps= cfg['trainer']['log_every_n_steps'],
        check_val_every_n_epoch= cfg['trainer']['check_val_every_n_epoch'],
        num_sanity_val_steps= cfg['trainer']['num_sanity_val_steps'],
        callbacks= callbacks,
        logger= logger,
        profiler= profiler
    )

    trainer.fit(OCRModel, train_dataloaders= dataModule.train_dataloader(), val_dataloaders= dataModule.val_dataloader())
