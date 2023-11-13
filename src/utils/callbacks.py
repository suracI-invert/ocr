from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def get_callbacks(cfg):
    ckpt_callback = ModelCheckpoint(
        cfg['checkpoint']['dirpath'],
        filename= cfg['checkpoint']['filename']+'_{epoch:0.2f}_{val_cer:0.3f}',
        monitor= cfg['checkpoint']['monitor'],
        save_on_train_epoch_end= cfg['checkpoint']['save_on_train_epoch_end'],
        save_top_k= cfg['checkpoint']['save_top_k'],
    )

    lr_monitor = LearningRateMonitor('step', True)
    es_callback = EarlyStopping(
        cfg['early_stopping']['monitor'],
        min_delta= cfg['early_stopping']['min_delta'],
        patience= cfg['early_stopping']['patience'],
        mode= cfg['early_stopping']['mode'],
    )

    return [ckpt_callback, lr_monitor, es_callback]