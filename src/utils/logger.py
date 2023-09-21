from lightning.pytorch import loggers

def get_logger(cfg):
    if cfg['logger']['type'] == 'tensorboard':
        return loggers.TensorBoardLogger(save_dir= cfg['logger']['save_dir'])
    if cfg['logger']['type'] == 'wandb':
        return loggers.WandbLogger(
            save_dir= cfg['logger']['save_dir'], 
            project= cfg['logger']['save_dir'], 
            log_model= cfg['logger']['log_model']
            )
    return False