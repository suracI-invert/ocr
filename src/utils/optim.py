from torch import optim

def get_lr_scheduler(cfg):
    if cfg['scheduler']['type'] == 'reduceOnPlataeu':
        return {
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau,
            'args': cfg['scheduler']['arg'],
            'extra_args': cfg['scheduler']['extra']
        }
    return None

def get_optimizer(cfg):
    if cfg['optimizer']['type'] == 'adamw':
        return {
            'optimizer': optim.AdamW,
            'args': cfg['optimizer']['arg']
        }
    return None