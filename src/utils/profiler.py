from lightning.pytorch import profilers

def get_profiler(cfg):
    if cfg['profiler']['type'] == 'advanced':
        return profilers.AdvancedProfiler(cfg['profiler']['dirpath'], cfg['profiler']['filename'])
    if cfg['profiler']['type'] == 'simple':
        return profilers.SimpleProfiler(cfg['profiler']['dirpath'], cfg['profiler']['filename'])
    return None