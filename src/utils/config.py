import yaml

class Config(dict):
    def __init__(self, config_dict):
        super(Config, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(path):
        base_config = {}
        with open(path, mode= 'r', encoding= 'utf8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)
        return Config(base_config)