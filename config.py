import os
import yaml


def load_config():
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), 'model_configs.yml')
    with open(CONFIG_PATH, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError:
            raise yaml.YAMLError
