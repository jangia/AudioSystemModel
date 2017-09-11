import yaml


def load_config():
    with open("model_configs.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError:
            raise yaml.YAMLError
