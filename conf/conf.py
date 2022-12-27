import logging
from dynaconf import Dynaconf

logging.basicConfig(level=logging.INFO)

# It depends on the relative path of entrypoint file
settings = Dynaconf(settings_file='conf/setting.toml')
