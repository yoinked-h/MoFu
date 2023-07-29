from typing import Any
import toml
from pathlib import Path

class MoFUConfig():
    def __init__(self, config_file: str|Path = "config.toml"):
        if isinstance(config_file, str):
            self.config_file = Path(config_file)
        else:
            self.config_file = config_file
        self.config = toml.load(self.config_file)

    def __getitem__(self, key):
        return self.config[key]
    def __getattr__(self, __name: str) -> Any:
        return self.config[__name]