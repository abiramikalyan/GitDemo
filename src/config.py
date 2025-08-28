from dataclasses import dataclass
from typing import Dict, Any
import yaml


@dataclass
class Config:
    data: Dict[str, Any]


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(data=cfg)


def get(cfg: Config, key: str, default=None):
    return cfg.data.get(key, default)
