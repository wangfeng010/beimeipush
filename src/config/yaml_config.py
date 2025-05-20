import yaml
from pydantic import BaseModel
from typing import Type

from src.config.base_config import AppConfig


def load_config_from_yaml(file_path: str, config_model: Type[BaseModel]) -> BaseModel:
    """Loads configuration from a YAML file into a Pydantic model."""
    with open(file_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return config_model(**config_data)


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    cur_path = Path.cwd() / "config" / "config.yml"
    config = load_config_from_yaml(cur_path, AppConfig)
    pprint(config)
