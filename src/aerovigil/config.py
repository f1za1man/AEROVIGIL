"""Central configuration loader for AEROVIGIL."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load project configuration from a YAML file."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    settings = load_config()

    print("Configuration loaded successfully.")
    print(f"Project: {settings['project']['name']}")
    print(f"Version: {settings['project']['version']}")
    print(f"Random seed: {settings['environment']['random_seed']}")
