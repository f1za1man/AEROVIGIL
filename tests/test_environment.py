"""Tests for the AEROVIGIL development environment."""

import numpy as np
import pandas as pd

from aerovigil import __version__
from aerovigil.config import load_config
from aerovigil.reproducibility import set_global_seed


def test_project_version():
    assert __version__ == "0.1.0"


def test_configuration_loading():
    config = load_config()

    assert config["project"]["name"] == "AEROVIGIL"
    assert config["environment"]["random_seed"] == 42


def test_reproducibility():
    set_global_seed(42)
    first = np.random.rand(5)

    set_global_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_numpy_and_pandas():
    values = np.array([1, 2, 3])

    dataframe = pd.DataFrame({"values": values})

    assert len(dataframe) == 3
    assert dataframe["values"].sum() == 6
