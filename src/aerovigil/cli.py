"""Command-line interface for AEROVIGIL."""

import argparse
import sys

import numpy as np
import pandas as pd
import sklearn

from aerovigil import __version__
from aerovigil.config import load_config
from aerovigil.reproducibility import set_global_seed


def run_smoke_test() -> None:
    """Verify that the initial environment is operational."""

    config = load_config()

    seed = config["environment"]["random_seed"]
    set_global_seed(seed)

    sample = np.random.rand(5)

    dataframe = pd.DataFrame(
        {
            "sample_id": range(1, 6),
            "random_value": sample,
        }
    )

    print("=" * 50)
    print("AEROVIGIL ENVIRONMENT SMOKE TEST")
    print("=" * 50)

    print(f"Project version: {__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Random seed: {seed}")

    print("\nSample data:")
    print(dataframe.to_string(index=False))

    print("\nAll core components initialized successfully.")
    print("SMOKE TEST PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AEROVIGIL project CLI")

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify the development environment",
    )

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
