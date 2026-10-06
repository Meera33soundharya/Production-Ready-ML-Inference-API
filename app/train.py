"""Train and save the bundled Iris model."""

import argparse
import os
from pathlib import Path

from app.model import DEFAULT_MODEL_PATH, save_model, train_model


def main() -> None:
    """Train a classifier and write its artifact to ``MODEL_PATH``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="model artifact destination (defaults to MODEL_PATH)",
    )
    args = parser.parse_args()

    save_model(train_model(), args.output)
    print(f"Trained Iris model saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
