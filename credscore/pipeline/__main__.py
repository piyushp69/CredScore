"""Command line entry point: `python -m credscore.pipeline <step> [options]`.

Steps:
  download   fetch the raw competition CSVs from Kaggle (skipped if present)
  features   raw CSVs -> dataset/features.parquet
  train      features -> models/ (model, metadata, evaluation report)
  insights   features -> models/insights.json (portfolio statistics)
  all        every step above, in order
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .. import config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m credscore.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "features", "train", "insights", "all"])
    parser.add_argument("--data-dir", type=Path, default=None, help="raw data + features (default: dataset/)")
    parser.add_argument("--model-dir", type=Path, default=None, help="model bundle output (default: models/)")
    parser.add_argument("--sample", type=int, default=None, help="use N random applicants (quick runs)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    data_dir = Path(args.data_dir or config.data_dir())
    model_dir = Path(args.model_dir or config.model_dir())
    started = time.perf_counter()

    from .build_features import build_features, load_features
    from .download import download
    from .insights import build_insights, save_insights
    from .train import TrainConfig, save, train

    steps = ["download", "features", "train", "insights"] if args.step == "all" else [args.step]
    base = None
    for step in steps:
        logging.info("=== %s ===", step)
        if step == "download":
            download(data_dir, force=args.force_download)
        elif step == "features":
            build_features(data_dir, sample=args.sample)
        elif step == "train":
            base = load_features(data_dir) if base is None else base
            cfg = TrainConfig(device=args.device, learning_rate=args.learning_rate, num_boost_round=args.max_rounds)
            model, report = train(base, cfg)
            save(model, report, model_dir)
        elif step == "insights":
            base = load_features(data_dir) if base is None else base
            save_insights(build_insights(base), model_dir)
    logging.info("Done in %.0fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
