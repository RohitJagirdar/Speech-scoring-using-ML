"""
Command-line entry point.

    python main.py                  # run whatever is out of date, then verify
    python main.py --status         # show each stage's state, run nothing
    python main.py --stage train    # re-run one stage, then verify
    python main.py --force          # re-run everything (slow: Whisper + wav2vec2)
"""
import argparse

from src.pipeline import STAGE_NAMES, print_status, run_pipeline
from src.verify import run_full_pipeline_verification


def main() -> int:
    """Parse the command line, run the pipeline, return the exit code."""
    parser = argparse.ArgumentParser(description="Spoken grammar scoring pipeline")
    parser.add_argument("--status", action="store_true", help="show stage status and exit")
    parser.add_argument("--stage", choices=STAGE_NAMES, help="run only this stage")
    parser.add_argument("--force", action="store_true", help="re-run stages even if up to date")
    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    run_pipeline(force=args.force, only=args.stage)
    return 0 if run_full_pipeline_verification() else 1


if __name__ == "__main__":
    raise SystemExit(main())
