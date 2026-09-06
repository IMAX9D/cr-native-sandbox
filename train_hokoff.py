"""AutoDL preset: run `python train_hokoff.py` for a 1000-step LSTM trial.

Optional trainer flags override the preset, e.g. --workers 4.
Use --dry-run to display the resolved arguments without starting training.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

from hokoff_model.train import parser, run


# Edit these paths if the server dataset is moved.
DATA = "/root/autodl-tmp/expert-dataset/native-bc-v1"
CACHE = "/root/autodl-tmp/policy-v1-cache"
RUNS = Path("/root/autodl-tmp/runs")


def main(argv=None):
    # Each invocation starts a fresh trial; previous checkpoints stay intact.
    run_dir = RUNS / ("hokoff-lstm-check-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    defaults = {
        "data": DATA,
        "cache": CACHE,
        "run-dir": run_dir,
        # Historical dataset: the large training split is named validation.
        "train-split": "validation",
        "val-split": "train",
        "device": "cuda",
        "precision": "fp16",
        "width": 256,
        "hidden-size": 512,
        "frame-window": 128,
        "targets": 32,
        "batch-size": 32,
        "workers": 8,
        "epochs": 1,
        "max-steps": 1000,
        "log-every": 100,
        "save-every": 500,
        "eval-batches": 100,
    }
    arguments = []
    for name, value in defaults.items():
        arguments.extend(["--" + name, str(value)])
    arguments.extend(sys.argv[1:] if argv is None else argv)
    p = parser()
    p.description = __doc__
    p.add_argument("--dry-run", action="store_true", help="show configuration only")
    args = p.parse_args(arguments)
    dry_run = args.dry_run
    del args.dry_run
    print(json.dumps({"phase": "launcher", "arguments": vars(args)}, default=str), flush=True)
    if not dry_run:
        run(args)


if __name__ == "__main__":
    main()
