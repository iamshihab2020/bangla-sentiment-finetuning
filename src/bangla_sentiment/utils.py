"""Shared helpers: project paths, seeding, the UTF-8 guard, run metadata and result files."""
import json
import locale
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"  # checkpoints and adapters, not committed


def ensure_utf8():
    """Fail fast if Python would read or write text files in a non-UTF-8 encoding.

    On Windows the default is cp1252, which corrupts Bangla text. Fix: set PYTHONUTF8=1.
    """
    encoding = locale.getpreferredencoding(False).lower().replace("-", "")
    if not sys.flags.utf8_mode and encoding != "utf8":
        raise RuntimeError(
            f"Python is not in UTF-8 mode (default file encoding is {encoding}). "
            "Set PYTHONUTF8=1 and open a new terminal."
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    """Current commit hash, with a '-dirty' suffix if there are uncommitted changes."""
    try:
        run = lambda *args: subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_metadata():
    """Machine and library details recorded with every result."""
    import bitsandbytes
    import peft
    import transformers

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }


def save_result(result, name):
    """Write one run's result to results/<name>.json with run metadata attached."""
    path = RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": run_metadata(), **result}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
