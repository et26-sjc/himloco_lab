"""Read the actual Lab release metadata; no invented physical validation."""
import copy
from pathlib import Path

from go1_lab_deploy.himloco.config import load_config

RELEASE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = RELEASE_ROOT / "himloco.json"

def release_config():
    return copy.deepcopy(load_config(CONFIG_PATH))
