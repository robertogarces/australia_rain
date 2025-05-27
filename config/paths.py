from pathlib import Path

# Get the route of the actual directory
ROOT_PATH = Path(__file__).resolve().parent.parent

ARTIFACTS_PATH = ROOT_PATH / "artifacts"
CONFIG_PATH = ROOT_PATH / 'config'
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
FINAL_DATA_PATH = DATA_PATH / "final"
MODELS_PATH = ARTIFACTS_PATH / "models"
NOTEBOOKS_PATH = ROOT_PATH / "notebooks"
SRC_PATH = ROOT_PATH / "src"
TESTS_PATH = ROOT_PATH / "tests"
UTILS_PATH = ROOT_PATH / "utils"
PLOTS_PATH = ARTIFACTS_PATH / 'plots'
REPORTS_PATH = ARTIFACTS_PATH / 'reports'