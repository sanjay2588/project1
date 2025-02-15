import pathlib

# Get the parent of the current file (core), which is the project root.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def get_data_dir() -> pathlib.Path:
    """Gets the path to the data directory."""
    return PROJECT_ROOT / "data" 