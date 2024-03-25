#Generated with ./bin/upd-version.py
from pathlib import Path

from mmcore.utils.env import load_dotenv_from_path, load_dotenv_from_stream

load_dotenv_from_path(".env")

TOLERANCE = 0.000_001


def __version__():
    import toml
    with open((Path(__file__).parent/Path("../pyproject.toml")).resolve()) as f:
        data=toml.load(f)

    return  data['tool']['poetry']['version']
