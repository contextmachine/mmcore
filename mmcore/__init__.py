#Generated with ./bin/upd-version.py
from mmcore.utils.env import load_dotenv_from_path, load_dotenv_from_stream

load_dotenv_from_path(".env")

TOLERANCE = 0.000_001


def __version__():
    return "0.23.5"
