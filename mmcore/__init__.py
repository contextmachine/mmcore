#Generated with ./bin/upd-version.py
import numpy as np

from mmcore.utils.env import load_dotenv_from_path, load_dotenv_from_stream
np.set_printoptions(suppress=True)
load_dotenv_from_path(".env")

TOLERANCE = 0.000_001


def __version__():
    return "0.18.12"
