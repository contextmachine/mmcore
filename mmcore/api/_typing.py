from __future__ import annotations
import sys

Self = object
import typing

if sys.version_info.minor < 11:
    typing.Self = object
from typing import *
