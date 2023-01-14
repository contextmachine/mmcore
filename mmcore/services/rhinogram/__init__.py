import os
import sys
from subprocess import Popen


def popen():
    return


class RhinoGramService:
    def __init__(self, addr):
        self.addr = addr

    def __enter__(self):
        self.popen = Popen(["/bin/zsh", f"{os.getenv('PWD')}/mmcore/services/rhinogram/rhinogram.sh"],
                           stderr=sys.stdout,
                           stdout=sys.stdout).universal_newlines
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.popen.kill()
