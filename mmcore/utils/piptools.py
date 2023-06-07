import importlib
import subprocess as sp
import sys

from mmcore.utils.termtools import TermColors, ColorStr, mmprint


def install(*packages):
    to_install=[]
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError as err:
            #print(err)
            to_install.append(package)
    if len(to_install)>0:
        mmprint(f"{ColorStr(f'{to_install}', color=TermColors.light_yellow).__str__()} will be installed.")
        proc=sp.Popen([sys._base_executable, "-m", "pip", "install"]+to_install)
        proc.communicate()
        proc.wait()
        proc.kill()
    else:
        mmprint(f"{ColorStr(f'Nothing', color=TermColors.light_yellow).__str__()} to install.")
