#!/usr/local/env/python3
import os

import sys

sys.path.extend(["/".join(os.getcwd().split("/")[:-1]) if os.getcwd().endswith("bin") else os.getcwd()])
import click
import mmcore


@click.command()
@click.option('--main', is_flag=True, default=0, help='Update main')
@click.option('--major', is_flag=True, default=0, help='Update major')
@click.option('--minor', is_flag=True, default=0, help='Update minor')
@click.option('--lock', is_flag=True, default=0, help='Run poetry lock after update version')
def cli(main, major, minor, lock):
    prev = mmcore.__version__()
    _main, _major, _minor = prev.split(".")
    new_minor = int(_minor)
    new_major = int(_major)
    new_main = int(_main)
    print(main, major,minor)
    if bool(minor):


        new_minor=new_minor + 1
        new_major=new_major
        new_main=new_main
    if bool(major):
        new_minor = 0
        new_major=new_major + 1
        new_main = new_main
    if bool(main):
        new_minor = 0
        new_major = 0
        new_main = new_main + 1
    with open("mmcore/__init__.py", "w") as f:
        current = f"{new_main}.{new_major}.{new_minor}"
        print(f'{prev}->{current}')
        data = f"""#Generated with ./bin/upd-version.py
from mmcore.utils.env import load_dotenv_from_path, load_dotenv_from_stream

load_dotenv_from_path(".env")

TOLERANCE = 0.000_001


def __version__():
    return "{current}"
"""
        f.write(data)
    with open("pyproject.toml", "r") as fl:

        data = fl.read()

    with open("pyproject.toml", "w") as fll:
        current = f"{new_main}.{new_major}.{new_minor}"
        print(f'{prev}->{current}')

        fll.write(data.replace(prev, current))
    with open(".version", "w") as s2:
        s2.write(current)
    if lock:
        import subprocess as sp
        proc = sp.Popen(['poetry', 'lock'], stdout=sys.stdout, stderr=sys.stdout)
        proc.communicate()

if __name__ == "__main__":
    cli()
