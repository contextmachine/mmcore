import click
import dill
import requests


@click.group("mmcore")
def mmcli():
    ...


@click.command("load")
@click.argument("source")
def load(source="asset.pkl"):
    f = source
    global asset
    if isinstance(f, str):
        if f.startswith("http"):
            asset = dill.loads(requests.get(f).raw)
            return asset
        with open(f, "rb") as fl:
            asset = dill.load(fl)
            return asset
    else:
        asset = dill.load(f)
        return asset
