import IPython
import click
import dill
import requests
import rich


@click.group("assets")
def mmcli():
    ...


@click.command()
@click.argument("source")
@click.option("--ipy", is_flag=True, default=0)
def load(source, ipy):
    f = source
    global asset
    if isinstance(f, str):
        if f.startswith("http"):
            asset = dill.loads(requests.get(f).raw)

        with open(f, "rb") as fl:
            asset = dill.load(fl)

    else:
        asset = dill.load(f)
    rich.print(asset, asset.todict())
    if bool(ipy):
        IPython.embed()
    return asset


if __name__ == "__main__":
    from IPython import embed

    asset = load()
