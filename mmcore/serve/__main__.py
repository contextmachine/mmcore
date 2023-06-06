from mmcore.mastermodel import MasterModel
import click


@click.command
@click.option("-c", default='', help='exec command')
@click.option("--serve-start", default=True, help='Start Base Server Server')
@click.option("--rpyc-start", default=False, help='Start RPyC Server')
def start(c=None, serve_start=True, rpyc_start=False, *args, **kwargs):
    mm = MasterModel()
    if serve_start:
        mm.start()
    if rpyc_start:
        mm.stop_rpyc()
    if c is not None:

        mm.embed(c, *args, **kwargs)
    else:
        mm.embed( *args, **kwargs)


if __name__ == "__main__":
    start()
