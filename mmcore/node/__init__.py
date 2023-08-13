import functools
import json
import subprocess as sp


def node_eval(fun):
    """

    Parameters
    ----------
    fun
    Выполните любой код с помощью node js
    Returns
    -------

    """
    def wrap(*args, **kwargs):
        proc = sp.Popen(["node", "-e", fun(*args, **kwargs)], stdout=sp.PIPE)
        res, _ = proc.communicate()
        return json.loads(res)

    return wrap


class NodeJs:
    def __init__(self, path="node"):
        super().__init__()
        self.path = path

    def __call__(self, fun):
        @functools.wraps(fun)
        def wrap(*args, **kwargs):
            proc = sp.Popen([self.path, "--import=three", "-e", fun(*args, **kwargs)], stdout=sp.PIPE)
            res, _ = proc.communicate()

            return self.decode(res)

        return wrap

    def decode(self, o):
        return json.loads(o)


class NodeJsType(NodeJs):

    def decode(self, o):
        def init(slf, **kwargs):
            slf.__dict__ |= kwargs

        dct = super().decode(o)
        classname = dct["type"]
        dct["__init__"] = init
        return type(classname, (object,), dct)(**dct)
