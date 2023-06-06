import copy

from mmcore.base.sharedstate import SharedStateServer

import IPython
from mmcore import __version__
from mmcore.base import deep_merge


class MasterModel(SharedStateServer):
    def __init__(self):
        self._ipython = None

    def __new__(cls, *args, header="MasterModel", **kwargs):
        inst = super().__new__(cls, *args, header=f'[mmcore {__version__()}]\n{header}API \n', **kwargs)
        return inst

    def embed(self, command, *args, **kwargs):
        exec(command, globals(), locals())
        IPython.embed(header=self.header,  *args, **kwargs)

    def __setstate__(self, state):
        reg = state.get("registry")
        from mmcore.base.registry import adict, ageomdict, amatdict
        adict |= reg["object"]
        ageomdict |= reg["geometries"]
        amatdict |= amatdict["materials"]

        for k, v in deep_merge(self.__dict__, state.get("model") if state.get("model") is not None else {}).items():
            setattr(self, k, v)



