import copy
from mmcore.addons.compute import StandardGrasshopperRequest

class Filter:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, seq, inv=False):
        nm = iter(copy.deepcopy(self.mask))
        if inv:

            return filter(lambda x: not next(nm), iter(seq))
        else:

            return filter(lambda x: next(nm), iter(seq))
class BrepMask(StandardGrasshopperRequest):
    def __init__(self, mask=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.mask=mask
    def __call__(self, seq=None):
        dfl=self.defaults
        if self.mask is None:

            self.mask==dfl["mask"]

        if seq==None:
            seq == dfl["geoms"]
        return Filter(self.post(data={"geoms":seq, "mask":self.mask})["inside"])







