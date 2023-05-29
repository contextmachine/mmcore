import numpy as np

from mmcore.geom.shapes import Shape
from mmcore.geom.extrusion import simple_extrusion
import json
from mmcore.base.sharedstate import serve
HIGH = 10
with open("examples/data/profile.json") as f:
    profile = json.load(f)
shape = Shape(boundary=profile["bounds"], holes=profile["holes"])
extrusion = simple_extrusion(shape, HIGH)


def bim_sequence_generator(count=150, rot_axis=(0, 1, 1), translate=(1, 0, 0)):
    for i in range(count):
        next_extrusion = extrusion.__copy__()
        next_extrusion.rotate(np.pi / 360, rot_axis)
        next_extrusion.translate(translate)
        yield next_extrusion


from mmcore.base import AGroup


class MyGroup(AGroup):
    _uuid = "sofya"
    def __call__(self, *args, **kwargs):
        for ch in self.children:
            ch.dispose()
        for bim in bim_sequence_generator(*args, **kwargs):
            self.add(bim)
        return super().__call__()


group = MyGroup()

if __name__ == "__main__":

    serve.start_as_main()