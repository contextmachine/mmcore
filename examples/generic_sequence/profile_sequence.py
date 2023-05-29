import numpy as np

from mmcore.geom.shapes import Shape
from mmcore.geom.extrusion import simple_extrusion
import json
from mmcore.base.sharedstate import serve

HIGH = 10
with open("examples/data/profile.json") as f:
    profile = json.load(f)
shape = Shape(boundary=profile["bounds"], holes=profile["holes"])

extr_hor = simple_extrusion(shape, HIGH)

extr_ver = simple_extrusion(shape, HIGH)
extr_ver.rotate(np.pi / 2, [0, 1, 0])
extr_ver.translate([0, 0.07, 0])

def bim_sequence_generator(extr, transl=(1, 0, 0)):
    for i in range(10):
        next_extrusion = extr.__copy__()
        next_extrusion.translate(transl)
        yield next_extrusion


if __name__ == "__main__":
    from mmcore.base import AGroup
    group = AGroup()

    for bim in bim_sequence_generator(extr_hor):
        group.add(bim)

    for bim in bim_sequence_generator(extr_ver,transl=(-1, 0, 0)):
        group.add(bim)

    with open("model.json", "w") as f:
        json.dump(group.root(), f)  # now you can view it with three js
    serve.start_as_main()