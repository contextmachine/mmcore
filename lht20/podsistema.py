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

rot_extrusion = simple_extrusion(shape, HIGH)
rot_extrusion.rotate(np.pi / 2, [1, 0, 0])

def bim_sequence_generator(extr=extrusion):
    for i in range(10):
        next_extrusion = extr.__copy__()
        #next_extrusion.rotate(np.pi / 360, [0, 1, 1])
        next_extrusion.translate([1, 0, 0])
        yield next_extrusion

if __name__ == "__main__":
    from mmcore.base import AGroup
    group = AGroup()

    for bim in bim_sequence_generator(extrusion):
        group.add(bim)
    for bim in bim_sequence_generator(rot_extrusion):
        group.add(bim)
    with open("model.json", "w") as f:
        json.dump(group.root(), f)  # now you can view it with three js
    serve.start_as_main()