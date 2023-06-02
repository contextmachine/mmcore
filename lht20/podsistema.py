import numpy as np

from mmcore.geom.shapes import Shape
from mmcore.geom.extrusion import simple_extrusion
import json
from mmcore.base.sharedstate import serve

HIGH = 10
with open("examples/data/profile.json") as f:
    profile = json.load(f)

with open("examples/data/artem_profile.json") as f:
    artem_profile = json.load(f)



shape = Shape(boundary=profile["bounds"], holes=profile["holes"])
extrusion = simple_extrusion(shape, HIGH)

shape_profile = Shape(boundary=artem_profile["bounds"], holes=artem_profile["holes"])

extr_ver = simple_extrusion(shape_profile, HIGH)
extr_ver.rotate(np.pi / 2, [1, 0, 0])

def bim_sequence_generator(extr=extrusion, transl=(1, 0, 0)):
    for i in range(10):
        next_extrusion = extr.__copy__()
        next_extrusion.translate(transl)
        yield next_extrusion

if __name__ == "__main__":
    from mmcore.base import AGroup
    group = AGroup()

    for bim in bim_sequence_generator(extrusion):
        group.add(bim)
    for bim in bim_sequence_generator(extr_ver, transl=(-1, 0, 0)):
        group.add(bim)

    with open("model.json", "w") as f:
        json.dump(group.root(), f)  # now you can view it with three js
    serve.start_as_main()