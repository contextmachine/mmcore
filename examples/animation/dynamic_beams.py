# run this example in python console
import numpy as np
import json
import time
from mmcore.base.sharedstate import serve
from mmcore.geom.extrusion import simple_extrusion
from mmcore.geom.materials import ColorRGB
from mmcore.geom.shapes import Shape

# Server run in non-blocked thread on http://localhost:7711
# It dynamically update when you will create a new objects or change exist
serve.start()

HIGH = 2

with open("examples/data/profile.json") as f:
    profile = json.load(f)
shape = Shape(boundary=profile["bounds"], holes=profile["holes"])
extrusion1 = simple_extrusion(shape, 1)
extrusion2 = simple_extrusion(shape, 2)
extrusion3 = simple_extrusion(shape, 3)
# Now, interact with extrusion model in console. You can transform , change color and more ...
extrusion1.translate([1, 0, 0])
extrusion2.translate([1, 1, 0])
extrusion3.translate([0, 1, 0])


for i in range(15000):
    time.sleep(0.0001)
    extrusion1.rotate(np.pi / 1000, [0, 1, 1])
    extrusion1.rotate(np.pi / 1000, [1, 0, 0])

    extrusion2.rotate(np.pi / 5000, [1, 0, 1])
    extrusion2.rotate(np.pi / 1000, [1, 1, 0])

    extrusion3.rotate(np.pi / 1000, [1, 0, 1])
    extrusion3.rotate(np.pi / 5000, [0, 1, 0])

    extrusion1.children[0].material.color = ColorRGB(int(np.abs(255 * np.cos(i))),
                                                     int(np.abs(255 * np.sin(i))),
                                                     int(np.abs(255 * np.tan(i)))).decimal
