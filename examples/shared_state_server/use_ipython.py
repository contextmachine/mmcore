# run this example in python console
from mmcore.base.sharedstate import serve
import json

from mmcore.geom.extrusion import simple_extrusion
from mmcore.geom.shapes import Shape

# Server run in non-blocked thread on http://localhost:7711
# It dynamically update when you will create a new objects or change exist
serve.start()

HIGH = 2

with open("examples/data/profile.json") as f:
    profile = json.load(f)
shape = Shape(boundary=profile["bounds"], holes=profile["holes"])
extrusion = simple_extrusion(shape, HIGH)

# Now, interact with extrusion model in console. Yo can transform , change color and more ...
