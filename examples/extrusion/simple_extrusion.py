from mmcore.geom.shapes import Shape
from mmcore.geom.extrusion import simple_extrusion
import json

HIGH = 2

if __name__ == "__main__":
    with open("examples/data/profile.json") as f:
        profile = json.load(f)
    shape = Shape(boundary=profile["bounds"], holes=profile["holes"])
    extrusion = simple_extrusion(shape, HIGH)
    with open("model.json", "w") as f:
        json.dump(extrusion.root(), f) # now you can view it with three js
