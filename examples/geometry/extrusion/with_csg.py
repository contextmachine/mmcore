import json

from mmcore.geom.extrusion import csg_extrusion
from mmcore.geom.shapes import LegacyShape

HIGH = 200

if __name__ == "__main__":
    with open("examples/data/profile.json") as f:
        profile = json.load(f)
    shape = LegacyShape(boundary=[[pp * 1000 for pp in p] for p in profile["bounds"]],
                        holes=[[[ppp * 1000 for ppp in pp] for pp in p] for p in profile["holes"]])
    extrusion = csg_extrusion(shape, HIGH)
    with open("model2.json", "w") as f:
        json.dump(extrusion.root(), f) # now you can view it with three js

