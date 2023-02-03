import json
import os
import sys
from types import ModuleType

from pydantic import create_model

_m = ModuleType("_m")
sys.modules["mmcore.geom.gui._m"] = _m

LOCAL_PATH = "/".join(__file__.split("/")[:-1])
classes = []
for schema_file in os.scandir(f"{LOCAL_PATH}/assets/examples"):
    with open(schema_file.path) as fp:
        data = json.load(fp)
        GenModel = create_model("".join(schema_file.name.split(".")[:-1])[0:], **data)
        classes.append(GenModel)

print(classes)
