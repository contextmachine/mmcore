import copy
import json
import warnings

from mmcore.api import *
import ast
from typing import get_args

import inspect


def generate_schema():
    import mmcore.api

    dct = dict()
    for k, i in mmcore.api.__dict__.items():
        if inspect.isclass(i) and hasattr(i, "__create_accept_args__"):
            dat = {
                "type": i.__name__,
                "spec": i.__create_accept_args__.spec.annotations,
            }
            dct[k] = dat

    return dct


from typing import get_args

from collections import namedtuple, Counter

SchemaTypeEntity = namedtuple("SchemaTypeEntity", ["typ", "leaf"])
SchemaType = namedtuple("SchemaBufferType", ["type", "spec"])


def generate_traverse_schema(obj, schema):
    typ = eval(obj)
    args = get_args(typ)

    if args:
        origin = get_origin(typ)
        leafs = []
        dct = {"type": obj, "spec": []}
        isleaf = True
        for arg in args:
            _ = generate_traverse_schema(arg.__name__, schema)
            if _.leaf:
                leafs.append(_.typ)
            else:
                leafs.extend(_.typ["leafs"])
                isleaf = False
            dct["spec"].append(_.typ)
        dct["leafs"] = leafs
        dct["leafsCounter"] = Counter(leafs)
        if isleaf:
            return SchemaTypeEntity(obj, isleaf)
        else:
            return SchemaTypeEntity(dct, isleaf)

    elif obj in schema:
        spec = copy.deepcopy(schema[obj]["spec"])
        if "return" in spec:
            del spec["return"]
        leafs = []
        dct = {"type": obj, "spec": {}}
        for k, v in spec.items():
            _ = generate_traverse_schema(v, schema)
            if _.leaf:
                leafs.append(_.typ)
            else:
                leafs.extend(_.typ["leafs"])
            dct["spec"][k] = _.typ

        dct["leafs"] = leafs
        dct["leafsCounter"] = Counter(leafs)
        return SchemaTypeEntity(dct, False)
    else:
        return SchemaTypeEntity(obj, True)


def dump_schemas():
    schema = generate_schema()
    gschema = {k: generate_traverse_schema(k, schema).typ for k in schema.keys()}

    with open("schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    with open("schemaTraversed.json", "w") as f:
        json.dump(gschema, f, indent=2)
