import numpy as np


def points_traverse(fun):
    def wrapper(points, *args, **kwargs):

        def wrp(ptts):
            for pt in ptts:
                if all([isinstance(x, float) for x in pt]):
                    yield fun(pt, *args, **kwargs)
                else:
                    yield list(wrp(pt))

        return list(wrp(points))

    return wrapper


def parse_attribute(attr):
    arr = np.array(attr.array)
    shp = int(len(arr) / attr.itemSize), attr.itemSize
    arr.reshape(shp)
    return arr



def is_geometry(obj):
    if isinstance(obj, dict):
        return "geometry" in obj.keys()

    else:
        return hasattr(obj, "geometry")


def pop_in_keys(dct, key):
    if key in dct.keys():
        return dct.pop(key)


"""
def create_object_from_threejs(kwargs):
    def trav(obj):
        cls = getattr(sys.modules['__main__'], obj.pop("type"), __default=Object3D)

        geometry = pop_in_keys("geometry")

        if isinstance(cls, GeometryObject):
            material = pop_in_keys("material")
            children = pop_in_keys("children")
            return cls.from_three(obj, geometry, material)
        if "children" in obj.keys() and (len(obj["children"]) > 0):
            children = obj.pop("children")
            cls(**obj)
            for child in obj.children: ...
"""

"""
def dump_model(data: dict, path=os.getenv("HOME") + os.getenv("MICROVIEWER_PATH")):
    with open(path, "w") as f:
        json.dump(data, f)


from functools import wraps


def debug_view(fn):
    @wraps(fn)
    def wrapper(*arga, **kwargs):
        model = fn(*arga, **kwargs)
        dump_model(model)
        return model
    return wrapper"""
