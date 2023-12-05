import dataclasses
from operator import attrgetter, itemgetter, methodcaller

import mmcore.base
from mmcore.base.models import gql
from mmcore.base.registry import matdict
from mmcore.collections import ElementSequence


def merge(list_of_dicts):
    return {k: v for dictionary in list_of_dicts for k, v in dictionary.items()}
def methods_owner(method):
    own = [k for k in method.__qualname__.split('.')[:-1]]
    if len(own) > 1:
        own = '.'.join(own)
    else:
        own, = own
    return globals()[own]

def getitemattrs(*attrs):
    def wrap(obj):
        return [getitemattr(attr)(obj) for attr in attrs]

    return wrap
def getitemattr(attr):
    def wrp(obj):
        errs = []
        try:
            return attrgetter(attr)(obj)
        except AttributeError as err1:
            errs.append(err1)
            return itemgetter(attr)(obj)
        except ExceptionGroup("", [KeyError(1), TypeError(2)]) as err2:
            errs.append(err2)
            return methodcaller(attr)(obj)
        except Exception as err3:
            errs.append(err3)
            raise errs

    return wrp


def generate_edges_material(uid, color, linewidth):
    return gql.LineBasicMaterial(
        **{"uuid": uid, "type": "LineBasicMaterial", "color": color.decimal, "vertexColors": True, "depthFunc": 3,
           "depthTest": True, "depthWrite": True, "colorWrite": True, "stencilWrite": False, "stencilWriteMask": 255,
           "stencilFunc": 519, "stencilRef": 0, "stencilFuncMask": 255, "stencilFail": 7680, "stencilZFail": 7680,
           "stencilZPass": 7680, "linewidth": linewidth, "toneMapped": False})

def to_camel_case(name: str):
    """
    Ключевая особенность, при преобразовании имени начинающиегося с подчеркивания, подчеркивание будет сохранено.

        foo_bar -> FooBar
        _foo_bar -> _FooBar
    @param name: str
    @return: str
    """
    if not name.startswith("_"):

        return "".join(nm[0].capitalize() + nm[1:] for nm in name.split("_"))

    else:
        return "_" + "".join(nm[0].capitalize() + nm[1:] for nm in name.split("_"))



def export_edgedata_to_json(edge_hash, point_set):
    """Export a set of points to a LineSegment buffergeometry"""
    # first build the array of point coordinates
    # edges are built as follows:
    # points_coordinates  =[P0x, P0y, P0z, P1x, P1y, P1z, P2x, P2y, etc.]

    points_coordinates = []
    for point in point_set:
        for coord in point:
            points_coordinates.append(coord)
    # then build the dictionary exported to json
    edges_data = {
        "uuid": edge_hash,
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "position": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": points_coordinates,
                }
            }
        },
    }
    return edges_data


def generate_material(self):
    vv = list(matdict.values())
    if len(vv) > 0:
        ##print(vv)
        es = ElementSequence(vv)
        ##print(self.color, es["color"])
        if self.color.decimal in es["color"]:
            i = es["color"].index(self.color.decimal)
            ##print(i)
            vvv = es._seq[i]
            ##print(vvv)
            self.mesh._material = vvv.uuid
        else:
            self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                          color=self.color.decimal)

    else:
        self.mesh.material = mmcore.base.models.gql.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                      color=self.color.decimal)


def deep_merge(dct, dct2):
    for k, v in dct.items():
        vv = dct2.get(k)
        if vv is not None:
            if isinstance(vv, dict):
                if isinstance(v, dict):
                    deep_merge(v, vv)
                else:
                    dct[k] = vv
            else:
                dct[k] = vv
    for k, v2 in dct2.items():
        if not (k in dct.keys()):
            dct[k] = v2
    return dct


def asdict(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    elif isinstance(obj, (list, tuple, set)):
        return [asdict(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: asdict(o) for k, o in obj.items()}
    else:
        return obj
