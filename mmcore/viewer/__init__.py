import collections
import functools

import more_itertools
import numpy as np

from mmcore.addons.rhino.compute import Mesh
from mmcore.baseitems import Matchable, descriptors
from mmcore.collections.multi_description import ElementSequence
from mmcore.geom.buffered import group_notation_from_mesh
from mmcore.geom.materials import MeshPhongFlatShading, ColorRGB
from mmcore.viewer.gui import chart

APP_ID = ""


def mesh_from_brep(x):
    return Mesh.CreateFromBrep(x["rhino_geometry"], multiple=True)


#  self.broker = RC(f'runtime:{APP_ID}:{self.__class__.__name__}:{self.uuid}:', conn=conn)
class ThreeJSGroup(ElementSequence, Matchable):
    gui = property(fget=lambda self: [
        chart.Chart("ceiling_type"),
        chart.Chart("zone"),
        chart.Chart("area"),

        {
            "type": "controls",
            "data": {
                "primary_tag": self.primary_tag,
            },
            "post": {
                "endpoint": f"https://api.contextmachine.online/api/{APP_ID}/{self.__class__.__name__.lower()}/{self.uuid}",
                "mutation": {
                    "scene": {
                        "where": {
                            "userData": {
                                "properties": {
                                    "": {
                                        "_eq": "original"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    ])

    properties = descriptors.UserDataProperties()
    userdata = descriptors.UserData()
    matrix = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

    # Это костыль, чтобы не было пересвета сцены

    def __init_subclass__(cls, item_type=None, **kwargs):
        cls.item_typ = item_type
        super().__init_subclass__(**kwargs)

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @property
    def materials(self, md):
        ...

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value

    @property
    def primary_tag(self):
        return self._primary_tag

    @primary_tag.setter
    def primary_tag(self, value):
        self._primary_tag = value
        self.generate_mat_dict()

    _breps = None

    @property
    def breps(self):
        return self._breps

    @breps.setter
    def breps(self, value):
        self._breps = value

    @property
    def geometries(self):
        return ElementSequence(self["mesh"])["buffer_geometry"]

    @property
    def geometries_uuids(self):
        return ElementSequence(self["mesh"])["uuid"]

    def generate_mat_dict(self):
        cnt = collections.Counter(self[self.primary_tag])

        if (self.primary_tag in self.broker[f"mat_dict"].keys()) and (
                len(cnt) == len(self.broker[f"mat_dict"][self.primary_tag].keys())):
            pass
        else:
            self.broker[f"mat_dict"] |= {
                self.primary_tag: dict(zip(cnt.keys(), [MeshPhongFlatShading(ColorRGB(*s)).data for s in
                                                        np.random.random((len(cnt), 3))]))}

    @functools.lru_cache(1024)
    def solve_mesh(self):
        return list(more_itertools.flatten(mesh_from_brep(self)))

    def get_geoms(self, uid):
        return self.geometries[self.geometries_uuids.index(uid)]

        # [mesh_area(ElementSequence(self.model_objs)) for d in dat]

    def get_geoms_by_obj(self, obj):
        return self.geometries[self.geometries_uuids.index(obj.mesh.uuid)]

    @property
    def children(self):
        return self["object"]

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def object(self):
        return group_notation_from_mesh(self.name, userdata=self.userdata,
                                        matrix=self.matrix,
                                        children=self.children, uid=self.uuid)

    @property
    def root(self):
        return {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Object3D.toJSON"
            },
            "geometries": self.geometries,
            "materials": self.materials,
            "object": self.object
        }

    @property
    def materials(self):

        return list(self.mat_dict.values())
