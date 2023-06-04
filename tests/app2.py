import copy
from collections import namedtuple

from localparams import *
from mmcore.collections import DoublyLinkedList
from mmcore.geom.parametric import PlaneLinear, HypPar4ptGrid
from mmcore.geom.parametric.algorithms import ClosestPoint

from mmcore.geom.transform import Transform
from mmcore.geom.vectors import unit
import multiprocess as mp
from mmcore.base.models.gql import MeshPhongMaterial

from dataclasses import InitVar

from mmcore.gql.client import GQLReducedQuery

data_point_query = GQLReducedQuery(
    # language=GraphQL
    """
    query MyQuery($id: Int! ) {
      panels_points_by_pk(id: $id) {
        arch_type
        eng_type
        id
        subzone
        x
        y
        z
        zone
      }
    }
    """
)


@dataclasses.dataclass
class DataPoint:
    id: InitVar[int]

    x: typing.Optional[float] = None
    y: typing.Optional[float] = None
    z: typing.Optional[float] = None
    subzone: typing.Optional[str] = None
    zone: typing.Optional[str] = None
    eng_type: typing.Optional[int] = None
    arch_type: str = "A"

    def __post_init__(self, id=None):
        if self.x is not None:
            self.id = id


        else:
            self.id = id
            self.__dict__ |= data_point_query(variables={"id": id})
        self.x = self.x * 1e-3
        self.y = self.y * 1e-3
        self.z = self.z * 1e-3


from mmcore.collections import ElementSequence
from scipy.spatial import cKDTree


class DataPointSet:
    _i: -1
    child_query = data_point_query

    def __iter__(self):
        que = GQLReducedQuery("""
        query MyQuery {
          panels_points {
            arch_type
            eng_type
            id
            subzone
            x
            y
            z
            zone
          }
        }
        """)

        def itr_():
            for i in que():
                yield DataPoint(**i)

        return itr_()

    def __getitem__(self, item):
        return DataPoint(item)

    def __len__(self):
        return GQLReducedQuery("""
        query MyQuery {
          panels_points_aggregate {
            aggregate {
              count
            }
          }
        }""")()["aggregate"]["count"]


dt = DataPointSet()
from mmcore.collections import ElementSequence

ldt = list(dt)
es = ElementSequence(ldt)


def to_Kd(ess):
    *aa, = zip(ess["x"], ess["y"], ess["z"])
    return cKDTree(np.array(aa) / 1000)


KD = to_Kd(es)


def tri_transform(line, next_line, flip=1, step=0.4):
    grp = AGroup(name="row", uuid=uuid.uuid4().hex)
    for x in line.divide_distance_planes(step):
        panel = AMesh(name="Panel",
                      geometry=ageomdict[pnl.uuid],
                      uuid=uuid.uuid4().hex,
                      material=m)
        point2 = np.array(ClosestPoint(x.origin, next_line.extend(60, 60))(x0=[0.5], bounds=[(0, 1)]).pt[0]).flatten()
        y = unit(point2 - x.origin)
        v2 = np.cross(unit(flip * x.normal), unit(y))
        pln = PlaneLinear(origin=x.origin, xaxis=unit(y), yaxis=v2)
        t = Transform.from_world_to_plane(pln)
        #tr = Transform()

        #tr.rotate(axis=[1, 0, 0], angle=-np.pi / 2)
        #panel @ tr
        panel @ t
        d, i = KD.query(x.origin + (y * 0.3))
        # #print(x.origin, x.origin + (y * 0.3), d, i, ldt[i])
        # panel.properties |= ldt[i].__dict__
        grp.add(panel)
    return grp.root()

def tri_transform_gen(line, next_line, flip=1, step=0.4):
    #grp = AGroup(name="row", uuid=uuid.uuid4().hex)
    for x in line.divide_distance_planes(step):

        point2 = np.array(ClosestPoint(x.origin, next_line.extend(60, 60))(x0=[0.5], bounds=[(0, 1)]).pt[0]).flatten()
        y = unit(point2 - x.origin)
        v2 = np.cross(unit(flip * x.normal), unit(y))
        pln = PlaneLinear(origin=x.origin, xaxis=unit(y), yaxis=v2)
        t = Transform.from_world_to_plane(pln)
        #tr = Transform()

        #tr.rotate(axis=[1, 0, 0], angle=-np.pi / 2)
        #panel @ tr

        #d, i = KD.query(x.origin + (y * 0.3))
        # #print(x.origin, x.origin + (y * 0.3), d, i, ldt[i])
        # panel.properties |= ldt[i].__dict__
        yield t


#
def solve_transforms_parallel(ll, step=0.4):
    def f2(i):
        node = ll.get(i)
        if node.next is None:
            pass
        elif node.prev is not None:
            if i % 2 == 0:
                ln = node.data.extend(-0.2, 1.3)
            else:
                ln = node.data.extend(0, 1.3)
            return merge_roots(tri_transform(ln, node.next.data, flip=-1, step=step),
                               tri_transform(ln, node.prev.data, flip=1, step=step))

    with mp.Pool(10) as p:
        return p.map(f2, range(1, len(ll)))


from mmcore.geom.extrusion import makeNodeJsExtrusions
from more_itertools import flatten

panel_mat = MeshPhongMaterial(color=ColorRGB(*PANEL_COLOR).decimal)

m = MeshPhongMaterial(color=ColorRGB(*PANEL_COLOR).decimal)


def merge_shapes_roots(one, other):
    one["object"]["children"].append(other["object"])
    one["geometries"].extend(other["geometries"])
    one["materials"].extend(other["materials"])
    if "shapes" in list(one.keys()) + list(other.keys()):
        if not ("shapes" in one.keys()):
            one["shapes"] = []
        if not ("shapes" in other.keys()):
            other["shapes"] = []

    one["shapes"].extend(other["shapes"])
    return one


def merge_roots(one, other):
    one["object"]["children"].append(other["object"])

    one["geometries"].extend(other["geometries"])
    one["materials"].extend(other["materials"])
    return one


Profile = namedtuple("Profile", ["boundary", "holes"])
profile1 = Profile(crd, holes)
profile2 = Profile(crdP2, holesP2)
profile3 = Profile(crdP3, holesP3)


class SubSyst:
    A = [-31.02414546224999, -17.3277158585, 9.136232981]
    B = [-22.583505462250002, 11.731284141500002, 1.631432487]
    C = [17.44049453775, 12.911284141500003, -5.555767019000001]
    D = [36.167156386749994, -7.314852424499997, -5.211898449000001]

    def __call__(self, step1=0.6, step2=1.8, step3=2.4, high1=0, high2=-0.16, high3=-0.24, **kwargs):
        self.__dict__ |= kwargs
        self.hp = HypPar4ptGrid(self.A, self.B, self.C, self.D)
        self.initial = self.first(step=step1, side="D", color=(70, 10, 240))
        self.lay1 = self.layer(copy.deepcopy(self.initial), high=high2, step=step2, color=(259, 49, 10), name="layer-1")
        self.lay2 = self.layer(copy.deepcopy(self.lay1), high=high3, step=step3, color=(25, 229, 100), name="layer-2")
        self.panels = self.solve_panels()
        ext1 = self.solve_subsystem_layer(self.initial, profile1, color=ColorRGB(50, 210, 220))
        ext2 = self.solve_subsystem_layer(self.lay1, profile2, color=ColorRGB(50, 60, 200))
        ext3 = self.solve_subsystem_layer(self.lay2, profile3, color=ColorRGB(50, 180, 20))
        self.extrusions = merge_shapes_roots(merge_shapes_roots(ext1, ext2), ext3)
        self.extrusions["object"]
        a = AGroup(name="Sub", uuid=uuid.uuid4())
        tr = Transform()
        tr.rotate(axis=[1, 0, 0], angle=-np.pi / 2)
        a @ tr
        rt = a.root()
        rt["object"]["children"].append(self.extrusions.pop("object"))
        rt["geometries"].extend(self.extrusions["geometries"])
        rt["shapes"] = self.extrusions["shapes"]
        rt["materials"].extend(self.extrusions["materials"])
        return merge_roots(rt, self.panels)

    def solve_subsystem_layer(self, layer, prof: Profile, color: ColorRGB):
        return makeNodeJsExtrusions(prof.boundary, prof.holes,
                                    list(map(lambda x: [x.start.tolist(), x.end.tolist()], layer)),
                                    color.decimal)

    def root(self):
        # grp = AGroup(name="Facade Layer", uuid=uuid.uuid4().hex)
        # grp.add(self.panels)
        # root = grp.root(shapes=self.extrusions["shapes"])
        # root = merge_roots(root, self.extrusions)
        # return root
        ...

    def solve_panels(self):
        _panels = AGroup(name="Panels", uuid=uuid.uuid4().hex)
        dll = DoublyLinkedList()
        for i in self.initial:
            dll.append(i)
        s = solve_transforms_parallel(dll)

        frst = s.pop(0)
        for itm in s:
            frst = merge_roots(frst, itm)

        # nn=[0.3, 0, 0.000, 1]

        # for o in s:
        #    cent=np.array(nn)
        #    panel = AMesh(name="Panel",
        #                  geometry=ageomdict[pnl.uuid],
        #                  uuid=uuid.uuid4().hex,
        #                  material=m)
        #    centt = (cent@o.matrix).tolist()[:-1]
        #    #print(o)
        #    d, idx = KD.query(centt, eps=0.00001)
        #    #print(d,idx,centt, ldt[idx])
        #    panel @ o
        #
        #    prp=dict(ldt[idx].__dict__)
        #    #print(prp.pop("x"))
        #    #print(prp.pop("y"))
        #    #print(prp.pop("z"))
        #    panel.properties = prp
        #    _panels.add(panel)
        # return _panels
        return frst

    def offset_hyp(self, h=2):
        hyp = self.hp
        A1, B1, C1, D1 = hyp.evaluate((0, 0)), hyp.evaluate((1, 0)), hyp.evaluate((1, 1)), hyp.evaluate((0, 1))
        hpp = []
        for item in [A1, B1, C1, D1]:
            hpp.append(np.array(item.point) + np.array(item.normal) * h)
        return HypPar4ptGrid(*hpp)

    def first(self, step=0.6, side="D", color=(70, 10, 240)):
        hyp = self.hp
        current = hyp.parallel_side_grid(step, side=side)

        return current

    def layer(self, prev, high, step, color=(70, 10, 240), name="layer"):
        hyp = self.hp
        prev.sort(key=lambda x: x.length)
        nxt = prev[-1]
        hp_next = self.offset_hyp(high)
        current = hp_next.parallel_vec_grid(step, nxt)

        return current

    def dump(self):
        with open("grp.json", "w") as f:
            ujson.dump(self(), f)
