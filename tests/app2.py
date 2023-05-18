import copy
import json
from collections import namedtuple

from localparams import *
from mmcore.collections import DoublyLinkedList
from mmcore.geom.parametric import PlaneLinear, ClosestPoint, HypPar4ptGrid

from mmcore.base.registry import adict
from mmcore.geom.transform import Transform
from mmcore.geom.vectors import unit
import multiprocess as mp
from mmcore.base.models.gql import LineBasicMaterial, MeshPhongMaterial
from mmcore.node import node_eval


def tri_transform(line, next_line, flip=1, step=0.4):
    for x in line.divide_distance_planes(step):
        point2 = np.array(ClosestPoint(x.origin, next_line.extend(60, 60))(x0=[0.5], bounds=[(0, 1)]).pt[0]).flatten()
        y = unit(point2 - x.origin)
        v2 = np.cross(unit(flip * x.normal), unit(y))
        pln = PlaneLinear(origin=x.origin, xaxis=unit(y), yaxis=v2)
        t = Transform.from_world_to_plane(pln)
        yield t


@node_eval
def makeExtrusions(coords, hls, axis, color):
    # language=JavaScript
    return """
    const THREE=require("three")
    
    function makeExtrusion(coords, holes, line_coords, color){
    let pts=[];
    const material1 = new THREE.MeshLambertMaterial( { color: color, wireframe: false } );
    const spline = new THREE.LineCurve3( v1=new THREE.Vector3( line_coords[0][0] , line_coords[0][1], line_coords[0][2]),
                        v2=new THREE.Vector3(  line_coords[1][0] , line_coords[1][1], line_coords[1][2]));
    coords.forEach((value, index, array)=>{
        pts.push(new THREE.Vector2(value[0],value[1]))
    })
   const extrudeSettings1 = {
                        steps: 1,
                        depth:1.0,
                        bevelEnabled: false,
                        extrudePath:spline
                    };
    let shape1 = new THREE.Shape( pts );

    let pttt=[];
    holes.forEach((value1,index1,array1)=>{

        let ptt=[]
        value1.forEach((value2, index2, array2)=>{
        ptt.push(new THREE.Vector2(value2[0],value2[1]))
    });
        pttt.push(new THREE.Shape(ptt));

    })
    pttt.forEach((value)=>{
         shape1.holes.push(value)
    })


    const geometry1 = new THREE.ExtrudeGeometry( shape1, extrudeSettings1 );
    
    const mesh =new THREE.Mesh(geometry1,  material1);
    return mesh;


};
function makeMany(cords,holes,axis,color){
    let grp=new THREE.Group();
    axis.forEach((value)=>{
    grp.add(makeExtrusion(cords,holes, value, color))
    })
    console.log(JSON.stringify(grp.toJSON()))

}
""" + f"makeMany({coords},{hls},{axis},{color});"


# r = makeExtrusion(crd, holes, l)
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
            return list(
                zip(tri_transform(ln, node.next.data, flip=-1, step=step),
                    tri_transform(ln, node.prev.data, flip=1, step=step)))

    with mp.Pool(20) as p:
        return p.map(f2, range(1, len(ll)))


from more_itertools import flatten

panel_mat = MeshPhongMaterial(color=ColorRGB(*PANEL_COLOR).decimal)

m = MeshPhongMaterial(color=ColorRGB(*PANEL_COLOR).decimal)


def merge_shapes_roots(one, other):
    one["object"]["children"].append(other["object"])
    one["geometries"].extend(other["geometries"])
    one["shapes"].extend(other["shapes"])
    one["materials"].extend(other["materials"])
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

        return

    def solve_subsystem_layer(self, layer, prof: Profile, color: ColorRGB):
        return makeExtrusions(prof.boundary, prof.holes, list(map(lambda x: [x.start.tolist(), x.end.tolist()], layer)),
                              color.decimal)

    def root(self):
        grp = AGroup(name="Facade Layer", uuid=uuid.uuid4().hex)
        grp.add(self.panels)
        root = grp.root(shapes=self.extrusions["shapes"])
        root = merge_roots(root, self.extrusions)
        return root

    def solve_panels(self):
        _panels = AGroup(name="Panels", uuid=uuid.uuid4().hex)
        dll = DoublyLinkedList()
        for i in self.initial:
            dll.append(i)
        s = solve_transforms_parallel(dll)
        *s, = flatten(flatten(s))
        for o in s:
            panel = AMesh(name="Panel",
                          geometry=ageomdict[pnl.uuid],
                          uuid=uuid.uuid4().hex,
                          material=m)

            panel @ o
            _panels.add(panel)
        return _panels

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
            ujson.dump(self.root(), f)


ss = SubSyst()

pnls = ss()
ss.dump()
from more_itertools import flatten

from mmcore.base import ALine

# ext=makeExtrusions(crd,holes, list(map(lambda x: [x.start.tolist(),x.end.tolist()], ss.initial)),ColorRGB(50,210,220).decimal)
# ext2=makeExtrusions(crdP2,holesP2, list(map(lambda x: [x.start.tolist(),x.end.tolist()], ss.lay1)),ColorRGB(50,60,200).decimal)
# ext3=makeExtrusions(crdP3,holesP3, list(map(lambda x: [x.start.tolist(),x.end.tolist()], ss.lay2)),ColorRGB(50,180,20).decimal)


# m = MeshPhongMaterial(color=ColorRGB(*PANEL_COLOR).decimal)

# jgm=utils.create_buffer_from_dict(jnt["geometries"][0])
# from mmcore.geom.parametric import ProximityPoints

# joints=AGroup(name="Joints-1-2",uuid=uuid.uuid4().hex)
# for i in ss.lay1:
#    for j in ss.initial:
#        res=ProximityPoints(i, j)(x0=[0.5,0.5],bounds=((0, 1), (0, 1)))
#        a,b=res.pt
#        ln=ALine(geometry=[a,b],uuid=uuid.uuid4().hex,material= LineBasicMaterial(color=ColorRGB(20,120,200).decimal))
# x=np.cross(unit(j.direction),unit(a-b))
# t=Transform.from_world_to_plane(PlaneLinear(normal=unit(a-b), origin=b))
# joint12 = AMesh(geometry=jgm,
#                material=skoba_mat,
#                uuid=uuid.uuid4().hex)
# joint12@t
# joints.add(joint12)
#        joints.add(ln)

# grp.add(joints)


# grp.add(grplns)
