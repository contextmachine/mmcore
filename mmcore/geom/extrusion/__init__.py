from collections import deque
from uuid import uuid4

import numpy as np
from more_itertools import flatten
from multipledispatch import dispatch

from mmcore.base import AGroup, AMesh
from mmcore.base.geom import MeshData
from mmcore.common.models.mixins import MeshViewSupport
from mmcore.func import vectorize
from mmcore.geom.materials import ColorRGB
from mmcore.geom.mesh import union_mesh_simple
from mmcore.geom.mesh.compat import build_mesh_with_buffer
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds

from mmcore.geom.shapes import LegacyShape
from mmcore.geom.vec import unit
from mmcore.node import node_eval


@vectorize(signature='(i,j)->(i,k,j)')
def polyline_to_lines(pln_points):
    return np.stack([pln_points, np.roll(pln_points, -1, axis=0)], axis=1)



@node_eval
def makeNodeJsExtrusions(coords, hls, axis, color):
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


from mmcore.base.models.gql import MeshPhongMaterial


def bnd(shp, h, color=ColorRGB(100,100,100).decimal):
    vrtx = []
    for c in shp:
        vrtx.extend([c + [h], c + [0.0]])
    d = deque(range(len(vrtx)))
    material = MeshPhongMaterial(color=color)
    d1 = d.copy()
    d2 = d.copy()
    d2.rotate(-1)
    d.rotate(1)
    *l, = zip(d1, d, d2)

    md = MeshData(vertices=list(vrtx), indices=l)
    md.calc_indices()
    buf = md.create_buffer()
    return AMesh(geometry=buf, uuid=uuid4().hex, material=material)

def tess(shp, h):
    a2 = np.array(shp.mesh_data.vertices)
    a2[..., 2] += h
    *l, = zip(shp.mesh_data.vertices.tolist(), a2.tolist())
    *ll, = flatten(l)
    ixs = []

    def tess_bound(boundary):
        for i, v in enumerate(boundary):
            if i > len(boundary):
                yield boundary[i] + [0], boundary[i] + [h], boundary[i + 1] + [h], boundary[i] + [0], boundary[
                    i - 1] + [0], boundary[i - 1] + [h]
            else:
                yield boundary[i] + [0], boundary[i] + [h], boundary[i - 1] + [h], boundary[i] + [0], boundary[
                    i - 1] + [0], boundary[i - 1] + [h]

    for i in [tess_bound(bndr) for bndr in [shp.boundary] + shp.holes]:
        for j in i:
            for jj in j:
                ixs.append(ll.index(jj))

    return ixs, ll


from mmcore.geom.csg import CSG, BspPolygon


def simple_extrusion(shape, h):
    grp=AGroup()
    ixs,vxs=tess(shape, h)
    grp.add(MeshData(vertices=vxs, indices=ixs).to_mesh())
    s2 = LegacyShape(shape.boundary, shape.holes, color=shape.color, h=h)
    grp.add(s2.mesh)
    grp.add(shape.mesh)

    return grp

def mesher(obj, **kwargs):
        grp2 = AGroup()
        for p in obj.toPolygons():
            p.mesh_data()
            grp2.add(p.mesh(**kwargs))
        return grp2

def csg_extrusion(shape, h):
    grp=AGroup()
    ixs,vxs=tess(shape, h)

    polys=[]
    s2 = LegacyShape(shape.boundary, shape.holes, color=shape.color, h=h)
    for sshape in shape.mesh_data.faces:
        polys.append(BspPolygon(sshape))
    for ss2 in s2.mesh_data.faces:
        pl=BspPolygon(ss2)
        pl.flip()
        polys.append(pl)

    md = MeshData(vertices=vxs, indices=ixs)

    for mmd in md.faces[1:]:
        #print(mmd)
        polys.append(BspPolygon(mmd))

    csgm = CSG.fromPolygons(polys).refine()
    csgm=csgm - CSG.cylinder(start=(0, 0, 50), end=(-200, 0, 25), radius=15)
    grp.add(mesher(csgm, material=MeshPhongMaterial(color=ColorRGB(40, 100, 140).decimal)))


    return grp





@vectorize(signature='(i),(i),(i)->(j,i)')
def extrude_line(start, end, vec):
    return np.array([start, end, end + vec, start + vec])


@vectorize(signature='(j,i),(i)->(k,i)')
def extrude_line2(startend, vec):
    return np.array([startend[0], startend[1], startend[1] + vec, startend[0] + vec])

def extrude_polyline(corners, vec):
    corn = np.array(corners)
    vec = np.array(vec)
    lns = polyline_to_lines(corn)
    return [corn.tolist()] + extrude_line(lns[:, 0, :], lns[:, 1, :], vec).tolist() + [(corn + vec).tolist()]


def _base_extrusion_init(self, profile, path):
    self.profile = profile
    self.path = path


class Extrusion(MeshViewSupport):
    """
    @param: profile: Это любой геометрические 2D объект возвращающий итератор собственных точек при вызове __iter__.
    Банальным примером является обычный список или numpy массив координат. Однако, также могут быть использованы
    объекты таких классов как Rectangle, RectangleUnion etc.
    @param: 3d вектор экструзии, при инициализации будет преобразован в numpy массив. Также в конструктор можно передать
    float. В этом случае он будет умножен на World Z.
    """
    profile: object
    path: 'np.ndarray(3, float)'
    faces: 'list'

    def __init__(self, shape, h: float, vec=(0., 0., 1.0), **kwargs):
        self._h = h
        self.shape = shape
        self.profile = list(shape)
        self._vec = unit(np.array(vec))
        self.path = unit(np.array(vec)) * h
        self.__init_support__(**kwargs)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, v):
        self._h = v
        self.path = self._vec * self._h

    @property
    def vec(self):
        return self._vec

    @vec.setter
    def vec(self, v):
        self._vec = unit(np.array(v))
        self.path = self._vec * self._h

    @property
    def faces(self):
        return extrude_polyline(self.profile, self.path)


    @property
    def caps(self):
        return self.faces[0], self.faces[-1]

    @property
    def sides(self):
        return self.faces[1:-1]

    def to_mesh_view(self):
        return union_mesh_simple([mesh_from_bounds(np.array(face).tolist(), color=self.color) for face in self.faces])



@dispatch(Extrusion, tuple)
def to_mesh(obj: Extrusion, color=(0.3, 0.3, 0.3)):
    return union_mesh_simple([mesh_from_bounds(face, color=color) for face in obj.faces])


@dispatch(Extrusion)
def to_mesh(obj: Extrusion):
    return union_mesh_simple([mesh_from_bounds(face, color=(0.3, 0.3, 0.3)) for face in obj.faces])
