import uuid
from collections import deque

import numpy as np
from more_itertools import flatten

from mmcore.base import AGroup, AMesh
from mmcore.base.geom import MeshData
from mmcore.geom.materials import ColorRGB
from mmcore.geom.shapes import LegacyShape
from mmcore.node import node_eval


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
    return AMesh(geometry=buf, uuid=uuid.uuid4().hex, material=material)

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

