from collections import deque

import uuid

import copy
import numpy as np

from mmcore.base import AGroup, AMesh
from mmcore.base.geom import MeshData
from mmcore.geom.materials import ColorRGB
from mmcore.node import node_eval
from mmcore.geom.shapes import Shape
import gmsh


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


from mmcore.base.models.gql import MeshPhongMaterial, LineBasicMaterial

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
    md.calc_normals()
    buf = md.create_buffer()
    return AMesh(geometry=buf, uuid=uuid.uuid4().hex, material=material)


def simple_extrusion(shape, h):
    grp=AGroup()
    grp.add(bnd(shape.boundary, h))
    for hole in shape.holes:
        grp.add(bnd(hole, h))

    s2=Shape(shape.boundary,shape.holes, h=h)
    grp.add(s2.mesh)
    grp.add(shape.mesh)

    return grp
