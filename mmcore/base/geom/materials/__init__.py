from mmcore.geom.materials import ColorRGB
from mmcore.node import node_eval



@node_eval
def makeMeshLine(points, color: ColorRGB, size_attenuation=1, line_width=2, opacity=1.0):
    line = f"makeMeshLine({points}, {color.decimal}, {size_attenuation}, {line_width}, {opacity});"
    # language=JavaScript
    return """const THREE = require('three');
              const MeshLine = require('three.meshline').MeshLine;
              const MeshLineMaterial = require('three.meshline').MeshLineMaterial;
              const MeshLineRaycast = require('three.meshline').MeshLineRaycast;
              function makeMeshLine(pts, color, sizeAttenuation, lineWidth, opacity) {
                const points = [];
                for (let j = 0; j < pts.length; j += 1) {
                    points.push(new THREE.Vector3(pts[j][0],pts[j][1],pts[j][2]));
                }
                const resolution= new THREE.Vector2(3840,2160)
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new MeshLine();
                const material = new MeshLineMaterial({color:color, 
                                                       opacity:opacity,
                                                       resolution:resolution, 
                                                       sizeAttenuation:sizeAttenuation, 
                                                       lineWidth:lineWidth});
                line.setGeometry(geometry);
                const mesh = new THREE.Mesh(line, material)
                console.log(JSON.stringify(mesh.toJSON()));
                
              };
    """ + line
