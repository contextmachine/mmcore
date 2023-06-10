import numpy as np
import time
from mmcore.base import ALine, AGroup, APoints
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import LineBasicMaterial, PointsMaterial
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import Linear
from mmcore.base.params import param_graph_node
from mmcore.base.sharedstate import serve


def line_from_points(start, end):
    return Linear.from_two_points(start, end)


@param_graph_node(
    params=dict(a=[-8.323991, 6.70421, -8.323991], b=[-8.323991, -6.70421, 8.323991], c=[8.323991, 6.70421, 8.323991],
                d=[8.323991, -6.70421, -8.323991]))
def hypar_sides(a, b, c, d):
    return line_from_points(a, b), line_from_points(c, d)


def hypar_sides_eval(sides, uv):
    a, b = sides
    u, v = uv


    return Linear.from_two_points(a.evaluate(u), b.evaluate(1-u)).evaluate(v)

col=ColorRGB(100,100,100).decimal
@param_graph_node(params=dict(sides=hypar_sides, uv=[[0, 1, 8], [1, 0, 8]], uuid="hypar-wires",color=col))
def hypar_wires(sides, uv, uuid, color):
    grp = AGroup(name="Hypar Wires", uuid=uuid)
    u, v = uv
    vertices=[]
    _vertices=[]
    _faces=[]
    for ik,i in enumerate(np.linspace(*u)):
        points = []

        for jk,j in enumerate(np.linspace(*v)):
            point=hypar_sides_eval(sides, uv=(i, j))
            #pt=APoints(name=f"{ik}-{jk}",geometry=point,uuid=f'{uuid}-u{ik}-v{jk}', material=PointsMaterial(color=color, size=0.2))
            #grp.add(pt)
            vertices.append(point)
            _vertices.append(f"{ik}-{jk}")
            _faces.append((f"{ik}-{jk}",f"{ik+1}-{jk}",f"{ik+1}-{jk+1}"))
            points.append(point)

        grp.add(ALine(geometry=points,name=f'u-{ik}', uuid=f'{uuid}-u-{ik}', material=LineBasicMaterial(color=color)))

    for ik,i in enumerate(np.linspace(*v)):
        points = []

        for jk,j in enumerate(np.linspace(*u)):
            point=hypar_sides_eval(sides, uv=(j, i))
            #pt = APoints(name=f"{ik}-{jk}", geometry=point, uuid=f'{uuid}-v{jk}-u{ik}',material=PointsMaterial(color=color))
            #grp.add(pt)

            points.append(point)
            _faces.append((f"{ik}-{jk}", f"{ik+1}-{jk+1}", f"{ik}-{jk + 1}"))

        grp.add(ALine(geometry=points, name=f'v-{ik}',uuid=f'{uuid}-v-{ik}', material=LineBasicMaterial(color=color)))
    indices=[]

    for face in _faces:
        fc=[]
        try:

            for v in face:

                fc.append(_vertices.index(v))
            indices.append(fc)
        except:

                pass
    md=MeshData(vertices=vertices, indices=indices)
    md.calc_normals()
    grp.add(md.to_mesh(uuid=uuid+"-mesh" ,opacity=0.5, castShadow=False, flatShading=False,color=color))
    #print(md.normals)
    return grp


serve.start()


def animate(count=3):
    def inner():
        for j in np.linspace(100, -100, 50).tolist() + np.linspace(-100, 100, 50).tolist():
            time.sleep(0.01)
            hypar_wires(**{
                'sides': {
                    'a': [-100, j, -100],
                    'b': [-100, -j, 100],
                    'c': [100, j, 100],
                    'd': [100, -j, -100]
                }
            }
                        )

    for i in range(count):
        inner()


animate(5)
