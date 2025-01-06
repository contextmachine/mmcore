from mmcore.numeric.intersection import ccx
from mmcore.geom.nurbs import greville_abscissae,decompose_curve
from mmcore.numeric.numeric import evaluate_curvature
from mmcore.topo.brep import Face,Loop,Edge
def tessellate_edge(edge:Edge):

    greville_abscissae(edge.represented_by.value)
class QuadTree2DAdaptiveTessellation:
    face:Face
    outer:Loop
    inner:list[Loop]
    def __init__(self, face:Face):
        self.face=face
        self.outer=face.bounded_by[0]
        self.inner=face.bounded_by[1:]
    def step(self):
        self.face

