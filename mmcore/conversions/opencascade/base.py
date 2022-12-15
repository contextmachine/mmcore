#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
import OCC.Core.GeomConvert as gc
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepFill import *
from OCC.Core.GC import *
from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
from OCC.Display.SimpleGui import init_display

from mm.baseitems import GeometryItem

gg = gc.Convert_ConicToBSplineCurve()

edge1 = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, 1, 0))
arc = GC_MakeArcOfCircle(gp_Pnt(0, 1, 0), gp_Pnt(0.5, 1.5, 0), gp_Pnt(1, 1, 0))
edge2 = BRepBuilderAPI_MakeEdge(arc.Value())
edge3 = BRepBuilderAPI_MakeEdge(gp_Pnt(1, 1, 0), gp_Pnt(1, 0.5, 0))
spine = BRepBuilderAPI_MakeWire(edge1.Edge(), edge2.Edge(), edge3.Edge())
profile1 = BRepBuilderAPI_MakeWire(
    BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), 0.1)).Edge())
profile2 = BRepBuilderAPI_MakeWire(
    BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), 0.11)).Edge())

ppp = BRepFill_PipeShell(spine.Wire())
ppp.Add(profile1.Wire())
ppp.Build()

ppp.MakeSolid()

ptarr = [[-121.17591093169389, 42.648567217674390, 0.0], [-36.442468672889866, 70.043589902851664, 0.0],
         [106.90358026117707, 77.051618961850480, 0.0], [214.57239034943180, 39.463099463584008, 0.0],
         [272.54790347387672, -23.609162067405457, 0.0]]


# array = TColgp_Array1OfPnt(1, len(ptarr))
# [array.SetValue(i + 1, gp_Pnt(*pt)) for i, pt in enumerate(ptarr)]


class OccPoint(GeometryItem):
    x, y, z = 0.0, 0.0, 0.0

    def __init__(self, x, y, z, **kwargs):
        super().__init__(x=x, y=y, z=z, **kwargs)

    @property
    def occ(self):
        return gp_Pnt(self.x, self.y, self.z)


class OccPointArray1d(list, GeometryItem):
    def __init__(self, points):
        self._array = TColgp_Array1OfPnt(1, len(points))
        [self._array.SetValue(i + 1, gp_Pnt(*pt)) for i, pt in enumerate(points)]

        super().__init__()

    def __setitem__(self, key, value):
        self._array.SetValue(key, value)

    def __getitem__(self, key):
        return self._array.Value(key)

    def __next__(self):
        return self._array.next()


class BSpline:
    def __init__(self):
        ...

    @classmethod
    def from_points(cls, points):
        array = OccPointArray1d(points)


display, start_display, add_menu, add_function_to_menu = init_display()


class TrimmingCone(GeometryItem):
    dsp = display
    strt = start_display

    def __init__(self, pt, ptb, r1, r2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occ = GC_MakeTrimmedCone(gp_Pnt(*pt), gp_Pnt(*ptb), r1, r2).Value()

    def set_trim(self, a, b, c, d):
        self.occ.SetTrim(a, b, c, d)

    def show(self):
        dsp = display.DisplayShape(self.occ, update=True)


tc = TrimmingCone([1, 2, 3], [12, 15, 8], 5, 7)
tc.set_trim(0.1, 3.14, 0.1, 11.9)
tc.show()
start_display()
