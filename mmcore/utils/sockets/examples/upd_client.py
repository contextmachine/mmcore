from mmcore.baseitems import Matchable
from mmcore.services.service import RhinoIronPython, Serviceable


class IronPyCommand(Serviceable):
    __match_args__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def injection(cls):
        return cls.__doc__


class BrepExtruder(IronPyCommand):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg

# Create 4 corner surf
# noinspection PyUnresolvedReferences
mybrep=rg.Brep.CreateFromCornerPoints(input_msg['pt1'],input_msg['pt2'],input_msg['pt3'],input_msg['pt4'],
                                        tolerance=input_msg["tolerance"])

# Make brep offset
a,b,c = rg.Brep.CreateOffsetBrep(mybrep, 1.0, True, True, 0.1)
offset_brep=list(a)


    """
    __match_args__ = "mybrep", "offset_brep"


class Sweep(IronPyCommand):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg
# noinspection PyUnresolvedReferences
rail = rg.NurbsCurve.CreateControlPointCurve(input_msg['points'], input_msg['degree'])
_, pln = rail.FrameAt(0.0)
# noinspection PyUnresolvedReferences
section=input_msg['section']
plnn = rg.Plane(pln.Origin, pln.YAxis,pln.ZAxis)
section_transform = rg.Transform.PlaneToPlane(rg.Plane.WorldXY, plnn)
section.Transform(section_transform)
swp = rg.SweepOneRail()
brp = list(swp.PerformSweep(rail, section))
b
    """
    __match_args__ = "brp", "swp"


@classmethod
def injection(cls):
    return cls.__doc__


@RhinoIronPython(('localhost', 10081), bytesize=1024 ** 10)
class StopSignal(Matchable):
    """stop"""
    __match_args__ = ()
