from mmcore.baseitems import Matchable
from mmcore.services.core import RhinoIronPython
from cxmdata import CxmData
import rhino3dm as rg
# Create a UDP socket


@RhinoIronPython(('localhost', 10081), bytesize=1024 * 8)
class BrepExtruder(Matchable):
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



@RhinoIronPython(('localhost', 10006), bytesize=1024 * 8)
class SweepOneRail(Matchable):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg
sweep = rg.SweepOneRail()
# noinspection PyUnresolvedReferences
rail=input_msg["rail"]
# noinspection PyUnresolvedReferences
profile=input_msg["crossSection"]
framestart = rg.Plane.WorldXY
_, frame = rail.FrameAt(0.0)
# noinspection PyTypeChecker
transform = rg.Transform.PlaneToPlane(framestart, frame)
profile.Transform(transform)
sweep_result=list(sweep.PerformSweep(rail=rail, crossSection=profile))
    """
    __match_args__ = "transform", "sweep_result"


@RhinoIronPython(('localhost', 10006), bytesize=1024 * 8)
class SweepTwoRail(Matchable):
    """
# You can use "Inject language reference" in Pycharm to continue using the hints & intelligence.
# It is simple and native.
# Maybe we could also probably use something like dockets ...

import Rhino.Geometry as rg
rg.SweepFrame()
sweep = rg.SweepTwoRail()
# noinspection PyUnresolvedReferences
rails=input_msg["rails"]
# noinspection PyUnresolvedReferences
profile=input_msg["crossSection"]
framestart = rg.Plane.WorldXY
_, frame = rails[0].FrameAt(0.0)
# noinspection PyTypeChecker
transform = rg.Transform.PlaneToPlane(framestart, frame)
profile.Transform(transform)
c1,c2=rails
sweep_result=list(sweep.PerformSweep(c1,c2, crossSection=profile))
    """
    __match_args__ = "transform", "sweep_result"


@RhinoIronPython(('localhost', 10088), bytesize=1024 * 8)
class StopSignal(Matchable):
    """
    stop
    """
    __match_args__ = ()


stop=CxmData({
    "py":"stop"
    })
