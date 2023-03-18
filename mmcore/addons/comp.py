import compas.geometry as cg
from mmcore.addons import ModuleResolver
with ModuleResolver() as rsl:
    import rhino3dm
import rhino3dm



def from_rhino_plane_transform(plane: rhino3dm.Plane):
    return cg.Transformation.from_frame(
        cg.Frame.from_plane(cg.Plane(
            [plane.Origin.X, plane.Origin.Y, plane.Origin.Z],
            [plane.ZAxis.X, plane.ZAxis.Y, plane.ZAxis.Z])))
