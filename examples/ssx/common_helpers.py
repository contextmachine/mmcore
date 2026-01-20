import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np

from mmcore.geom.bvh.lbvh import AABB

from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple, _nurbs_to_tuple
try:
    from mmcore.extras.renderer.renderer3d import Viewer, OrbitCamera
    VIEWER_INSTALLED = True
except ImportError:
    VIEWER_INSTALLED = False

from mmcore.construction import nurbs_curve
from dataclasses import dataclass,field
@dataclass
class PointMaterial:
    color: tuple[float, float, float,float] = field(default=(1.0, 1.0, 1.0, 1.0))
    size:int=8

@dataclass
class ControlNetMaterial:
    color: tuple[float, float, float,float] = field(default=(1.0, 1.0, 1.0, 1.0))
    control_point_material: PointMaterial =  field(default_factory=PointMaterial)


@dataclass
class CurveMaterial:
    color: tuple[float, float, float,float] = field(default=(1.0, 1.0, 1.0, 1.0))
    show_control_net: bool = field(default=False)
    control_net_material: ControlNetMaterial= field(default_factory=ControlNetMaterial)


@dataclass
class WiresMaterial(CurveMaterial):

    u_count: int = field(default=1)
    v_count: int = field(default=1)

@dataclass
class SurfaceMaterial:

    color:tuple[float,float,float,float]=field(default=(0.5, 0.5, 0.9, 0.05))
    wires_material: WiresMaterial = field(
        default_factory=lambda: WiresMaterial((1.0, 1.0, 1.0, 1.0), show_control_net=False, u_count=1, v_count=1)
    )
    show_control_net: bool = field(default=False)
    control_net_material: ControlNetMaterial= field(default_factory=ControlNetMaterial)


# Defaults
ssx_point_material = PointMaterial( color=(0.0, 1.0, 0.5, 1.0), size=12)
ssx_branch_material:CurveMaterial=CurveMaterial((0.0, 1.0, 0.5, 1.0),
                                                show_control_net=True,
                                                control_net_material=ControlNetMaterial((0.0, 1.0, 0.5, 0.7),

                                                                                        control_point_material=PointMaterial((0.0, 1.0, 0.5, 0.4), size=8)
                                                                                        )
                                                )


surface_material:SurfaceMaterial=SurfaceMaterial()

def parse_args():
    parser = argparse.ArgumentParser()
    ssx_params=parser.add_argument_group(title="SSX Parameters")
    ssx_params.add_argument("--atol", type=float, default=1e-3)
    ssx_params.add_argument("--angle_tol", type=float, default=0.052)

    general_params=parser.add_argument_group(title="General")
    general_params.add_argument('--viewer', action='store_true')

    general_params.add_argument("--save-pkl", action="store_true")
    general_params.add_argument("--pkl-path", type=Path,default=None)

    general_params.add_argument("--loglevel", type=str, default="INFO")

    view_params = parser.add_argument_group(title="View Options (if --viewer is specified)")
    view_params.add_argument('-c','--show-inter-cpts',action='store_true',help='show control points of intersection curves')

    return parser.parse_args()
def _find_root_frame(fr=None):

    cur_fr = inspect.currentframe() if fr is None else fr
    while True:
        if cur_fr.f_back and cur_fr.f_back != cur_fr:
            cur_fr = cur_fr.f_back
        else:
            break
    return cur_fr


def save_pkl(s1,s2,result,fp=None)->Path:
    if fp is None:
        fp=inspect.getfile(_find_root_frame(inspect.currentframe()))
    pth=Path(fp).with_suffix('.pkl')
    with open(pth, 'wb') as f:
        pickle.dump(((s1, s2), [r.curve_xyz for r in result[0]], [r.curve_st for r in result[0]], [r.curve_uv for r in result[0]]), f)
    return pth


def draw_ssx(
    s1: NURBSSurfaceTuple,
    s2: NURBSSurfaceTuple,
    result,
    viewer=None,
    recompute_camera: bool = True,
    surf1_material: SurfaceMaterial = None,
    surf2_material: SurfaceMaterial = None,
    intersection_curves_material: CurveMaterial = None,
    intersection_points_material: PointMaterial = None,
):
    if surf1_material is None:
        surf1_material=surface_material
    if surf2_material is None:
        surf2_material=surface_material
    if intersection_curves_material is None:
        intersection_curves_material=ssx_branch_material
    if intersection_points_material is None:
        intersection_points_material=ssx_point_material

    bb = AABB.from_points(s1.control_points.reshape(-1, 3)).merge(AABB.from_points(s2.control_points.reshape(-1, 3)))
    if viewer is None:
        viewer = Viewer(camera=OrbitCamera(target=bb.centroid(), distance=np.linalg.norm(bb.diag())*2, near=1.0))

    viewer.add_nurbs_surface(s1, color=surf1_material.wires_material.color, surface_color=surf1_material.color, u_count=surf1_material.wires_material.u_count, v_count=surf1_material.wires_material.v_count)
    viewer.add_nurbs_surface(
        s2,color=surf2_material.wires_material.color,surface_color=surf2_material.color,u_count=surf2_material.wires_material.u_count,v_count=surf2_material.wires_material.v_count

    )

    for branch in result[0]:

        viewer.add_nurbs_curve(branch.curve_xyz, color=intersection_curves_material.color)
        if intersection_curves_material.show_control_net:
            for p in branch.curve_xyz.control_points:
                viewer.add_point3d(p, color=intersection_curves_material.control_net_material.control_point_material.color, size_px=intersection_curves_material.control_net_material.control_point_material.size)
            viewer.add_nurbs_curve(nurbs_curve(branch.curve_xyz.control_points, 1), color=intersection_curves_material.control_net_material.color)
        # renderer.add_point3d(branch.curve_xyz.end(), color=(0.0, 1.0, 0.5, 1.0), size_px=6)
    for p in result[1]:
        viewer.add_point3d(p.xyz, color=intersection_points_material.color, size_px=intersection_points_material.size)
    if recompute_camera:
        viewer.cam.target=viewer.scene_info.bbox.centroid()
        viewer.cam.distance=np.linalg.norm(bb.diag())*2
    return viewer
