from __future__ import annotations


from mmcore.geom._nurbs_eval import _tuple_to_nurbs, _nurbs_to_tuple, NURBSCurveTuple, to_homogeneous_1d,NURBSSurfaceTuple,to_homogeneous_2d, \
    from_homogeneous_2d
try:
    import Rhino.Geometry as rg
    
except ImportError:
    import rhino3dm as rg
import numpy as np
def rhcurve_to_nt(x:rg.Curve)->NURBSCurveTuple:
    pts = []
    w = []
    nx = x.ToNurbsCurve()
    for pt in nx.Points:
        pts.append([pt.X / pt.Weight, pt.Y / pt.Weight, pt.Z / pt.Weight])
        w.append(pt.Weight)

    knots = list(nx.Knots)
    knots = [knots[0]] + knots + [knots[-1]]
    order = nx.Order
    return NURBSCurveTuple(order, np.array(knots), np.array(pts), np.array(w))





def nt_to_rhcurve(nt: NURBSCurveTuple)->rg.NurbsCurve:
    # rg.NurbsCurve(dimension: int, rational: bool, order: int, pointCount: int)
    crv = rg.NurbsCurve(3, True, nt.order, nt.control_points.shape[0])
    cpts = to_homogeneous_1d(nt.control_points, nt.weights)
    knots = nt.knot[1:][:-1]
    for i in range(len(crv.Knots)):
        crv.Knots[i] = knots[i]
    for i in range(nt.control_points.shape[0]):
        crv.Points.SetPoint(i, rg.Point3d(*cpts[i][:3]))

        crv.Points.SetWeight(i, cpts[i][-1])
    return crv


def is_opennurbs_style_knots(degree, knots):
    _, cnt = np.unique(knots, return_counts=True)

    return cnt[0] == degree


def create_nt_surf(
    deg_u: int,
    deg_v: int,
    knots_u: list[float],
    knots_v: list[float],
    control_points: list[list[rg.Point3d]],
    weights: list[list[float]] = None,
):
    # rg.NurbsCurve(dimension: int, rational: bool, order: int, pointCount: int)
    u_count, v_count = len(control_points), len(control_points[0])
    cpts = np.zeros((u_count, v_count, 3))

    surf = NURBSSurfaceTuple(
        deg_u + 1,
        deg_v + 1,
        np.asarray([knots_u[0]] + knots_u + [knots_u[-1]] if is_opennurbs_style_knots(deg_u, knots_u) else knots_u,dtype=float),
        np.asarray(
        [knots_v[0]] + knots_v + [knots_v[-1]] if is_opennurbs_style_knots(deg_v, knots_v) else knots_v,dtype=float),
        cpts,
        np.asarray(weights),
    )

    for i in range(u_count):
        for j in range(v_count):

            surf.control_points[i, j, :] = tuple(control_points[i][j])

    return surf


def nt_to_rhsurf(nt:NURBSSurfaceTuple)->rg.NurbsSurface:
    b = nt
    knots_u = b.knot_u
    knots_v = b.knot_v
    knots_u = knots_u[1:][:-1]
    knots_v = knots_v[1:][:-1]
    cpts = to_homogeneous_2d(b.control_points, b.weights)
    v_count = len(cpts[0])
    u_count = len(cpts)
    order_u = b.order_u
    order_v = b.order_v
    rational = True

    crv = rg.NurbsSurface.Create(3, rational, order_u, order_v, u_count, v_count)

    for i in range(crv.KnotsU.Count):

        crv.KnotsU[i] = knots_u[i]

    for i in range(crv.KnotsV.Count):

        crv.KnotsV[i] = knots_v[i]
    for i in range(crv.Points.CountU):
        for j in range(crv.Points.CountV):

            crv.Points.SetPoint(i, j, rg.Point3d(*cpts[i][j][:3]))
            if rational:
                crv.Points.SetWeight(i, j, cpts[i][j][-1])

    return crv

def rhsurf_to_nt(x:rg.Surface)->NURBSSurfaceTuple:
    nx = x.ToNurbsSurface()
    pts = [(pt.X, pt.Y, pt.Z, pt.Weight) for pt in nx.Points]
    ku = list(nx.KnotsU)
    kv = list(nx.KnotsV)
    ku = [ku[0]] + ku + [ku[-1]]
    kv = [kv[0]] + kv + [kv[-1]]
    ou = nx.OrderU
    ov = nx.OrderV
    cpt, ws = from_homogeneous_2d(np.array(pts).reshape((nx.Points.CountU, nx.Points.CountV, 4)))
    return NURBSSurfaceTuple(ou, ov, knot_u=np.array(ku), knot_v=np.array(kv), control_points=cpt, weights=ws)
    
