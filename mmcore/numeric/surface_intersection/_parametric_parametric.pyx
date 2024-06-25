from mmcore.numeric.vectors cimport scalar_unit,scalar_cross,scalar_norm
from mmcore.numeric cimport calgorithms
def get_plane(origin,du,dv):
    duu = du/scalar_norm(du)

    dn = scalar_unit(scalar_cross(duu, dv))
    dvu = scalar_cross(dn, duu)
    return [origin,duu,dvu,dn]

def freeform_step_debug(pt1,pt2,du1,dv1,du2,dv2):
    pl1, pl2 = get_plane(pt1,du1,dv1),get_plane(pt2,du2,dv2)

    ln = np.array(plane_plane_intersect(pl1, pl2))

    np1 = np.asarray(closest_point_on_ray((ln[0], ln[1]), pt1))
    np2 = np.asarray(closest_point_on_ray((ln[0],  ln[1]), pt2))

    return np1, np1 + (np2 - np1) / 2, np2

def get_normal(du,dv):
    duu = scalar_unit(du)
    dn = scalar_unit(scalar_cross(duu, dv))

    return duu,dn
def solve_marching(pt1,pt2,du1,dv1,du2, dv2,tol, side=1):

    pl1, pl2 = get_plane(pt1,du1,dv1),get_plane(pt2,du2,dv2
                                            )


    marching_direction = scalar_unit(scalar_cross(pl1[-1], pl2[-1]))

    tng=np.zeros((2,3))


    calgorithms.evaluate_curvature(marching_direction*side, pl1[-1], tng[0],tng[1] )

    K=tng[1]


    r = 1 / scalar_norm(K)

    step = np.sqrt(np.abs(r ** 2 - (r - tol) ** 2)) * 2


    new_pln = np.array([pt1 + marching_direction*side * step, pl1[-1], pl2[-1], marching_direction*side])

    return plane_plane_plane_intersect(pl1, pl2, new_pln), step


def improve_uv(du ,dv, xyz_old,xyz_better):

    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv =  dv

    delta= xyz_better-xyz_old

    xy = [[dxdu, dxdv], [dydu, dydv]], [delta[0], delta[1]]
    xz = [[dxdu, dxdv], [dzdu, dzdv]], [delta[0], delta[2]]
    yz = [[dydu, dydv], [dzdu, dzdv]], [delta[1], delta[2]]

    max_det = sorted( [xy, xz, yz], key=lambda Ab: scipy.linalg.det(Ab[0]), reverse=True)[0]
    return np.linalg.solve(*max_det)
