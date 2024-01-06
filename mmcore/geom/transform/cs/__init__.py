from mmcore.geom.vec import *


def make_hcs_3d(a, b, c):
    """build a 3D homogeneus coordiate system from three vectors"""
    u = b - a
    u = u / norm(u)
    v = c - a
    v = v / norm(v)
    w = cross(u, v)
    v = cross(w, u)
    hcs = np.array([
        [u[0], v[0], w[0], a[0]],
        [u[1], v[1], w[1], a[1]],
        [u[2], v[2], w[2], a[2]],
        [0.0, 0.0, 0.0, 1.0]])
    return hcs


def make_hcs_3d_scaled(a, b, c):
    """build a 3D homogeneus coordiate system from three vectors"""
    # create orthnormal basis
    u = b - a
    u = u / norm(u)
    v = c - a
    v = v / norm(v)
    w = cross(u, v)
    v = cross(w, u)
    # scale
    u = u / norm(u) / norm(b - a)
    v = v / norm(v) / norm(c - a)
    hcs = np.array([
        [u[0], v[0], w[0], a[0]],
        [u[1], v[1], w[1], a[1]],
        [u[2], v[2], w[2], a[2]],
        [0.0, 0.0, 0.0, 1.0]])
    return hcs


def make_hcs_2d(a, b):
    """build a 2D homogeneus coordiate system from two vectors"""
    u = b - a
    if np.allclose(norm(u), 0.0):  # 2006/6/30
        return None
    else:
        u = u / norm(u)
    v = np.array([-u[1], u[0]])
    hcs = np.array([[u[0], v[0], a[0]], [u[1], v[1], a[1]], [0.0, 0.0, 1.0]])
    return hcs


def make_hcs_2d_scaled(a, b):
    """build a 2D homogeneus coordiate system from two vectors, but scale with distance between input point"""
    u = b - a
    if np.allclose(norm(u), 0.0):  # 2006/6/30
        return None
    # else:
    #    u = u / norm(u)
    v = np.array([-u[1], u[0]])
    hcs = np.array([[u[0], v[0], a[0]], [u[1], v[1], a[1]], [0.0, 0.0, 1.0]])
    return hcs


# def cs_transform(from_cs, to_cs, point):
#    """transform a point from from_cs to to_cs"""
#    transform = from_cs.mmul(to_cs.inverse())
#    hpoint = Vec(point)
#    hpoint.append(1.0)
#    hres = transform.mmul(hpoint)
#    res = np.array(hres[1:-1]) / hres[-1]
#    return res

def translate_2D(dx, dy):
    mat = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0]])
    return mat


def rotate_2D(angl):
    mat = np.array([[np.cos(angl), -np.sin(angl), 0.0],
        [np.sin(angl), np.cos(angl), 0.0],
        [0.0, 0.0, 1.0]])
    return mat


def translate_3D(dx, dy, dz):
    mat = np.array([
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
        [0.0, 0.0, 0.0, 1.0]])
    return mat


def scale_3D(sx, sy, sz):
    mat = np.array([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return mat


def uniform_scale_3D(scale):
    mat = np.array([
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return mat
