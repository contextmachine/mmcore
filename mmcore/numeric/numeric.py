from scipy.optimize import minimize, fsolve

from mmcore.geom.vec import norm_sq, cross, norm, unit, gram_schmidt

import numpy as np

from mmcore.numeric.fdm import FDM


def plane_on_curve(O, T, D2):
    N = unit(gram_schmidt(T, D2))
    B = cross(T, N)
    return np.array([O, T, N, B])


def normal_at(D1, D2):
    N = unit(gram_schmidt(unit(D1), D2))
    return N


def minimize_all(fun, bounds: tuple, tol=1e-5, step=0.001, **kwargs):
    bounds = np.array(bounds)
    ress = []
    funs = []

    def bb(bnds):
        nonlocal ress
        ends = bnds[:, -1]
        starts = bnds[:, 0]
        if (np.abs(ends[0] - starts[0])) <= step:
            return
        else:

            res = minimize(fun, x0=starts, bounds=bnds, **kwargs)
            print(res)
            if res.fun <= tol:
                ress.append(res.x)
                funs.append(res.fun)

            bnds[0, 0] = res.x[0] + (bnds[0, 1] - res.x[0]) / 2

            bb(bnds)

    bb(np.copy(bounds))
    return np.array(ress), np.array(funs)




def intersectiont_point(crv1, crv2, tol=1e-4, step=0.001):
    def fun(t):
        return norm(crv1(t[0]) - crv2(t[1]))

    return minimize_all(fun, bounds=(crv1.interval(), crv2.interval()), tol=tol, step=step)


def evaluate_tangent(D1, D2):
    d1 = np.linalg.norm(D1)
    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        T = D2 / d1 if d1 > 0.0 else np.zeros(D2.shape)
    else:
        T = D1 / d1
    return T, bool(d1)


evaluate_tangent_vec = np.vectorize(evaluate_tangent, signature='(i),(i)->(i),()')

from scipy.integrate import quad


def evaluate_length(first_der, t0: float, t1: float, **kwargs):
    """
  """

    def ds(t):
        return norm(first_der(t))

    return quad(ds, t0, t1, **kwargs)


evaluate_length_vec = np.vectorize(evaluate_length, excluded=[0], signature='(),()->(),()')


def evaluate_curvature(D1, D2) -> tuple[np.ndarray, np.ndarray, bool]:
    d1 = np.linalg.norm(D1)

    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        if d1 > 0.0:
            T = D2 / d1
        else:
            T = np.zeros_like(D2)
        K = np.zeros_like(D2)
        rc = False
    else:
        T = D1 / d1
        negD2oT = -np.dot(D2, T)
        d1 = 1.0 / (d1 * d1)
        K = d1 * (D2 + negD2oT * T)
        rc = True

    return T, K, rc


evaluate_curvature_vec = np.vectorize(evaluate_curvature, signature='(i),(i)->(i),(i),()')


def evaluate_jacobian(ds_o_ds, ds_o_dt, dt_o_dt):
    a = ds_o_ds * dt_o_dt
    b = ds_o_dt * ds_o_dt
    det = a - b;
    if ds_o_ds <= dt_o_dt * np.finfo(float).eps or dt_o_dt <= ds_o_ds * np.finfo(float).eps:
        # One of the partials is (numerically) zero w.r.t. the other partial - value of det is unreliable
        rc = False
    elif abs(det) <= max(a, b) * np.sqrt(np.finfo(float).eps):
        # Du and Dv are (numerically) (anti) parallel - value of det is unreliable.
        rc = False
    else:
        rc = True

    return det, rc


def evaluate_normal(Du, Dv, Duu, Duv, Dvv, limit_dir=None):
    r"""

bool
ON_EvNormal(int limit_dir,
                const ON_3dVector& Du, const ON_3dVector& Dv,
                const ON_3dVector& Duu, const ON_3dVector& Duv, const ON_3dVector& Dvv,
                ON_3dVector& N)
{
  const double DuoDu = Du.LengthSquared();
  const double DuoDv = Du*Dv;
  const double DvoDv = Dv.LengthSquared();
  if ( ON_EvJacobian(DuoDu,DuoDv,DvoDv,nullptr) ) {
    N = ON_CrossProduct(Du,Dv);
  }
  else {
    /* degenerate jacobian - try to compute normal using limit
     *
     * P,Du,Dv,Duu,Duv,Dvv = srf and partials evaluated at (u0,v0).
     * Su,Sv,Suu,Suv,Svv = partials evaluated at (u,v).
     * Assume that srf : R^2 -> R^3 is analytic near (u0,v0).
     *
     * srf(u0+u,v0+v) = srf(u0,v0) + u*Du + v*Dv
     *                  + (1/2)*u^2*Duu + u*v*Duv + (1/2)v^2*Dvv
     *                  + cubic and higher order terms.
     *
     * Su X Sv = Du X Dv + u*(Du X Duv + Duu X Dv) + v*(Du X Dvv + Duv X Dv)
     *           + quadratic and higher order terms
     *
     * Set
     * (1) A = (Du X Duv + Duu X Dv),
     * (2) B = (Du X Dvv + Duv X Dv) and assume
     * Du X Dv = 0.  Then
     *
     * |Su X Sv|^2 = u^2*AoA + 2uv*AoB + v^2*BoB + cubic and higher order terms
     *
     * If u = a*t, v = b*t and (a^2*AoA + 2ab*AoB + b^2*BoB) != 0, then
     *
     *        Su X Sv                   a*A + b*B
     * lim   ---------  =  ----------------------------------
     * t->0  |Su X Sv|      sqrt(a^2*AoA + 2ab*AoB + b^2*BoB)
     *
     * All I need is the direction, so I just need to compute a*A + b*B as carefully
     * as possible.  Note that
     * (3)  a*A + b*B = Du X (a*Duv + b*Dvv)  - Dv X (a*Duu + b*Duv).
     * Formaula (3) requires fewer flops than using formulae (1) and (2) to
     * compute a*A + b*B.  In addition, when |Du| << |Dv| or |Du| >> |Dv|,
     * formula (3) reduces the number of subtractions between numbers of
     * similar size.  Since the (nearly) zero first partial is the most common
     * is more common than the (nearly) (anti) parallel case, I'll use
     * formula (3).  If you're reading this because you're not getting
     * the right answer and you can't find any bugs, you might want to
     * try using formulae (1) and (2).
     *
     * The limit_dir argument determines which direction is used to compute the
     * limit.
     *                  |
     *   limit_dir == 2 |  limit_dir == 1
     *           \      |      /
     *            \     |     /
     *             \    |    /
     *              \   |   /
     *               \  |  /
     *                \ | /
     *                 \|/
     *   ---------------*--------------
     *                 /|\
     *                / | \
     *               /  |  \
     *              /   |   \
     *             /    |    \
     *            /     |     \
     *           /      |      \
     *   limit_dir == 3 |  limit_dir == 4
     *                  |
     *
     */

    double a,b;
    ON_3dVector V, Au, Av;

    switch(limit_dir) {
    case  2: /* from 2nd  quadrant to point */
      a = -1.0; b =  1.0; break;
    case  3: /* from 3rd  quadrant to point */
      a = -1.0; b = -1.0; break;
    case  4: /* from 4rth quadrant to point */
      a =  1.0; b = -1.0; break;
    default: /* from 1rst quadrant to point */
      a =  1.0; b =  1.0; break;
    }

    V = a*Duv + b*Dvv;
    Av.x = Du.y*V.z - Du.z*V.y;
    Av.y = Du.z*V.x - Du.x*V.z;
    Av.z = Du.x*V.y - Du.y*V.x;

    V = a*Duu + b*Duv;
    Au.x = V.y*Dv.z - V.z*Dv.y;
    Au.y = V.z*Dv.x - V.x*Dv.z;
    Au.z = V.x*Dv.y - V.y*Dv.x;

    N = Av + Au;
  }

  return N.Unitize();
}

    :param limit_dir:
    :param Du:
    :param Dv:
    :param Duu:
    :param Duv:
    :param Dvv:
    :return:
    """
    DuoDu = norm_sq(Du);

    DuoDv = Du * Dv;

    DvoDv = norm_sq(Dv)
    det, success = evaluate_jacobian(DuoDu, DuoDv, DvoDv)
    if success:
        return np.cross(Du, Dv)
    else:
        a, b = {
            2: [-1.0, 1.0],
            3: [-1.0, -1.0],
            4: [1.0, -1.0],
        }.get(limit_dir, [1.0, 1.0])
        V = a * Duv + b * Dvv
        Av = cross(Du, V)
        V = a * Duu + b * Duv
        Au = cross(V, Dv)
        N = Av + Au
        N = N / np.linalg.norm(N)
        return N
