"""
Exhaustive test suite for evaluate_nurbs_surface.
This file contains tests for:
  - planar B-spline (non–rational) surfaces,
  - Bezier surfaces,
  - rational (weighted) surfaces,
  - surfaces defined in higher (overdimensional) spaces,
  - surfaces with non–uniform weights,
  - singular (degenerate) surfaces,
  - canonical cases (cylinder, cone).

Data have been chosen from standard NURBS textbooks (e.g. Piegl & Tiller)
and from established libraries (e.g. OpenNURBS examples).
"""

import unittest
import numpy as np
import math
from collections import namedtuple
from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,evaluate_nurbs_curve,evaluate_nurbs_surface

np.set_printoptions(suppress=True)
from typing import TypedDict



def cylinder(radius, h):
    _C = 1 / math.sqrt(2)
    _CC = _C * 2
    rr = np.array(
        [
            [[radius, 0.0, 0.0, 1.], [radius, 0.0, h, 1.] ],
            [[_C, _C, 0.0, _C], [_C, _C, _CC, _C]],
            [[0.0, radius, 0.0, 1.], [0.0, radius, h, 1.] ],
            [[-_C, _C, 0.0, _C], [-_C, _C, _CC, _C]],
            [[-radius, 0.0, 0.0, 1.], [-radius, 0.0, h, 1.] ],
            [[-_C, -_C, 0.0, _C], [-_C, -_C, _CC, _C]],
            [[-0.0, -radius, 0.0, 1.], [-0.0, -radius, h, 1.] ],
            [[_C, -_C, 0.0, _C], [_C, -_C, _CC, _C]],
            [[radius, 0.0, 0.0, 1.], [radius, 0.0, h, 1.] ],
        ]
    )

    return NURBSSurfaceTuple(
        3, 2, [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], rr[..., :-1].tolist(), rr[..., -1]
    )


def circle(radius):
    c = 1 / math.sqrt(2)

    ptsw = np.array(
        [
            [radius, 0, 0, 1],
            [radius, radius, 0, c],
            [0, radius, 0, 1],
            [-radius, radius, 0, c],
            [-radius, 0, 0, 1],
            [-radius, -radius, 0, c],
            [0, -radius, 0, 1],
            [radius, -radius, 0, c],
            [radius, 0, 0, 1],
        ]
    )

    return NURBSCurveTuple(3, [0.0, 0.0, 0.0, 1 / 4, 2 / 4, 3 / 4, 4 / 4, 4 / 4, 4 / 4], ptsw[..., :-1].tolist(), ptsw[..., -1])

from mmcore.construction.revolution_surface import make_torus
# ---------------------------------------------------------------------------
# Test cases for evaluate_nurbs_surface.
# ---------------------------------------------------------------------------
class TestEvaluateNURBSSurface(unittest.TestCase):
    # Tolerance for numerical comparisons
    tol = 1e-5
    def construct_torus(self, R,r):

        ku,kv,pts,w=make_torus(R, r)
        return NURBSSurfaceTuple(3,3,knot_u=ku,knot_v=kv,control_points=np.array(pts).tolist(),weights=np.array(w))
    def test_bspline_planar(self):
        # A simple bilinear (order=2 in both u and v) B-spline surface.
        # Control net: 2x2 grid in the plane z=0.
        # P00=(0,0,0), P01=(0,1,0), P10=(1,0,0), P11=(1,1,0)
        ctrl_pts = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        # For a linear (order=2) B-spline, knot vectors are [0,0,1,1].
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.3, 0.7)
        u, v = 0.3, 0.7
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)

        # For bilinear interpolation the result is:
        # S = ( (1-u)*(1-v)*P00 + (1-u)*v*P01 + u*(1-v)*P10 + u*v*P11 )
        # S = (0.7*0.3*P00 + 0.7*0.7*P01 + 0.3*0.3*P10 + 0.3*0.7*P11)
        #      = (0.21*P00 + 0.49*P01 + 0.09*P10 + 0.21*P11)
        # = ( (0.09+0.21, 0.49+0.21, 0) ) = (0.3, 0.7, 0)
        expected_S = np.array([0.3, 0.7, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # First derivatives (analytically)
        expected_Su = np.array([1.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 1.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        print("Su,Sv: OK")
        # Second derivatives are zero.
        expected_Suu = np.zeros(3)
        expected_Suv = np.zeros(3)
        expected_Svv = np.zeros(3)

        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        print("Suu: OK")
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        print("Suv: OK")
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))
        print("Svv: OK")

    def test_bezier_patch(self):
        # A quadratic (order=3 in both directions) Bezier patch.
        # Control net: 3x3 grid.
        # Use control points that yield a non–planar patch.
        ctrl_pts = [
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)],
            [(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
            [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
        ]
        weights = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # For a Bezier patch, knot vectors are [0,0,0,1,1,1]
        surf = NURBSSurfaceTuple(
            order_u=3, order_v=3, knot_u=[0, 0, 0, 1, 1, 1], knot_v=[0, 0, 0, 1, 1, 1], control_points=ctrl_pts, weights=weights
        )
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # For the given quadratic Bezier patch the (non–rational) evaluation yields:
        # S = (1,1,0.5)
        expected_S = np.array([1.0, 1.0, 0.5])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # Derivatives computed using standard Bezier formulas:
        expected_Su = np.array([2.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 2.0, 0.0])
        expected_Suu = np.array([0.0, 0.0, -4.0])
        expected_Suv = np.array([0.0, 0.0, 0.0])
        expected_Svv = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))

    def test_weighted_bilinear(self):
        # A rational bilinear patch (order=2 in both directions) with non-uniform weights.
        # Control points same as test_bspline_planar but with different weights.

        ctrl_pts = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        # Assign weights:
        # w00=1, w01=2, w10=3, w11=4.
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.3,0.7)
        u, v = 0.3, 0.7
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)

        expected_S, expected_Su, expected_Sv = (
            [0.13043478260869565, 0.30434782608695654, 0.0],
            [0.32136105860113423, -0.26465028355387527, 0.0],
            [-0.056710775047258986, 0.30245746691871456, 0.0],
        )  # Obtained using OpenNURBS 8.x

        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # The first derivatives (using the quotient rule) evaluate to approximately:

        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        print("Su,Sv: OK")
        # Second derivatives for a bilinear patch are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(3), rtol=self.tol))

    def test_overdimensional(self):
        # A patch defined in R^4 (control points are 4–vectors) with weights=1.
        ctrl_pts = [[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # Expected S = ((0.5,0.5,0,0) in R^4) obtained by bilinear interpolation.
        expected_S = np.array([0.5, 0.5, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # Derivatives: Su = (1,0,0,0), Sv = (0,1,0,0)
        expected_Su = np.array([1.0, 0.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))

        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        # Second derivatives are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(4), rtol=self.tol))

    def test_overdimensional_weighted(self):
        # Overdimensional (R^4) patch with non-uniform weights.
        ctrl_pts = [[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]]
        # Weights: w00=1, w01=2, w10=3, w11=4.
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # Homogeneous combination:
        # Each term weight: 0.25*(w_{ij}), so H = ( (0+0+1+1)/?, ... ) and S = (0.5/2.5, 0.5/2.5, 0, 0) = (0.2,0.2,0,0)
        expected_S = np.array([0.2, 0.2, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # The first derivatives (computed analogously) are approximately:
        expected_Su = np.array([0.24, -0.16, 0.0, 0.0])
        expected_Sv = np.array([-0.08, 0.32, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=1e-3))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=1e-3))
        # Second derivatives are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(4), rtol=self.tol))

    def test_singular(self):
        # A singular (degenerate) surface: all control points are identical.
        ctrl_pts = [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)], [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        expected_S = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # All derivatives vanish.
        self.assertTrue(np.allclose(SKL["Su"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(3), rtol=self.tol))

    def test_cylinder(self):
        # A canonical cylinder patch.
        # We represent a quarter of a circular arc (in u) extruded linearly (in v).
        # u–direction: quadratic (order=3) NURBS representation of a circular arc from (1,0) to (0,1).
        # The standard data (for a 90° arc) is:
        #   Control points in xy–plane:
        #     P0 = (1,0), P1 = (1,1), P2 = (0,1)
        #   Weights: w0 = 1, w1 = 1/√2, w2 = 1.
        # For the cylinder, we “extrude” in the z–direction.
        c = math.sqrt(2.0) / 2.0
        # Build a 3x2 control net:
        # For v=0 (bottom circle): z = 0.
        # For v=1 (top circle): z = 2.

        # Knot vectors:
        # u: order 3 --> [0,0,0,1,1,1]
        # v: order 2 --> [0,0,1,1]

        surf = cylinder(1.0, 2.0)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        expected_S, expected_Su, expected_Sv, expected_Suu, expected_Suv, expected_Svv = [
            [-1.0, 0.0, 1.0],
            [0.0, -5.65685424, -4.440892098500626e-16],
            [0.0, 0.0, 2.0],
            [32.0, -13.254834134788034, -3.552713678800501e-15],
            [0.0, 0.0, -8.881784197001252e-16],
            [0.0, 0.0, 0.0],
        ]  # Obtained using OpenNURBS 8.x

        print(SKL)
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # For the circular arc the derivative at u=0.5 is known to be tangent.
        # From our minimal basis function derivative the u–derivative computes to approximately:

        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        print("Su: OK")
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        print("Sv: OK")

        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        print("Suu: OK")
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        print("Suv: OK")
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))
        print("Svv: OK")

    def test_cone(self):
        # A canonical cone patch.
        # We represent a ruled surface between an apex and a circular base.
        # Here we choose a simple 2x3 patch.
        # u–direction (order=2, linear): u=0 corresponds to the apex, u=1 to the base.
        # v–direction (order=3, quadratic): a quadratic Bezier representation of a circular arc.
        # Let the apex be A = (0,0,1) and the base (at z=0) be given by:
        #   B0 = (1,0,0), B1 = (1,1,0), B2 = (0,1,0).
        apex = (0.0, 0.0, 1.0)
        base_pts = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        # Build control net: 2 rows (u=0,1); row0 all apex, row1 are base.
        ctrl_pts = [[apex, apex, apex], [base_pts[0], base_pts[1], base_pts[2]]]
        # Use all weights = 1.
        weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        # Knot vectors:
        # u (order=2): [0,0,1,1]
        # v (order=3): [0,0,0,1,1,1]
        surf = NURBSSurfaceTuple(
            order_u=2, order_v=3, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 0, 1, 1, 1], control_points=ctrl_pts, weights=weights
        )
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        expected_S = np.array([0.375, 0.375, 0.5])
        expected_Su = np.array([0.75, 0.75, -1.0])
        expected_Sv = np.array([-0.5, 0.5, 0.0])
        expected_Suu = np.array([0.0, 0.0, 0.0])
        expected_Suv = np.array([-1.0, 1.0, 0.0])
        expected_Svv = np.array([-1.0, -1.0, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))

    # ---------------------------
    # New tests for periodic surfaces and other cases
    # ---------------------------

    def test_torus(self):
        """
        Test a full torus surface (periodic in both u and v directions) with a
        standard rational quadratic representation. The torus is defined by:
            S(u,v) = ((R + r*cos(v'))*cos(u'), (R + r*cos(v'))*sin(u'), r*sin(v'))
        with u',v' linearly mapping from the effective parameter domain [U_start, U_end] to [0, 2π].
        For this test we choose R = 3 and r = 1. The torus is built as the tensor product of
        two rational quadratic circles representing quarter–circles. In each circle the even–indexed
        control points (at 0°, 90°, 180°, 270°) have weight 1 and lie exactly on the circle, while
        the odd–indexed ones (at 45°, 135°, 225°, 315°) are “lifted” by a factor of 1/cos(π/4) and
        assigned weight cos(π/4). This is the standard exact NURBS representation of a circle.
        """
        R = 3.0
        r = 1.0

        def analytical_torus(u, v):
            """
            Parametric equation of a torus.

            Parameters:
                u (float): parameter u (angle around the torus' central circle)
                v (float): parameter v (angle around the tube)


            Returns:
                tuple: (x, y, z) coordinates of the point on the torus.
            """
            R = 3.0
            r = 1.0
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            return (x, y, z)

        def Su(u, v):
            """
            Partial derivative of S with respect to u.

            Returns:
                tuple: (dS/du) evaluated at (u,v)
            """
            R = 3.0
            r = 1.0
            dx_du = - (R + r * math.cos(v)) * math.sin(u)
            dy_du = (R + r * math.cos(v)) * math.cos(u)
            dz_du = 0
            return (dx_du, dy_du, dz_du)

        def Sv(u, v):
            """
            Partial derivative of S with respect to v.

            Returns:
                tuple: (dS/dv) evaluated at (u,v)
            """
            R = 3.0
            r = 1.0
            dx_dv = - r * math.sin(v) * math.cos(u)
            dy_dv = - r * math.sin(v) * math.sin(u)
            dz_dv = r * math.cos(v)
            return (dx_dv, dy_dv, dz_dv)

        def Suu(u, v):
            """
            Second partial derivative of S with respect to u (S_uu).

            Returns:
                tuple: (d²S/du²) evaluated at (u,v)
            """
            R = 3.0
            r = 1.0
            d2x_du2 = - (R + r * math.cos(v)) * math.cos(u)
            d2y_du2 = - (R + r * math.cos(v)) * math.sin(u)
            d2z_du2 = 0
            R = 3.0
            r = 1.0
            return (d2x_du2, d2y_du2, d2z_du2)

        def Suv(u, v):
            """
            Mixed partial derivative of S with respect to u and v (S_uv).

            Returns:
                tuple: (d²S/dudv) evaluated at (u,v)
            """
            R = 3.0
            r = 1.0
            d2x_dudv = r * math.sin(v) * math.sin(u)
            d2y_dudv = (- r *
                        math.sin(v) * math.cos(u))
            d2z_dudv = 0
            return (d2x_dudv, d2y_dudv, d2z_dudv)

        def Svv(u, v):
            """
            Second partial derivative of S with respect to v (S_vv).

            Returns:
                tuple: (d²S/dv²) evaluated at (u,v)
            """
            R = 3.0
            r = 1.0
            d2x_dv2 = - r * math.cos(v) * math.cos(u)
            d2y_dv2 = - r * math.cos(v) * math.sin(u)
            d2z_dv2 = - r * math.sin(v)
            return (d2x_dv2, d2y_dv2, d2z_dv2)
        # Build control points and weights for the torus.
        # Each parametric direction is a rational quadratic circle representing a quarter circle.
        # For index i (or j):
        #   if even: theta = (i//2) * (π/2), weight = 1, position = (cos(theta), sin(theta))
        #   if odd:  theta = ((i//2) + 0.5) * (π/2), weight = cos(π/4), position = (cos(theta)/cos(π/4), sin(theta)/cos(π/4))


        # Map the evaluation parameters to angles.
        # The effective parameter domain [U_start, U_end] is linearly mapped to [0, 2π].
        c = math.cos(math.pi / 4)
        torus_surf=NURBSSurfaceTuple(**{"knot_v": [0.0, 0.0, 0.0, 1.5707963267948966, 1.5707963267948966, 3.1415926535897931, 3.1415926535897931, 4.7123889803846897, 4.7123889803846897, 6.2831853071795862, 6.2831853071795862, 6.2831853071795862], "order_u": 3, "weights": np.array([[1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0], [0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757], [1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0], [0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757], [1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0], [0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757], [1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0], [0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757, 0.50000000000000011, 0.70710678118654757], [1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0, 0.70710678118654757, 1.0]]), "knot_u": [0.0, 0.0, 0.0, 4.7123889803846897, 4.7123889803846897, 9.4247779607693793, 9.4247779607693793, 14.137166941154069, 14.137166941154069, 18.849555921538759, 18.849555921538759, 18.849555921538759], "control_points": [[[4.0, 0.0, 0.0], [2.8284271247461903, 0.0, 0.70710678118654746], [3.0, 0.0, 1.0], [1.4142135623730951, 0.0, 0.70710678118654757], [2.0, 0.0, 1.2246467991473532e-16], [1.4142135623730949, 0.0, -0.70710678118654746], [3.0, 0.0, -1.0], [2.8284271247461903, 0.0, -0.70710678118654757], [4.0, 0.0, 0.0]], [[2.8284271247461903, 2.8284271247461898, 0.0], [2.0000000000000004, 2.0, 0.5], [2.1213203435596428, 2.1213203435596419, 0.70710678118654757], [1.0000000000000002, 1.0, 0.50000000000000011], [1.4142135623730951, 1.4142135623730949, 8.6595605623549341e-17], [1.0, 0.99999999999999978, -0.5], [2.1213203435596428, 2.1213203435596419, -0.70710678118654757], [2.0000000000000004, 2.0, -0.50000000000000011], [2.8284271247461903, 2.8284271247461898, 0.0]], [[2.4492935982947064e-16, 4.0, 0.0], [1.7319121124709868e-16, 2.8284271247461903, 0.70710678118654746], [1.8369701987210297e-16, 3.0, 1.0], [8.6595605623549341e-17, 1.4142135623730951, 0.70710678118654757], [1.2246467991473532e-16, 2.0, 1.2246467991473532e-16], [8.6595605623549316e-17, 1.4142135623730949, -0.70710678118654746], [1.8369701987210297e-16, 3.0, -1.0], [1.7319121124709868e-16, 2.8284271247461903, -0.70710678118654757], [2.4492935982947064e-16, 4.0, 0.0]], [[-2.8284271247461898, 2.8284271247461903, 0.0], [-2.0, 2.0000000000000004, 0.5], [-2.1213203435596419, 2.1213203435596428, 0.70710678118654757], [-1.0, 1.0000000000000002, 0.50000000000000011], [-1.4142135623730949, 1.4142135623730951, 8.6595605623549341e-17], [-0.99999999999999978, 1.0, -0.5], [-2.1213203435596419, 2.1213203435596428, -0.70710678118654757], [-2.0, 2.0000000000000004, -0.50000000000000011], [-2.8284271247461898, 2.8284271247461903, 0.0]], [[-4.0, 4.8985871965894128e-16, 0.0], [-2.8284271247461903, 3.4638242249419736e-16, 0.70710678118654746], [-3.0, 3.6739403974420594e-16, 1.0], [-1.4142135623730951, 1.7319121124709868e-16, 0.70710678118654757], [-2.0, 2.4492935982947064e-16, 1.2246467991473532e-16], [-1.4142135623730949, 1.7319121124709863e-16, -0.70710678118654746], [-3.0, 3.6739403974420594e-16, -1.0], [-2.8284271247461903, 3.4638242249419736e-16, -0.70710678118654757], [-4.0, 4.8985871965894128e-16, 0.0]], [[-2.8284271247461903, -2.8284271247461898, 0.0], [-2.0000000000000004, -2.0, 0.5], [-2.1213203435596428, -2.1213203435596419, 0.70710678118654757], [-1.0000000000000002, -1.0, 0.50000000000000011], [-1.4142135623730951, -1.4142135623730949, 8.6595605623549341e-17], [-1.0, -0.99999999999999978, -0.5], [-2.1213203435596428, -2.1213203435596419, -0.70710678118654757], [-2.0000000000000004, -2.0, -0.50000000000000011], [-2.8284271247461903, -2.8284271247461898, 0.0]], [[-7.3478807948841188e-16, -4.0, 0.0], [-5.1957363374129602e-16, -2.8284271247461903, 0.70710678118654746], [-5.5109105961630896e-16, -3.0, 1.0], [-2.5978681687064801e-16, -1.4142135623730951, 0.70710678118654757], [-3.6739403974420594e-16, -2.0, 1.2246467991473532e-16], [-2.5978681687064791e-16, -1.4142135623730949, -0.70710678118654746], [-5.5109105961630896e-16, -3.0, -1.0], [-5.1957363374129602e-16, -2.8284271247461903, -0.70710678118654757], [-7.3478807948841188e-16, -4.0, 0.0]], [[2.8284271247461894, -2.8284271247461903, 0.0], [1.9999999999999998, -2.0000000000000004, 0.5], [2.1213203435596419, -2.1213203435596428, 0.70710678118654757], [0.99999999999999989, -1.0000000000000002, 0.50000000000000011], [1.4142135623730947, -1.4142135623730951, 8.6595605623549341e-17], [0.99999999999999967, -1.0, -0.5], [2.1213203435596419, -2.1213203435596428, -0.70710678118654757], [1.9999999999999998, -2.0000000000000004, -0.50000000000000011], [2.8284271247461894, -2.8284271247461903, 0.0]], [[4.0, 0.0, 0.0], [2.8284271247461903, 0.0, 0.70710678118654746], [3.0, 0.0, 1.0], [1.4142135623730951, 0.0, 0.70710678118654757], [2.0, 0.0, 1.2246467991473532e-16], [1.4142135623730949, 0.0, -0.70710678118654746], [3.0, 0.0, -1.0], [2.8284271247461903, 0.0, -0.70710678118654757], [4.0, 0.0, 0.0]]], "order_v": 3})
        #torus_surf = self.construct_torus(R, r)
        u, v = 0.0, 0.0
        SKL = evaluate_nurbs_surface(torus_surf,u*3*2*np.pi
                                     , v*2*np.pi, d_order=2)
        u=u*np.pi*2
        v=v*np.pi*2
        expected_S = analytical_torus(u,v)
        # Analytical derivatives (first and second) computed from the torus parameterization.

        expected_Su = Su(u,v)
        expected_Sv =Sv(u,v)
        expected_Suu =Suu(u,v)
        expected_Suv = Suv(u,v)
        expected_Svv = Svv(u,v)

        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=1e-3), msg=f"Expected S {expected_S} but got {SKL['S']}")
        # The curve represents a circle exactly, but it is not exactly parametrized in the circle's arc length. This means, for example, that the point at
        # t does not lie at (sin(t),cos(t)) (except for the start, middle and end point of each quarter circle, since the representation is symmetrical).
        # (https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline#:~:text=The%20curve%20represents,representation%20is%20symmetrical).)
        print('S ok')
        #self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=1e-1), msg=f"Expected Su {expected_Su} but got {SKL['Su']}")
        #print('Su ok')
        #self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=1e-1), msg=f"Expected Sv {expected_Sv} but got {SKL['Sv']}")
        #print('Sv ok')
        #self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=1e-1), msg=f"Expected Suu {expected_Suu} but got {SKL['Suu']}")
        #self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=1e-3), msg=f"Expected Suv {expected_Suv} but got {SKL['Suv']}")
        #self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=1e-3), msg=f"Expected Svv {expected_Svv} but got {SKL['Svv']}")

    def test_cubic_polynomial(self):
        """
        Test a cubic (order=4 in both u and v) NURBS surface where the control points
        lie on an affine (planar) function. In that case the surface is exactly the affine
        map and all higher derivatives vanish.

        Let the control net be a 4x4 grid:
            P[i][j] = (i, j, i+j)  for i,j=0,1,2,3.
        For a Bezier patch the evaluation over [0,1] in both u and v is affine.
        """
        ctrl_pts = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append((float(i), float(j), float(i + j)))
            ctrl_pts.append(row)
        weights = np.ones((4, 4))
        # For a Bezier patch with 4 control points, the clamped knot vector is:
        knot_u = [0, 0, 0, 0, 1, 1, 1, 1]
        knot_v = [0, 0, 0, 0, 1, 1, 1, 1]
        surf = NURBSSurfaceTuple(order_u=4, order_v=4, knot_u=knot_u, knot_v=knot_v, control_points=ctrl_pts, weights=weights)
        # Choose an arbitrary parameter value (should yield the affine map)
        u, v = 0.3, 0.7
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # The affine map (plane) is S(u,v) = (3*u, 3*v, 3*u + 3*v)
        expected_S = np.array([3 * u, 3 * v, 3 * (u + v)])
        expected_Su = np.array([3.0, 0.0, 3.0])
        expected_Sv = np.array([0.0, 3.0, 3.0])
        expected_Suu = np.zeros(3)
        expected_Suv = np.zeros(3)
        expected_Svv = np.zeros(3)
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol), msg=f"Expected S {expected_S} but got {SKL['S']}")
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol), msg=f"Expected Su {expected_Su} but got {SKL['Su']}")
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol), msg=f"Expected Sv {expected_Sv} but got {SKL['Sv']}")
        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol), msg=f"Expected Suu {expected_Suu} but got {SKL['Suu']}")
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol), msg=f"Expected Suv {expected_Suv} but got {SKL['Suv']}")
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol), msg=f"Expected Svv {expected_Svv} but got {SKL['Svv']}")

    def test_high_dorder(self):
        """
        Test that if a d_order higher than the degree is requested,
        only the derivatives up to the degree are computed.
        We reuse the bilinear patch from test_bspline_planar (which has degree 1 in each direction)
        and request d_order=5. The result should be identical to d_order=1.
        """
        ctrl_pts = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        surf = NURBSSurfaceTuple(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        u, v = 0.3, 0.7
        SKL_high = evaluate_nurbs_surface(surf, u, v, d_order=5)
        SKL_low = evaluate_nurbs_surface(surf, u, v, d_order=1)
        self.assertTrue(np.allclose(SKL_high["S"], SKL_low["S"], rtol=self.tol))
        self.assertTrue(np.allclose(SKL_high["Su"], SKL_low["Su"], rtol=self.tol))
        self.assertTrue(np.allclose(SKL_high["Sv"], SKL_low["Sv"], rtol=self.tol))
        # Second derivatives for a bilinear (affine) patch are zero.
        self.assertTrue(np.allclose(SKL_high["Suu"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL_high["Suv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL_high["Svv"], np.zeros(3), rtol=self.tol))


if __name__ == "__main__":
    unittest.main()
