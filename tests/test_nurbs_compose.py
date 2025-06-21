"""
Test suite for exact NURBS curve composition.
"""

import numpy as np
import pytest

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve,to_homogeneous_1d
from mmcore.geom._nurbs_compose import (
    compose_nurbs_curves,
    compose_bernstein_polynomials,
    find_composition_breakpoints,
    extract_bezier_from_bspline,
    BezierSegment,
    compose_bezier_segments,
    find_polynomial_roots_in_interval
)
from mmcore.numeric.sbern import compose_curve_curve_sb, bern_to_nurbs_bezier, nurbs_bezier_to_bern



class TestPolynomialRootFinding:
    """Test polynomial root finding utilities."""
    
    def test_linear_roots(self):
        """Test finding roots of linear polynomials."""
        # f(x) = 2x - 1, root at x = 0.5
        coeffs = np.array([-1.0, 2.0])
        roots = find_polynomial_roots_in_interval(coeffs, 0.0, 1.0)
        assert len(roots) == 1
        assert np.allclose(roots[0], 0.5)
    
    def test_quadratic_roots(self):
        """Test finding roots of quadratic polynomials."""
        # f(x) = x^2 - x = x(x-1), roots at 0 and 1
        coeffs = np.array([0.0, -1.0, 1.0])
        roots = find_polynomial_roots_in_interval(coeffs, -0.5, 1.5)
        assert len(roots) == 2
        assert np.allclose(sorted(roots), [0.0, 1.0])
    
    def test_no_roots_in_interval(self):
        """Test polynomial with no roots in given interval."""
        # f(x) = x^2 + 1, no real roots
        coeffs = np.array([1.0, 0.0, 1.0])
        roots = find_polynomial_roots_in_interval(coeffs, -2.0, 2.0)
        assert len(roots) == 0


class TestBernsteinComposition:
    """Test Bernstein polynomial composition."""
    
    def test_identity_composition(self):
        """Test composing with identity function."""
        # Outer: quadratic Bezier
        outer = np.array([1.0, 2.0, 1.5])
        # Inner: identity function (linear from 0 to 1)
        inner = np.array([0.0, 1.0])
        
        composed = compose_bernstein_polynomials(outer, inner)
        
        # Should get back the original quadratic
        assert len(composed) == 3  # degree 2*1 = 2
        assert np.allclose(composed, outer)
    
    def test_constant_composition(self):
        """Test composing with constant function."""
        # Outer: cubic Bezier
        outer = np.array([1.0, 2.0, 3.0, 4.0])
        # Inner: constant at 0.5
        inner = np.array([0.5, 0.5])
        
        composed = compose_bernstein_polynomials(outer, inner)
        
        # Should evaluate outer at 0.5 for all coefficients
        # Cubic Bernstein at 0.5: [0.125, 0.375, 0.375, 0.125]
        expected_value = 0.125 * 1.0 + 0.375 * 2.0 + 0.375 * 3.0 + 0.125 * 4.0
        assert np.allclose(composed, expected_value)
    
    def test_quadratic_composition(self):
        """Test composing two quadratics."""
        # Outer: f(u) = u^2 represented as quadratic Bezier
        # For u^2 on [0,1], Bezier control points are [0, 0, 1]
        outer = np.array([0.0, 0.0, 1.0])
        
        # Inner: g(t) = t^2 represented as quadratic Bezier  
        # For t^2 on [0,1], Bezier control points are [0, 0, 1]
        inner = np.array([0.0, 0.0, 1.0])
        
        composed = compose_bernstein_polynomials(outer, inner)
        
        # Result should be degree 2*2 = 4
        assert len(composed) == 5
        
        # Verify by sampling
        # Composition should give (t^2)^2 = t^4
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            expected = t**4
            
            # Evaluate composed using Bernstein
            n = len(composed) - 1
            from math import comb
            actual = sum(composed[i] * (comb(n, i) * t**i * (1-t)**(n-i)) 
                        for i in range(n+1))
            
            assert np.allclose(actual, expected, atol=1e-10)


class TestBezierExtraction:
    """Test B-spline to Bezier extraction."""
    
    def test_single_span_extraction(self):
        """Test extracting Bezier from single span."""
        # Create a quadratic B-spline with single span
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        control_points = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        weights = np.array([1.0, 1.0, 1.0])
        
        curve = NURBSCurveTuple(
            order=3,
            knot=knots,
            control_points=control_points,
            weights=weights
        )
        
        bezier = extract_bezier_from_bspline(curve, 0.0, 1.0)
        
        assert bezier.degree == 2
        assert np.allclose(bezier.control_points, control_points)
        assert np.allclose(bezier.weights, weights)
    
    def test_multi_span_extraction(self):
        """Test extracting Bezier from curve with multiple spans."""
        # Create a quadratic B-spline with 2 spans
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        control_points = np.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.5], [1.0, 0.0]])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        
        curve = NURBSCurveTuple(
            order=3,
            knot=knots,
            control_points=control_points,
            weights=weights
        )
        
        # Extract first span
        bezier1 = extract_bezier_from_bspline(curve, 0.0, 0.5)
        assert bezier1.degree == 2
        assert bezier1.start_param == 0.0
        assert bezier1.end_param == 0.5
        
        # Extract second span
        bezier2 = extract_bezier_from_bspline(curve, 0.5, 1.0)
        assert bezier2.degree == 2
        assert bezier2.start_param == 0.5
        assert bezier2.end_param == 1.0


class TestFindBreakpoints:
    """Test finding composition breakpoints."""
    
    def test_linear_reparameterization(self):
        """Test with linear reparameterization."""
        # Linear f(t) = 2t maps [0,1] to [0,2]
        f_knots = np.array([0.0, 0.0, 1.0, 1.0])
        f_control = np.array([[0.0], [2.0]])
        f_weights = np.array([1.0, 1.0])
        
        f_curve = NURBSCurveTuple(
            order=2,
            knot=f_knots,
            control_points=f_control,
            weights=f_weights
        )
        
        # C has knots at 0, 0.5, 1, 1.5, 2
        c_knots = np.array([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0])
        
        breakpoints = find_composition_breakpoints(f_curve, c_knots)
        
        # Should find t where f(t) = 0.5, 1.0, 1.5
        # f(t) = 2t, so t = u/2
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert len(breakpoints) == len(expected)
        for bp, exp in zip(breakpoints, expected):
            assert np.allclose(bp, exp, atol=1e-10)


class TestNURBSComposition:
    """Test full NURBS composition algorithm."""

    def test_identity_reparameterization(self):
        """Test composing with identity reparameterization."""
        # Create a quadratic NURBS curve C(u)
        c_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        c_control = np.array([[0.0, 0.0,0], [1.0, 1.0,0], [2.0, 0.0,0]])
        c_weights = np.array([1.0, 2.0, 1.0])  # Rational quadratic

        c_curve = NURBSCurveTuple(
            order=3,
            knot=c_knots,
            control_points=c_control,
            weights=c_weights
        )

        # Identity reparameterization f(t) = t
        f_knots = np.array([0.0, 0.0, 1.0, 1.0])
        f_control = np.array([[0.0], [1.0]])
        f_weights = np.array([1.0, 1.0])

        f_curve = NURBSCurveTuple(
            order=2,
            knot=f_knots,
            control_points=f_control,
            weights=f_weights
        )
        c_bern= nurbs_bezier_to_bern(c_curve)
        f_bern =  nurbs_bezier_to_bern(f_curve)

        # Compose
        composed = bern_to_nurbs_bezier(compose_curve_curve_sb(c_bern, f_bern)
                                        )
        # Verify degree
        assert composed.degree == 2  # 2 * 1 = 2

        # Verify by evaluation - should match original
        for t in np.linspace(0, 1, 11):
            original = evaluate_nurbs_curve(c_curve, t)['C']
            composed_val = evaluate_nurbs_curve(composed, t)['C']
            assert np.allclose(original, composed_val, atol=1e-10)

    def test_quadratic_reparameterization(self):
        """Test composing with quadratic reparameterization."""
        # Linear C(u) = u
        c_knots = np.array([0.0, 0.0, 1.0, 1.0])
        c_control = np.array([[0.0, 0.0,0.], [1.0, 1.0,0.]])
        c_weights = np.array([1.0, 1.0])

        c_curve = NURBSCurveTuple(
            order=2,
            knot=c_knots,
            control_points=c_control,
            weights=c_weights
        )

        # Quadratic f(t) = t^2
        f_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        f_control = np.array([[0.0], [0.0], [1.0]])  # Quadratic from 0 to 1
        f_weights = np.array([1.0, 1.0, 1.0])

        # Adjust control points for exact t^2
        # Quadratic Bezier for t^2: [0, 0, 1]
        f_curve = NURBSCurveTuple(
            order=3,
            knot=f_knots,
            control_points=f_control,
            weights=f_weights
        )
        c_bern = nurbs_bezier_to_bern(c_curve)
        f_bern = nurbs_bezier_to_bern(f_curve)

        # Compose
        composed = bern_to_nurbs_bezier(compose_curve_curve_sb(c_bern, f_bern))

        # Verify degree

        # Verify by evaluation - C(f(t)) = f(t) = t^2
        for t in np.linspace(0, 1, 11):
            expected = np.array([t*t, t*t, 0])
            composed_val = evaluate_nurbs_curve(composed, t)['C']
            assert np.allclose(composed_val, expected, atol=1e-8)

    def test_rational_composition(self):
        """Test composing rational NURBS curves."""
        # Rational quadratic C(u) - circular arc
        c_knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        c_control = np.array([[1.0, 0.0,0.], [1.0, 1.0,0.], [0.0, 1.0,0.]])
        c_weights = np.array([1.0, 1.0/np.sqrt(2), 1.0])

        c_curve = NURBSCurveTuple(
            order=3,
            knot=c_knots,
            control_points=c_control,
            weights=c_weights
        )

        # Linear scaling f(t) = 0.5*t maps [0,1] to [0,0.5]
        f_knots = np.array([0.0, 0.0, 1.0, 1.0])
        f_control = np.array([[0.0], [0.5]])
        f_weights = np.array([1.0, 1.0])

        f_curve = NURBSCurveTuple(
            order=2,
            knot=f_knots,
            control_points=f_control,
            weights=f_weights
        )

        # Compose

        c_bern = nurbs_bezier_to_bern(c_curve)
        f_bern = nurbs_bezier_to_bern(f_curve)

        # Compose
        composed = bern_to_nurbs_bezier(compose_curve_curve_sb(c_bern, f_bern))

        # Verify by evaluation
        for i, t in enumerate(np.linspace(0, 1, 11)):
            # f(t) = 0.5*t
            u = 0.5 * t
            expected = evaluate_nurbs_curve(c_curve, u)['C']
            composed_val = evaluate_nurbs_curve(composed, t)['C']
            if not np.allclose(composed_val, expected, atol=1e-10):
                print(f"\nMismatch at t={t}, u={u}")
                print(f"Expected: {expected}")
                print(f"Got: {composed_val}")
                print(f"Error: {np.linalg.norm(composed_val - expected)}")
            assert np.allclose(composed_val, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
