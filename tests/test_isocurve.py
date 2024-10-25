import numpy as np
import pytest
from mmcore.numeric.gauss_map import extract_isocurve
from mmcore.geom.nurbs import NURBSSurface, NURBSCurve

@pytest.fixture
def sample_surface():
    """Create a sample NURBS surface for testing"""
    pts = np.array([
        [-25.0, -25.0, -10.0],
        [-25.0, -15.0, -5.0],
        [-25.0, -5.0, 0.0],
        [-25.0, 5.0, 0.0],
        [-25.0, 15.0, -5.0],
        [-25.0, 25.0, -10.0],
        [-15.0, -25.0, -8.0],
        [-15.0, -15.0, -4.0],
        [-15.0, -5.0, -4.0],
        [-15.0, 5.0, -4.0],
        [-15.0, 15.0, -4.0],
        [-15.0, 25.0, -8.0],
        [-5.0, -25.0, -5.0],
        [-5.0, -15.0, -3.0],
        [-5.0, -5.0, -8.0],
        [-5.0, 5.0, -8.0],
        [-5.0, 15.0, -3.0],
        [-5.0, 25.0, -5.0],
        [5.0, -25.0, -3.0],
        [5.0, -15.0, -2.0],
        [5.0, -5.0, -8.0],
        [5.0, 5.0, -8.0],
        [5.0, 15.0, -2.0],
        [5.0, 25.0, -3.0],
        [15.0, -25.0, -8.0],
        [15.0, -15.0, -4.0],
        [15.0, -5.0, -4.0],
        [15.0, 5.0, -4.0],
        [15.0, 15.0, -4.0],
        [15.0, 25.0, -8.0],
        [25.0, -25.0, -10.0],
        [25.0, -15.0, -5.0],
        [25.0, -5.0, 2.0],
        [25.0, 5.0, 2.0],
        [25.0, 15.0, -5.0],
        [25.0, 25.0, -10.0],
    ]).reshape((6, 6, 3))
    return NURBSSurface(pts, (3, 3))

def test_isocurve_directions(sample_surface):
    """Test that extracted isocurves follow the correct directions and match surface points"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    
    # Get surface control points for comparison
    surface_points = sample_surface.control_points[:,:,:3]  # Get 3D points (without weights)
    
    # Test u_min curve (should match first row of surface points)
    u_min_curve = extract_isocurve(sample_surface, u_min, 'u')
    expected_u_min_points = surface_points[0]  # First row
    assert np.allclose(u_min_curve.control_points[:,:3], expected_u_min_points)
    
    # Test u_max curve (should match last row of surface points)
    u_max_curve = extract_isocurve(sample_surface, u_max, 'u')
    expected_u_max_points = surface_points[-1]  # Last row
    assert np.allclose(u_max_curve.control_points[:,:3], expected_u_max_points)
    
    # Test v_min curve (should match first column of surface points)
    v_min_curve = extract_isocurve(sample_surface, v_min, 'v')
    expected_v_min_points = surface_points[:,0]  # First column
    assert np.allclose(v_min_curve.control_points[:,:3], expected_v_min_points)
    
    # Test v_max curve (should match last column of surface points)
    v_max_curve = extract_isocurve(sample_surface, v_max, 'v')
    expected_v_max_points = surface_points[:,-1]  # Last column
    assert np.allclose(v_max_curve.control_points[:,:3], expected_v_max_points)
    
    # Verify curve directions by checking endpoints
    t0, t1 = u_min_curve.interval()
    assert np.allclose(u_min_curve.evaluate(t0), surface_points[0,0])  # u_min curve starts at bottom-left
    assert np.allclose(u_min_curve.evaluate(t1), surface_points[0,-1])  # u_min curve ends at top-left
    
    t0, t1 = v_min_curve.interval()
    assert np.allclose(v_min_curve.evaluate(t0), surface_points[0,0])  # v_min curve starts at bottom-left
    assert np.allclose(v_min_curve.evaluate(t1), surface_points[-1,0])  # v_min curve ends at bottom-right

def test_isocurve_dimensions(sample_surface):
    """Test that extracted isocurves have correct dimensions"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    
    # Test u-direction curve (fixed u, varying v)
    u_curve = extract_isocurve(sample_surface, u_min, 'u')
    assert isinstance(u_curve, NURBSCurve)
    assert np.array(u_curve._control_points).shape == (6, 4)  # 6 points in v-direction, 4D homogeneous coords
    
    # Test v-direction curve (fixed v, varying u)
    v_curve = extract_isocurve(sample_surface, v_min, 'v')
    assert isinstance(v_curve, NURBSCurve)
    assert np.array(v_curve._control_points).shape == (6, 4)  # 6 points in u-direction, 4D homogeneous coords

def test_isocurve_endpoints(sample_surface):
    """Test that isocurves connect properly at surface corners"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    tol = 1e-10
    
    # Extract boundary curves
    u_min_curve = extract_isocurve(sample_surface, u_min, 'u')
    u_max_curve = extract_isocurve(sample_surface, u_max, 'u')
    v_min_curve = extract_isocurve(sample_surface, v_min, 'v')
    v_max_curve = extract_isocurve(sample_surface, v_max, 'v')
    
    # Test curve evaluations at endpoints
    t0_u, t1_u = u_min_curve.interval()
    t0_v, t1_v = v_min_curve.interval()
    
    # Check corner connections
    assert np.allclose(u_min_curve.evaluate(t0_u), v_min_curve.evaluate(t0_v), atol=tol)  # Bottom-left corner
    assert np.allclose(u_min_curve.evaluate(t1_u), v_max_curve.evaluate(t0_v), atol=tol)  # Top-left corner
    assert np.allclose(u_max_curve.evaluate(t0_u), v_min_curve.evaluate(t1_v), atol=tol)  # Bottom-right corner
    assert np.allclose(u_max_curve.evaluate(t1_u), v_max_curve.evaluate(t1_v), atol=tol)  # Top-right corner

def test_isocurve_evaluation_consistency(sample_surface):
    """Test that isocurves evaluate consistently across their parameter range"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    
    # Test u-direction curve at various parameters
    u_curve = extract_isocurve(sample_surface, (u_min + u_max)/2, 'u')
    t0, t1 = u_curve.interval()
    test_params = np.linspace(t0, t1, 5)
    for t in test_params:
        point = u_curve.evaluate(t)
        assert not np.any(np.isnan(point)), f"NaN values found at t={t} in u-direction curve"
    
    # Test v-direction curve at various parameters
    v_curve = extract_isocurve(sample_surface, (v_min + v_max)/2, 'v')
    t0, t1 = v_curve.interval()
    test_params = np.linspace(t0, t1, 5)
    for t in test_params:
        point = v_curve.evaluate(t)
        assert not np.any(np.isnan(point)), f"NaN values found at t={t} in v-direction curve"

def test_isocurve_parameter_validation(sample_surface):
    """Test that isocurve extraction validates parameters correctly"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        extract_isocurve(sample_surface, u_min - 1, 'u')
    
    with pytest.raises(ValueError):
        extract_isocurve(sample_surface, u_max + 1, 'u')
        
    with pytest.raises(ValueError):
        extract_isocurve(sample_surface, v_min - 1, 'v')
        
    with pytest.raises(ValueError):
        extract_isocurve(sample_surface, v_max + 1, 'v')
        
    with pytest.raises(ValueError):
        extract_isocurve(sample_surface, u_min, 'x')  # Invalid direction

def test_isocurve_recreation(sample_surface):
    """Test that recreating curves from extracted properties gives identical results"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    tol = 1e-10
    
    # Test u-direction curve
    original_u = extract_isocurve(sample_surface, u_min, 'u')
    recreated_u = NURBSCurve(original_u.control_points, 
                            degree=original_u.degree,
                            knots=original_u.knots)
    
    t0, t1 = original_u.interval()
    test_params = np.linspace(t0, t1, 5)
    for t in test_params:
        assert np.allclose(original_u.evaluate(t), recreated_u.evaluate(t), atol=tol)
    
    # Test v-direction curve
    original_v = extract_isocurve(sample_surface, v_min, 'v')
    recreated_v = NURBSCurve(original_v.control_points,
                            degree=original_v.degree,
                            knots=original_v.knots)
    
    t0, t1 = original_v.interval()
    test_params = np.linspace(t0, t1, 5)
    for t in test_params:
        assert np.allclose(original_v.evaluate(t), recreated_v.evaluate(t), atol=tol)

def test_isocurve_surface_consistency(sample_surface):
    """Test that isocurve evaluations match surface evaluations at corresponding points"""
    (u_min, u_max), (v_min, v_max) = sample_surface.interval()
    tol = 1e-10
    
    # Test u-direction curve (fixed u, varying v)
    test_u = (u_min + u_max) / 2
    u_curve = extract_isocurve(sample_surface, test_u, 'u')
    t0, t1 = u_curve.interval()
    test_v = np.linspace(v_min, v_max, 5)
    
    for v in test_v:
        # Map parameter from [v_min, v_max] to [t0, t1]
        t = t0 + (t1 - t0) * (v - v_min) / (v_max - v_min)
        curve_point = u_curve.evaluate(t)
        surface_point = sample_surface.evaluate(np.array([test_u, v]))
        assert np.allclose(curve_point, surface_point, atol=tol)
    
    # Test v-direction curve (fixed v, varying u)
    test_v = (v_min + v_max) / 2
    v_curve = extract_isocurve(sample_surface, test_v, 'v')
    t0, t1 = v_curve.interval()
    test_u = np.linspace(u_min, u_max, 5)
    
    for u in test_u:
        # Map parameter from [u_min, u_max] to [t0, t1]
        t = t0 + (t1 - t0) * (u - u_min) / (u_max - u_min)
        curve_point = v_curve.evaluate(t)
        surface_point = sample_surface.evaluate(np.array([u, test_v]))
        assert np.allclose(curve_point, surface_point, atol=tol)