from fractions import Fraction

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._ssx_planar_cut import exact_source_planar_cut


def sources(radius=2.**-18):
    controls = (0., .5, 1.)
    first = np.empty((3, 3, 4))
    u_square = np.array([.375**2, .375**2-.375, (1-.375)**2])
    v_square = np.array([.625**2, .625**2-.625, (1-.625)**2])
    for i, u in enumerate(controls):
        for j, v in enumerate(controls):
            first[i, j] = u, v, u_square[i]+v_square[j]-radius**2, 1.
    second = np.array([[[u, v, 0., 1.] for v in (-.25, 1.25)]
                       for u in (-.25, 1.25)])
    return first, second


@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_exact_source_cut_counts_original_circle_crossings(axis):
    first, second = sources()
    graph_center = (.375, .625)[axis % 2]
    cut = graph_center if axis < 2 else (graph_center+.25)/1.5
    result = exact_source_planar_cut(first, second, axis, cut, ((0., 1.),)*4,
                                     max_cells=1000, atol=1e-12)
    assert result is not None
    assert result['boundary_topology_complete'], result
    assert len(result['isolated']) == 2
    for root in result['isolated']:
        assert root['stuv'][axis] == cut
        cert = root['source_cut_certificate']
        assert cert['source_ids'] == (id(first), id(second))
        assert cert['kind'] == 'exact_source_planar_cut'
        assert cert['parameter_root_box']
        assert root['point'][2] == pytest.approx(0., abs=1e-15)


def test_source_extremum_is_one_double_root_after_narrow_restriction():
    radius = 2.**-18
    first, second = sources(radius)
    cut = .375+radius
    box = ((cut-radius/16, cut+radius/16), (.625-radius/16, .625+radius/16),
           (0., 1.), (0., 1.))
    result = exact_source_planar_cut(first, second, 0, cut, box, max_cells=1000)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['root_multiplicity'] == 2
    assert root['stuv'][1] == .625
    assert root['parameter_root_box'][1] == (.625, .625)


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_reverse_cut_keeps_exact_nonbinary_graph_pin(direction):
    radius = 2.**-18
    first, second = sources(radius)
    cut = float(np.nextafter((.375+radius+.25)/1.5, direction))
    exact_graph_pin = -Fraction(1, 4)+Fraction(3, 2)*Fraction(cut)
    discriminant = Fraction(radius)**2-(exact_graph_pin-Fraction(3, 8))**2
    result = exact_source_planar_cut(first, second, 2, cut, ((0., 1.),)*4,
                                     max_cells=1000)
    assert result['boundary_topology_complete'], result
    assert len(result['isolated']) == (2 if discriminant > 0 else 0)
    for root in result['isolated']:
        pins = dict(root['source_cut_certificate']['pinned'])
        assert Fraction(*map(int, pins[0])) == exact_graph_pin


def test_target_rectangle_clips_exactly_before_publishing_roots():
    first, second = sources()
    # Target y >= .625 admits just the upper crossing.
    low = Fraction(7, 12)  # (.625+.25)/1.5
    box = ((0., 1.), (0., 1.), (0., 1.), (float(low), 1.))
    result = exact_source_planar_cut(first, second, 0, .375, box, max_cells=1000)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['stuv'][1] > .625


@pytest.mark.parametrize('limit', [0, 1, 50])
def test_source_cut_construction_preflight_honors_budget(limit):
    result = exact_source_planar_cut(*sources(), 0, .375, ((0., 1.),)*4,
                                     max_cells=limit)
    assert result['cells_processed'] <= limit
    assert not result['boundary_topology_complete']


def test_nonaffine_plane_declines_and_identity_cut_retains_source_obligation():
    first, second = sources()
    second[1, 1, 0] += .125
    assert exact_source_planar_cut(first, second, 0, .375, ((0., 1.),)*4) is None
    first, second = sources()
    first[..., 2] = 0.
    result = exact_source_planar_cut(first, second, 0, .375, ((0., 1.),)*4)
    assert not result['boundary_topology_complete']
    assert result['unresolved_source_boxes'][0]['reason'] == 'positive_dimensional_cut'


def test_reverse_cut_skips_nonplanar_bilinear_graph_as_plane_candidate():
    graph = np.array([[[u, v, u*v-.25, 1.] for v in (0., 1.)] for u in (0., 1.)])
    _, plane = sources()
    result = exact_source_planar_cut(graph, plane, 2, .5, ((0., 1.),)*4,
                                     max_cells=1000)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['stuv'] == pytest.approx([.5, .5, .5, .5])


def _registered_cut(identity, first, second, axis, cut):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import BoundaryPoint
    result = exact_source_planar_cut(first, second, axis, cut, ((0., 1.),)*4,
                                     max_cells=1000)
    root, = result['isolated']
    point = BoundaryPoint(stuv=root['stuv'], xyz=root['point'], face=(axis, -1),
                          tangent_raw=np.ones(4), root_box=np.array(root['parameter_root_box']))
    point._source_root_box = True
    assert identity.register_source_root(point, root['source_cut_certificate'])
    return point


def test_distinct_exact_source_roots_do_not_merge_when_float_stuv_aliases():
    from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity
    graph = np.array([[[u, v, (v-.5)*z, 1.] for v in (0., 1.)]
                      for u, z in zip((0., .5, 1.), (1., 1., 2.))])
    plane = np.array([[[u, v, 0., 1.] for v in (0., 1.)] for u in (0., 3.)])
    identity = BoundaryRootIdentity(graph, plane)
    a = _registered_cut(identity, graph, plane, 0, .5)
    b = _registered_cut(identity, graph, plane, 2, 1/6)
    np.testing.assert_array_equal(a.stuv, b.stuv)
    # The plane cut pins graph u=3*float(1/6), not the exact value1/2.
    assert Fraction(3)*Fraction(1/6) != Fraction(1, 2)
    assert not identity(a, b, np.full(4, .001), .001)


def test_registered_temporary_point_cannot_release_its_id_for_another_root():
    import gc
    import weakref
    from mmcore.numeric.intersection.ssx._ssx_root_identity import BoundaryRootIdentity
    graph = np.array([[[u, v, u*v-.25, 1.] for v in (0., 1.)] for u in (0., 1.)])
    _, plane = sources()
    identity = BoundaryRootIdentity(graph, plane)
    point = _registered_cut(identity, graph, plane, 0, .5)
    reference = weakref.ref(point)
    del point
    gc.collect()
    # The registry uses object IDs: retaining the owner prevents stale
    # certificate lookup when CPython recycles a temporary proposal's ID.
    assert reference() is not None


def test_source_representation_work_is_reserved_before_exact_evaluation(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_planar_cut as module
    first, second = sources()
    full = exact_source_planar_cut(first, second, 0, .375, ((0., 1.),)*4,
                                  max_cells=1000)
    assert full['boundary_topology_complete']
    original = module._surface_point
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, '_surface_point', counted)
    limited = exact_source_planar_cut(first, second, 0, .375, ((0., 1.),)*4,
                                     max_cells=full['cells_processed']-1)
    assert not limited['boundary_topology_complete']
    assert limited['cells_processed'] <= full['cells_processed']-1
    assert len(limited['isolated']) == 1
    assert len(calls) == 2  # Only the prepaid first root evaluates both sources.
    assert limited['unresolved_source_boxes']


def test_reverse_cut_uses_exact_affine_identity_with_nonuniform_graph_weights():
    # W=3+3u, P_x=3u*W, P_y=v*W, P_z=(v-.5)*(1+u²)*W.
    # All homogeneous cubic coefficients are exact integers or halves.
    weights = (3., 4., 5., 6.)
    px = (0., 3., 9., 18.)
    zfactor = (3., 4., 6., 12.)
    graph = np.array([[[x, v*w, (v-.5)*z, w] for v in (0., 1.)]
                      for x, w, z in zip(px, weights, zfactor)])
    plane = np.array([[[u, v, 0., 1.] for v in (0., 1.)] for u in (0., 1.)])
    result = exact_source_planar_cut(graph, plane, 2, .5, ((0., 1.),)*4,
                                     max_cells=1000, atol=1e-12)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['stuv'] == pytest.approx([1/6, .5, .5, .5])
    pins = dict(root['source_cut_certificate']['pinned'])
    assert Fraction(*map(int, pins[0])) == Fraction(1, 6)
